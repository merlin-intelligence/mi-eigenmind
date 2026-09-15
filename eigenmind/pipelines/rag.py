"""Graph-based RAG pipeline.

:class:`GraphExplorer` builds the artifacts shown by the Graph Explorer page:
HTML graphs, eigenvalue plot, ranked chunks, Excel export. Does NOT call the LLM.
"""
from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pyvis.network import Network

from eigenmind.config import (
    DOCUMENT_COLORS,
    MAX_CHUNKS,
    METHOD_COLORS,
    NODE_MAX_SIZE,
    NODE_MIN_SIZE,
    PROMPT_NODE_COLOR,
    PROMPT_NODE_SIZE,
)
from eigenmind.core.embeddings import EmbeddingModel
from eigenmind.graph.exploration import explore_graph_with_initial_set
from eigenmind.graph.similarity_graph import SimilarityGraph
from eigenmind.vectordb.store import QdrantStore

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────
#  Graph-Explorer pipeline (HTML + eigenvalues + Excel)
# ───────────────────────────────────────────────

@dataclass
class ExplorerArtifacts:
    """Files and data produced by :meth:`GraphExplorer.explore`."""
    graph_html: str
    eigenvalue_png: str
    summary_text: str
    methods_html: str
    excel_path: str
    ranked_chunks: list[dict] = field(default_factory=list)
    document_colors: dict[str, str] = field(default_factory=dict)


def _generate_filename_from_prompt(prompt: str, extension: str = ".html") -> str:
    short = " ".join(prompt.split()[:5])
    sanitized = re.sub(r"[^\w\s-]", "", short).strip().replace(" ", "_")
    return f"{sanitized}{extension}"


def _scale_node_sizes(degrees: np.ndarray) -> list[int]:
    """Map node degrees to a fixed, readable pixel range regardless of how spread out
    the degree distribution is — an unbounded scale produces oversized nodes when the
    similarity threshold lets a few nodes accumulate very high degree."""
    if len(degrees) == 0:
        return []
    mn, mx = float(np.min(degrees)), float(np.max(degrees))
    if mx == mn:
        return [NODE_MIN_SIZE for _ in degrees]
    span = NODE_MAX_SIZE - NODE_MIN_SIZE
    return [int(NODE_MIN_SIZE + (d - mn) / (mx - mn) * span) for d in degrees]


def _extract_themes(text: str, nlp, top_n: int = 5) -> list[str]:
    doc = nlp(text)
    keywords = [
        token.text.lower() for token in doc
        if token.pos_ in ("NOUN", "PROPN")
        and not token.is_stop and not token.is_punct
        and len(token.text) > 2
    ]
    return [item for item, _ in Counter(keywords).most_common(top_n)]


_GRAPH_OPTIONS = {
    "nodes": {
        "font": {
            "size": 18,
            "face": "arial",
            "color": "#f5f5f5",
            "strokeWidth": 3,
            "strokeColor": "#1a1a1a",
        },
        "borderWidth": 2,
        "borderWidthSelected": 4,
    },
    "edges": {
        "color": {"color": "#5a5a5a", "opacity": 0.45, "inherit": False, "highlight": "#ffd700"},
        "width": 0.6,
        "smooth": {"type": "continuous"},
        "hoverWidth": 1.5,
    },
    "physics": {
        "solver": "barnesHut",
        "barnesHut": {
            "gravitationalConstant": -18000, "centralGravity": 0.25,
            "springLength": 160, "springConstant": 0.03,
            "damping": 0.15, "avoidOverlap": 0.6,
        },
        "stabilization": {"iterations": 1000, "fit": True},
    },
    "interaction": {
        "hover": True,
        "tooltipDelay": 150,
        "navigationButtons": True,
        "keyboard": {"enabled": True},
    },
    "layout": {"improvedLayout": True},
}


class GraphExplorer:
    """Build the artifacts shown by the Graph Explorer page (HTMLs, plot, ranking, Excel)."""

    def __init__(
        self,
        nlp,
        store: QdrantStore | None = None,
        embedder: EmbeddingModel | None = None,
        device: str | None = None,
    ):
        self.nlp = nlp
        self.store = store or QdrantStore()
        self._owns_embedder = embedder is None
        self.embedder = embedder or EmbeddingModel(device=device)

    def explore(
        self,
        collection_name: str,
        prompt: str,
        output_dir: str = "temp_graph_outputs",
    ) -> ExplorerArtifacts:
        logger.info("Graph explore collection=%r prompt=%r", collection_name, prompt[:80])
        os.makedirs(output_dir, exist_ok=True)

        query_vector = self.embedder.encode_query(prompt).tolist()
        retrieved, initial_point_ids = explore_graph_with_initial_set(
            collection_name, query_vector, self.store.client, MAX_CHUNKS,
        )
        graph = SimilarityGraph(retrieved)

        # 1. Main graph HTML
        main_html, document_colors = self._render_main_graph(graph, prompt, initial_point_ids, output_dir)

        # 2. Eigenvalue analysis
        base = os.path.join(output_dir, _generate_filename_from_prompt(prompt, ""))
        eig_path, _ev_summary, singular_info = graph.eigenvalue_analysis(base)
        singular_indices = set(singular_info.keys())

        # 3. Hinge ranking
        ranking, _, _, _, _ = graph.hinge_ranking(pole_quantile=0.90)
        hinge_indices = [r[0] for r in ranking[:10]]

        # 4. Theta diversity
        theta_indices = graph.theta_diversity(k=10)

        # 5. Methods graph HTML
        methods_html = self._render_methods_graph(
            graph, prompt, initial_point_ids,
            singular_indices, hinge_indices, theta_indices,
            output_dir,
        )

        if self._owns_embedder:
            self.embedder.release()

        # 6. Consensus ranking
        ranked_chunks = self._consensus_ranking(graph, singular_info, hinge_indices, theta_indices)
        summary = self._format_summary(ranked_chunks)

        # 7. Excel export
        excel_path = self._export_excel(graph, singular_indices, hinge_indices, theta_indices, prompt, output_dir)

        return ExplorerArtifacts(
            graph_html=main_html,
            eigenvalue_png=eig_path,
            summary_text=summary,
            methods_html=methods_html,
            excel_path=excel_path,
            ranked_chunks=ranked_chunks,
            document_colors=document_colors,
        )

    # ─── private rendering helpers ─────────────────────────────────

    def _render_main_graph(
        self, graph: SimilarityGraph, prompt: str, initial_ids: set, output_dir: str,
    ) -> tuple[str, dict[str, str]]:
        path = os.path.join(output_dir, _generate_filename_from_prompt(prompt, ".html"))
        net = Network(height="900px", width="100%", notebook=False, directed=False,
                      bgcolor="#222222", font_color="white")
        prompt_node_id = -1
        net.add_node(prompt_node_id, label="PROMPT", title=prompt, shape="box",
                     color=PROMPT_NODE_COLOR, size=PROMPT_NODE_SIZE)

        unique_files = sorted({p.payload.get("filename") for p in graph.points})
        file_color = {fn: DOCUMENT_COLORS[i % len(DOCUMENT_COLORS)] for i, fn in enumerate(unique_files)}
        degrees = np.sum(graph.W > 0, axis=1)
        sizes = _scale_node_sizes(degrees)

        node_ids, labels, titles, colors, prompt_edges = [], [], [], [], []
        for i, pid in enumerate(graph.ordered_ids):
            p = graph.id_to_point[pid]
            node_ids.append(i)
            labels.append(str(i))
            titles.append(f"File: {p.payload.get('filename', 'N/A')}\n\n{p.payload.get('text', 'N/A')}")
            colors.append(file_color.get(p.payload.get("filename"), "#97c2fc"))
            if pid in initial_ids:
                prompt_edges.append((prompt_node_id, i))
        net.add_nodes(node_ids, label=labels, title=titles, color=colors, size=sizes)

        # Prune for visualization: keep top 5 neighbors per node
        n = graph.n
        vis_mask = np.zeros_like(graph.W, dtype=bool)
        for i in range(n):
            nbrs = np.where(graph.W[i] > 0)[0]
            if len(nbrs) > 5:
                top = nbrs[np.argsort(graph.W[i, nbrs])[-5:]]
                vis_mask[i, top] = True
            else:
                vis_mask[i, nbrs] = True
        vis_mask = vis_mask | vis_mask.T
        W_vis = np.where(vis_mask, graph.W, 0)

        net.set_options(json.dumps(_GRAPH_OPTIONS))
        rows, cols = np.where(np.triu(W_vis) > 0)
        edges = list(prompt_edges)
        for r, c in zip(rows, cols):
            edges.append((int(r), int(c), float(W_vis[r, c])))
        net.add_edges(edges)
        net.save_graph(path)
        return path, file_color

    def _render_methods_graph(
        self, graph: SimilarityGraph, prompt: str, initial_ids: set,
        singular_indices: set, hinge_indices: list[int], theta_indices: list[int],
        output_dir: str,
    ) -> str:
        path = os.path.join(output_dir, _generate_filename_from_prompt(prompt, "_methods.html"))
        net = Network(height="900px", width="100%", notebook=False, directed=False,
                      bgcolor="#222222", font_color="white")
        prompt_node_id = -1
        net.add_node(prompt_node_id, label="PROMPT", title=prompt, shape="box",
                     color=PROMPT_NODE_COLOR, size=PROMPT_NODE_SIZE)

        relevant = singular_indices | set(hinge_indices) | set(theta_indices)
        for idx in relevant:
            p = graph.id_to_point[graph.ordered_ids[idx]]
            if idx in singular_indices: color = METHOD_COLORS["Singular"]
            elif idx in hinge_indices:  color = METHOD_COLORS["Hinge"]
            elif idx in theta_indices:  color = METHOD_COLORS["Theta"]
            else: color = "#97c2fc"
            methods = []
            if idx in singular_indices: methods.append("Singular")
            if idx in hinge_indices: methods.append("Hinge")
            if idx in theta_indices: methods.append("Theta")
            title = (f"File: {p.payload.get('filename', 'N/A')}\n"
                     f"Method: {', '.join(methods)}\n\n{p.payload.get('text', 'N/A')}")
            net.add_node(int(idx), label=str(idx), title=title, color=color, size=NODE_MAX_SIZE)

        for idx in relevant:
            if graph.ordered_ids[idx] in initial_ids:
                net.add_edge(prompt_node_id, int(idx))
            for other in relevant:
                if idx < other:
                    sim = graph.W[idx, other]
                    if sim > 0:
                        net.add_edge(int(idx), int(other), value=float(sim))
        net.set_options(json.dumps(_GRAPH_OPTIONS))
        net.save_graph(path)
        return path

    def _consensus_ranking(
        self, graph: SimilarityGraph,
        singular_info: dict, hinge_indices: list[int], theta_indices: list[int],
    ) -> list[dict]:
        relevant = set(singular_info.keys()) | set(hinge_indices) | set(theta_indices)
        out = []
        for idx in relevant:
            p = graph.id_to_point[graph.ordered_ids[idx]]
            methods_display = []
            if idx in singular_info:
                methods_display.append(f"Eigenvectors ({', '.join(singular_info[idx])})")
            if idx in hinge_indices: methods_display.append("Hinge")
            if idx in theta_indices: methods_display.append("Theta")
            out.append({
                "filename": p.payload.get("filename", "N/A"),
                "text": p.payload.get("text", "N/A"),
                "chunk_id": p.payload.get("chunk_number", "N/A"),
                "methods": methods_display,
                "count": len(methods_display),
                "themes": _extract_themes(p.payload.get("text", ""), self.nlp),
            })
        out.sort(key=lambda c: c["count"], reverse=True)
        return out

    @staticmethod
    def _format_summary(ranked_chunks: list[dict]) -> str:
        summary = "--- Consolidated Ranking (by Method Consensus) ---\n"
        for i, c in enumerate(ranked_chunks):
            text = " ".join(c["text"].split())
            wrapped = textwrap.fill(text, width=120, initial_indent=" " * 6, subsequent_indent=" " * 6)
            summary += f"\n\n[{i + 1}] Consensus Score: {c['count']}/3 | Methods: {', '.join(c['methods'])}"
            summary += f"\n    Themes: {', '.join(c['themes'])}"
            summary += f"\n    - [{c['chunk_id']} | {c['filename']}]"
            summary += f'\n      "{wrapped.strip()}"'
        return summary

    @staticmethod
    def _export_excel(
        graph: SimilarityGraph,
        singular_indices: set, hinge_indices: list[int], theta_indices: list[int],
        prompt: str, output_dir: str,
    ) -> str:
        path = os.path.join(output_dir, _generate_filename_from_prompt(prompt, ".xlsx"))
        node_rows = []
        for idx, pid in enumerate(graph.ordered_ids):
            p = graph.id_to_point[pid]
            methods = []
            if idx in singular_indices: methods.append("Singular")
            if idx in hinge_indices: methods.append("Hinge")
            if idx in theta_indices: methods.append("Theta")
            sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", str(p.payload.get("text", "")))
            node_rows.append({
                "Node ID": idx,
                "Chunk Number": p.payload.get("chunk_number", "N/A"),
                "Source File": p.payload.get("filename", "N/A"),
                "Selected Methods": ", ".join(methods) if methods else "None",
                "Chunk Text": sanitized,
            })
        edge_rows = []
        rows, cols = np.where(np.triu(graph.W) > 0)
        for r, c in zip(rows, cols):
            edge_rows.append({"Source Node": int(r), "Target Node": int(c), "Weight": float(graph.W[r, c])})

        with pd.ExcelWriter(path) as writer:
            pd.DataFrame(node_rows).to_excel(writer, sheet_name="Nodes", index=False)
            pd.DataFrame(edge_rows).to_excel(writer, sheet_name="Adjacency Matrix", index=False)
        return path
