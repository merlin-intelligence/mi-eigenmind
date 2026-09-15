"""Graph Explorer page — pyvis HTML graphs + eigenvalue spectrum + ranked chunks."""
from __future__ import annotations

import os
import tempfile

import streamlit as st

from eigenmind.pipelines.rag import GraphExplorer
from eigenmind.ui.auth import (
    check_password,
    list_visible_collections,
)
from eigenmind.ui.components import (
    empty_state,
    get_embedder,
    load_nlp,
    render_sidebar,
    section_header,
)
from eigenmind.ui.styles import apply_global_styles, badge
from eigenmind.vectordb.store import QdrantStore

apply_global_styles()
if not check_password():
    st.stop()
sb = render_sidebar()

section_header("/explore graphs/", "map your insights")

if not sb.is_connected:
    empty_state("⚡", "Qdrant offline.")
    st.stop()

store = QdrantStore(sb.qdrant_host, sb.qdrant_port)
_visible = list_visible_collections(store.list_collections())
col_labels = [label for label, _ in _visible]
col_map = {label: qdrant_name for label, qdrant_name in _visible}

if not col_labels:
    empty_state("📂", "No collections found. Go to <strong>/enrich corpus/</strong> to ingest documents first.")
    st.stop()

col_e1, col_e2 = st.columns([1, 2])
with col_e1:
    collection_name = st.selectbox("collection", col_labels)
    pt_count, _ = store.collection_stats(col_map[collection_name])
    if pt_count:
        st.markdown(
            f'<p style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#8a6a50">'
            f'{pt_count:,} vectors indexed</p>',
            unsafe_allow_html=True,
        )

with col_e2:
    prompt = st.text_area(
        "exploration prompt",
        "what is intellectual humility?",
        height=80,
        help="Enter a concept, question, or assertion to navigate your corpus.",
    )

if st.button("▶ generate graph", type="primary"):
    if not (collection_name and prompt):
        st.warning("Provide a collection and a prompt.")
    else:
        with st.spinner("building knowledge graph…"):
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    explorer = GraphExplorer(load_nlp(), store=store, embedder=get_embedder(sb.selected_device))
                    art = explorer.explore(
                        col_map[collection_name], prompt, output_dir=tmp_dir,
                    )

                    st.success(f"✓ Subgraph built — {len(art.ranked_chunks)} singular chunks identified")

                    tab_graph, tab_methods, tab_eigen, tab_chunks, tab_dl = st.tabs([
                        "🕸 knowledge graph",
                        "🔬 methods graph",
                        "📈 eigenvalues",
                        "📋 ranked chunks",
                        "⬇ downloads",
                    ])

                    with tab_graph:
                        legend_items = "".join(
                            f'<span style="display:inline-flex;align-items:center;gap:0.35rem;'
                            f'margin-right:1rem"><span style="width:0.8rem;height:0.8rem;'
                            f'border-radius:50%;background:{color};display:inline-block"></span>'
                            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                            f'color:#2a1f18">{filename}</span></span>'
                            for filename, color in art.document_colors.items()
                        )
                        st.markdown(
                            f'<div class="info-box" style="line-height:2">'
                            f'<strong style="color:#2a1f18">Documents:</strong>&nbsp; {legend_items}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        with open(art.graph_html, "r", encoding="utf-8") as f:
                            st.components.v1.html(f.read(), height=900, scrolling=True)

                    with tab_methods:
                        st.markdown(
                            f'<div class="info-box"><strong style="color:#2a1f18">Method colors:</strong>&nbsp;'
                            f'{badge("Singular")} eigenvector poles &nbsp;'
                            f'{badge("Hinge")} bridge nodes &nbsp;'
                            f'{badge("Theta")} diverse coverage</div>',
                            unsafe_allow_html=True,
                        )
                        with open(art.methods_html, "r", encoding="utf-8") as f:
                            st.components.v1.html(f.read(), height=900, scrolling=True)

                    with tab_eigen:
                        st.image(art.eigenvalue_png, caption="Eigenvalues of the graph Laplacian")

                    with tab_chunks:
                        for i, ch in enumerate(art.ranked_chunks):
                            badges_html = "".join(badge(m.split(" ")[0]) for m in ch.get("methods", []))
                            label = (f"{i + 1}. {ch['filename']} · chunk {ch['chunk_id']} "
                                     f"· score {ch.get('count', 0)}/3")
                            with st.expander(label):
                                st.markdown(f'<p style="margin-bottom:0.5rem">{badges_html}</p>',
                                            unsafe_allow_html=True)
                                if ch.get("themes"):
                                    st.markdown(
                                        f'<p style="font-family:\'DM Mono\',monospace;'
                                        f'font-size:0.72rem;color:#8a6a50">'
                                        f'themes: {", ".join(ch["themes"])}</p>',
                                        unsafe_allow_html=True,
                                    )
                                st.write(ch["text"])

                    with tab_dl:
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            with open(art.graph_html, "rb") as f:
                                st.download_button("graph .html", f, os.path.basename(art.graph_html), mime="text/html")
                        with c2:
                            with open(art.methods_html, "rb") as f:
                                st.download_button("methods .html", f, os.path.basename(art.methods_html), mime="text/html")
                        with c3:
                            with open(art.eigenvalue_png, "rb") as f:
                                st.download_button("eigenvalues .png", f, os.path.basename(art.eigenvalue_png), mime="image/png")
                        with c4:
                            st.download_button(
                                "analysis .txt",
                                art.summary_text.encode("utf-8"),
                                file_name=f"analysis_{collection_name}.txt",
                                mime="text/plain",
                            )
                        if art.excel_path and os.path.exists(art.excel_path):
                            with open(art.excel_path, "rb") as f:
                                st.download_button(
                                    "graph data .xlsx", f,
                                    os.path.basename(art.excel_path),
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Graph generation failed: {e}")
                st.exception(e)
