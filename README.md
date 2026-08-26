# CodeMixQA

Research code for code-mixed question answering: dataset preparation, query
segmentation, translation, retrieval, and RAG evaluation.

## Layout

| Path | Contents |
|---|---|
| `english/scripts/` | **Main working directory** — retrieval + RAG experiments. See `english/scripts/README.md`. |
| `english/` | Translation / mock-RAG scripts, input and output CSV/JSON |
| `language_texts/` | Scraped source documents, one `.txt` per URL id |
| `language_texts_translated/` | The same documents translated per language (`bn`, `en`, `es`, `fr`, `hi`, `id`, `it`, `ja`, `ko`, `mr`, `ur`, `zh-cn`) |
| `logs/` | Run logs from the root-level scripts |

## Root-level files

Notebooks and the `.npy` / `.csv` data files they read are kept together at the
top level on purpose — the notebooks reference them by bare relative path
(`doc_embeddings.npy`, `url_id_map.csv`, ...), so moving either would break them.

| File | Role |
|---|---|
| `translate_docs.py` | Translates `language_texts/` into `language_texts_translated/<lang>/` via googletrans |
| `langextract_script.py` | Segments questions into `core_keyword` / `filler` spans with LangExtract |
| `dataset.ipynb`, `doc_analysis.ipynb`, `analysis_of_queries.ipynb` | Dataset and query analysis |
| `cosine_sim.ipynb`, `rag_implementation.ipynb` | Embedding similarity and RAG prototyping |
| `pos_tagging_grouping.ipynb`, `langextract_grouping.ipynb` | Span grouping experiments |
| `eng_translate.ipynb`, `eng_translate_zsp.ipynb` | Translation experiments |
| `*_embeddings.npy`, `*_ids.npy` | Cached query/document embeddings and their id arrays |
| `url_id_map.csv`, `unique_urls.csv` | URL ↔ document-id mapping |

## Environments

`.venv/` at the root and `english/scripts/.venv/` are separate virtualenvs.
`english/scripts/requirements.txt` covers the experiment dependencies.
