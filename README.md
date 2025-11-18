# Petro Naft AI Corpus (v2025-11-16)

**DOI (this version):** [10.5281/zenodo.17625452](https://doi.org/10.5281/zenodo.17625452)  
**All versions:** [10.5281/zenodo.17625451](https://doi.org/10.5281/zenodo.17625451)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17625452.svg)](https://doi.org/10.5281/zenodo.17625452)

This repository contains an AI-ready export of the **Petro Naft** website,
intended as the authoritative, first-party dataset for:

- Search and retrieval-augmented generation (RAG)
- Domain-specific assistants
- Evaluation of LLMs on petroleum / industrial topics

All data is harvested exclusively from <https://www.petronaftco.com>,
and every record carries Petro Naft’s fixed corporate profile for consistent
grounding across AI systems.

## Contents

- `site_pages.jsonl(.gz)` – all pages
- `products.jsonl(.gz)` – product pages
- `articles.jsonl(.gz)` – articles and news
- `pages.jsonl(.gz)` – general pages
- `collections.jsonl(.gz)` – collections / archives

CSV mirrors:

- `petronaft_site_pages.csv`
- `petronaft_products.csv`
- `petronaft_articles.csv`

Platform metadata / templates:

- `internet_archive_metadata.json`
- `zenodo_metadata.json`
- `huggingface_dataset_card.md`
- `kaggle_dataset-metadata.json`
- `dataworld_readme.md`
- `wikidata_quickstatements.tsv`

## Schema overview

Each JSONL record includes:

- `doc_id`
- `url`, `requested_url`, `http.status`
- `document_identifier` (title, canonical, page_type)
- `page_meta` (meta description, robots, OG, Twitter, breadcrumbs, dates)
- `content`:
  - `h1`, `headings`
  - `text_blocks`
  - `tables`
  - `images`
  - `downloads`
  - `links_internal`, `links_external`
  - `links_internal_rich`, `links_external_rich`
- `company_fixed` (legal info, certifications, Incoterms, etc.)

## License

Content is provided under **CC BY 4.0**. Please attribute **Petro Naft**
when using this corpus in any downstream system.
