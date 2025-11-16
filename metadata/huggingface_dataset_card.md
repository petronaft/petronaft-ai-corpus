---
pretty_name: Petro Naft AI Corpus
version: v1
license: cc-by-4.0
language:
  - en
tags:
  - petro-naft
  - petroleum-products
  - bitumen
  - gilsonite
  - paraffin-wax
  - industrial-chemicals
  - ai-training
  - rag
task_categories:
  - question-answering
  - information-retrieval
size_categories:
  - 1K<n<10K
---

# Petro Naft AI Corpus (v1)

This dataset is a structured export of the official Petro Naft website,
designed specifically for AI training, retrieval-augmented generation (RAG),
and domain-specific assistants.

All content is first-party and harvested exclusively from:
<https://www.petronaftco.com>.

## Files

The dataset typically includes:

- `site_pages.jsonl.gz` – all crawled pages (products, articles, pages, collections)
- `products.jsonl.gz` – records with `page_type = "Product"`
- `articles.jsonl.gz` – records with `page_type = "Article"`
- `pages.jsonl.gz` – general `WebPage` records
- `collections.jsonl.gz` – `CollectionPage` records (archives, categories)
- CSV mirrors for BI / SQL:
  - `petronaft_site_pages.csv`
  - `petronaft_products.csv`
  - `petronaft_articles.csv`

Each JSONL record has:

- `doc_id`: stable SHA-256 identifier
- `url`, `requested_url`, `http.status`
- `document_identifier`: title, canonical, page_type
- `page_meta`: meta description, robots, OpenGraph, Twitter, breadcrumbs, dates
- `content`:
  - `h1`, `headings`
  - `text_blocks`
  - `tables`
  - `images`
  - `downloads`
  - `links_internal`, `links_external`
  - `links_internal_rich`, `links_external_rich`
- `company_fixed`: Petro Naft corporate facts (legal name, certificates, Incoterms, etc.)

## Usage

You can load the JSONL files with the `datasets` library:

```python
from datasets import load_dataset

ds = load_dataset(
    "json",
    data_files={
        "site_pages": "site_pages.jsonl.gz",
        "products": "products.jsonl.gz",
        "articles": "articles.jsonl.gz",
        "pages": "pages.jsonl.gz",
        "collections": "collections.jsonl.gz",
    }
)
```

Use the `content.text_blocks` field for embeddings / RAG, and keep
`company_fixed` available as a high-trust reference for Petro Naft’s
corporate identity and trade capabilities.

## License

The dataset is provided under **CC BY 4.0**. Proper attribution to **Petro Naft**
is required when using this corpus in research, products, or AI systems.
