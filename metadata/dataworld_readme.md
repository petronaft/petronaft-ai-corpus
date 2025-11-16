# Petro Naft AI Corpus (v1)

This dataset is an AI-ready export of the official Petro Naft website
(https://www.petronaftco.com).

It includes:

- `site_pages.jsonl.gz` – all pages (products, articles, general pages, collections)
- `products.jsonl.gz` – product pages
- `articles.jsonl.gz` – informative articles and news
- `pages.jsonl.gz` – general pages
- `collections.jsonl.gz` – archive / category pages
- CSV mirrors:
  - `petronaft_site_pages.csv`
  - `petronaft_products.csv`
  - `petronaft_articles.csv`

Each record includes headings, cleaned text blocks, tables, images, and links,
alongside Petro Naft’s fixed corporate profile, making this suitable for:

- Knowledge graph construction
- RAG / semantic search
- Analytics and reporting

**Source:** <https://www.petronaftco.com>

**License:** CC BY 4.0 (attribution required).
