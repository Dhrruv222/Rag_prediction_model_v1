# Data directory

Place source financial documents here before running `python main.py --ingest`.

Supported formats:

- `.pdf` — page numbers are preserved in chunk metadata.
- `.txt` — plain text research notes, reports, or news articles.

Subfolders are scanned recursively, so you may organize documents by ticker, source, or
date, e.g. `data/AAPL/2024-q4-earnings.pdf`.

Nothing in this directory is committed to version control except this file
(see `.gitignore`).
