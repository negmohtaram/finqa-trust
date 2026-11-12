# Incremental indexing (all-MiniLM-L6-v2, seed 0)

| chunking | unique chunks | cold s | warm encoded | warm s | reindex after 50 edits: encoded | s | stale keys |
|---|---|---|---|---|---|---|---|
| fixed | 1679 | 40.76 | 0 | 0.12 | 175 | 3.84 | 174 |
| sentence | 1536 | 30.62 | 0 | 0.04 | 64 | 1.32 | 62 |
| table_whole | 1757 | 29.33 | 0 | 0.04 | 63 | 0.81 | 60 |

Chunks re-embedded per edited page (50 pages, "in that year" inserted mid-sentence)

| chunking | where | mean | max | share of the page | pages with more than one |
|---|---|---|---|---|---|
| fixed | random | 3.76 | 9 | 64% | 44 |
| fixed | top | 5.86 | 13 | 100% | 48 |
| fixed | bottom | 1.28 | 4 | 22% | 12 |
| sentence | random | 1.2 | 4 | 22% | 7 |
| sentence | top | 1.28 | 6 | 24% | 5 |
| sentence | bottom | 1.02 | 2 | 19% | 1 |
| table_whole | random | 1.34 | 5 | 22% | 8 |
| table_whole | top | 1.16 | 5 | 19% | 4 |
| table_whole | bottom | 1.02 | 2 | 17% | 1 |

Adding 20 new pages

| chunking | encoded | their chunks | s |
|---|---|---|---|
| fixed | 128 | 128 | 2.64 |
| sentence | 121 | 123 | 2.16 |
| table_whole | 137 | 142 | 2.35 |
