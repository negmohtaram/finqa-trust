# Reranking the top 50

883 FinQA dev questions, search inside the company, cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2`. Differences are paired on the same questions, in points, and * means the interval excludes 0.

| chunking | candidates | ranking | recall@1 | recall@3 | recall@5 | recall@10 | complete@5 | MRR | recall@1 change | recall@5 change | MRR change | seconds per question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| table_whole | fusion | as searched | 0.509 | 0.821 | 0.887 | 0.944 | 0.840 | 0.727 |  |  |  |  |
| table_whole | fusion | reranked | 0.510 | 0.792 | 0.882 | 0.952 | 0.837 | 0.724 | +0.2 [-2.5, +2.9] | -0.5 [-1.9, +1.0] | -0.4 [-2.1, +1.4] | 2.84 |
| table_whole | dense | as searched | 0.503 | 0.779 | 0.866 | 0.943 | 0.827 | 0.709 |  |  |  |  |
| table_whole | dense | reranked | 0.510 | 0.792 | 0.882 | 0.951 | 0.837 | 0.724 | +0.7 [-2.4, +3.7] | +1.6 [-0.2, +3.5] | +1.5 [-0.7, +3.6] | 2.91 |

## Does the first search method still matter after reranking?

- table_whole, reranked fusion minus reranked dense, recall@1: +0.0 [+0.0, +0.0]
- table_whole, reranked fusion minus reranked dense, recall@5: +0.0 [+0.0, +0.0]
- table_whole, reranked fusion minus reranked dense, mrr: -0.0 [-0.0, +0.0]

## By where the evidence sits (table_whole, fusion, recall@5)

| evidence | questions | as searched | reranked |
|---|---|---|---|
| table | 542 | 0.891 | 0.880 |
| text | 198 | 0.959 | 0.967 |
| both | 143 | 0.771 | 0.773 |
