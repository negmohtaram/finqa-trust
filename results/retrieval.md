# Retrieval comparison

883 FinQA dev questions, the same ones in every row, with evidence from `gold_inds`. A table row counts only when its header is in the same chunk. Intervals are bootstrap over questions, 95%; differences are paired on the same questions, in points, and * means the interval excludes 0.

- **recall@k**: share of a question's evidence in the top k chunks.
- **complete@5**: all of it in the top 5.
- **MRR**: 1 / rank of the first chunk with evidence, top 10 only.
- **fusion**: reciprocal rank fusion of bm25 and dense, k = 60.

## Search inside the question's company

| chunking | method | recall@1 | recall@3 | recall@5 (95% interval) | recall@10 | complete@5 | MRR | words in top 5 | recall@5 vs dense | MRR vs dense |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | bm25 | 0.607 | 0.770 | 0.832 [0.811, 0.854] | 0.859 | 0.780 | 0.759 | 711 | +6.4 [+4.7, +8.3] * | +8.8 [+6.9, +10.8] * |
| fixed | dense | 0.498 | 0.708 | 0.768 [0.743, 0.793] | 0.832 | 0.713 | 0.671 | 700 |  |  |
| fixed | fusion | 0.572 | 0.763 | 0.821 [0.797, 0.843] | 0.857 | 0.770 | 0.734 | 709 | +5.3 [+3.9, +6.8] * | +6.3 [+5.0, +7.7] * |
| fixed | dense_mpnet | 0.455 | 0.689 | 0.762 [0.737, 0.787] | 0.830 | 0.707 | 0.638 | 702 | -0.6 [-2.3, +1.1] | -3.3 [-5.1, -1.3] * |
| fixed | fusion_mpnet | 0.557 | 0.761 | 0.820 [0.798, 0.843] | 0.856 | 0.769 | 0.726 | 709 | +5.3 [+3.7, +6.9] * | +5.5 [+3.7, +7.3] * |
| sentence | bm25 | 0.610 | 0.768 | 0.811 [0.788, 0.834] | 0.833 | 0.749 | 0.758 | 637 | +4.5 [+2.9, +6.2] * | +8.5 [+6.5, +10.4] * |
| sentence | dense | 0.513 | 0.698 | 0.766 [0.742, 0.791] | 0.813 | 0.707 | 0.674 | 631 |  |  |
| sentence | fusion | 0.566 | 0.751 | 0.805 [0.782, 0.829] | 0.830 | 0.745 | 0.728 | 636 | +3.8 [+2.6, +5.1] * | +5.4 [+4.2, +6.6] * |
| sentence | dense_mpnet | 0.480 | 0.704 | 0.757 [0.731, 0.782] | 0.811 | 0.693 | 0.655 | 631 | -0.9 [-2.7, +0.8] | -1.8 [-3.7, +0.0] |
| sentence | fusion_mpnet | 0.572 | 0.763 | 0.809 [0.786, 0.833] | 0.837 | 0.745 | 0.734 | 638 | +4.3 [+2.8, +5.8] * | +6.0 [+4.2, +7.8] * |
| table_whole | bm25 | 0.515 | 0.786 | 0.866 [0.845, 0.886] | 0.922 | 0.824 | 0.720 | 540 | +0.0 [-2.1, +2.1] | +1.1 [-1.1, +3.3] |
| table_whole | dense | 0.503 | 0.779 | 0.866 [0.844, 0.887] | 0.943 | 0.827 | 0.709 | 504 |  |  |
| table_whole | fusion | 0.509 | 0.821 | 0.887 [0.868, 0.906] | 0.944 | 0.840 | 0.727 | 528 | +2.1 [+0.5, +3.7] * | +1.8 [+0.3, +3.3] * |
| table_whole | dense_mpnet | 0.505 | 0.778 | 0.879 [0.860, 0.899] | 0.941 | 0.838 | 0.706 | 506 | +1.3 [-0.6, +3.1] | -0.3 [-2.4, +1.6] |
| table_whole | fusion_mpnet | 0.497 | 0.811 | 0.896 [0.877, 0.915] | 0.948 | 0.860 | 0.717 | 532 | +3.1 [+1.2, +4.7] * | +0.8 [-1.2, +2.6] |

## Search over all companies

| chunking | method | recall@1 | recall@3 | recall@5 (95% interval) | recall@10 | complete@5 | MRR | words in top 5 | recall@5 vs dense | MRR vs dense |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | bm25 | 0.499 | 0.648 | 0.710 [0.682, 0.738] | 0.767 | 0.661 | 0.640 | 729 | +19.9 [+17.1, +22.8] * | +22.5 [+19.7, +25.0] * |
| fixed | dense | 0.273 | 0.436 | 0.511 [0.481, 0.543] | 0.600 | 0.463 | 0.415 | 706 |  |  |
| fixed | fusion | 0.423 | 0.584 | 0.667 [0.638, 0.697] | 0.730 | 0.617 | 0.571 | 724 | +15.7 [+13.2, +18.1] * | +15.6 [+13.8, +17.4] * |
| fixed | dense_mpnet | 0.255 | 0.419 | 0.483 [0.453, 0.515] | 0.558 | 0.435 | 0.391 | 717 | -2.7 [-5.5, +0.0] | -2.4 [-4.7, -0.1] * |
| fixed | fusion_mpnet | 0.391 | 0.566 | 0.626 [0.598, 0.656] | 0.713 | 0.574 | 0.542 | 726 | +11.5 [+8.8, +14.3] * | +12.7 [+10.4, +14.8] * |
| sentence | bm25 | 0.487 | 0.642 | 0.686 [0.658, 0.714] | 0.751 | 0.625 | 0.630 | 659 | +16.3 [+13.4, +19.3] * | +19.6 [+17.2, +22.1] * |
| sentence | dense | 0.291 | 0.452 | 0.523 [0.492, 0.555] | 0.597 | 0.467 | 0.434 | 644 |  |  |
| sentence | fusion | 0.408 | 0.574 | 0.642 [0.613, 0.672] | 0.714 | 0.586 | 0.558 | 656 | +11.9 [+9.7, +14.2] * | +12.4 [+10.8, +14.0] * |
| sentence | dense_mpnet | 0.265 | 0.413 | 0.475 [0.444, 0.506] | 0.569 | 0.424 | 0.395 | 649 | -4.8 [-7.3, -2.2] * | -3.9 [-6.1, -1.7] * |
| sentence | fusion_mpnet | 0.371 | 0.567 | 0.640 [0.611, 0.670] | 0.705 | 0.581 | 0.532 | 658 | +11.7 [+9.2, +14.3] * | +9.8 [+7.8, +12.0] * |
| table_whole | bm25 | 0.402 | 0.603 | 0.665 [0.635, 0.694] | 0.738 | 0.608 | 0.568 | 551 | +10.7 [+7.6, +13.9] * | +10.3 [+7.8, +12.8] * |
| table_whole | dense | 0.321 | 0.487 | 0.558 [0.527, 0.589] | 0.654 | 0.510 | 0.465 | 460 |  |  |
| table_whole | fusion | 0.382 | 0.591 | 0.661 [0.633, 0.691] | 0.754 | 0.612 | 0.548 | 516 | +10.3 [+7.9, +12.7] * | +8.3 [+6.5, +9.9] * |
| table_whole | dense_mpnet | 0.302 | 0.495 | 0.568 [0.537, 0.599] | 0.654 | 0.522 | 0.452 | 483 | +1.0 [-1.7, +3.7] | -1.3 [-3.7, +0.9] |
| table_whole | fusion_mpnet | 0.368 | 0.603 | 0.685 [0.657, 0.715] | 0.778 | 0.639 | 0.545 | 524 | +12.7 [+10.0, +15.6] * | +8.0 [+5.7, +10.2] * |

## Chunking and model, dense, inside the company

- table_whole minus fixed, recall@5: +9.8 [+6.9, +12.8] *
- table_whole minus sentence, recall@5: +10.0 [+6.9, +12.9] *
- mpnet minus MiniLM on table_whole, recall@5: +1.3 [-0.6, +3.1]
- mpnet minus MiniLM on table_whole, MRR: -0.3 [-2.4, +1.6]

## By where the evidence sits (table_whole, inside the company, recall@5)

| evidence | questions | bm25 | dense | fusion |
|---|---|---|---|---|
| table | 542 | 0.851 | 0.882 | 0.891 |
| text | 198 | 0.973 | 0.910 | 0.959 |
| both | 143 | 0.775 | 0.744 | 0.771 |

## Where each method wins (table_whole, inside the company)

**bm25 alone finds more on 25 questions.**
Example: "what is the carrying value of notes due by 2017 ? in millions $ ." (BLK/2013/page_124.pdf-2). Recall@5: bm25 1.00, dense 0.00, fusion 0.75.

**dense alone finds more on 22 questions.**
Example: "what amount of long-term debt due in the next 36 months as of december 31 , 2003 , in millions?" (ETR/2003/page_84.pdf-2). Recall@5: bm25 0.00, dense 1.00, fusion 0.00.

**fusion alone finds more on 3 questions.**
Example: "what is the total return is $ 100000 are invested in s&p500 on january 1st , 2015 and sold at the end of 2016?" (CDNS/2018/page_31.pdf-1). Recall@5: bm25 0.00, dense 0.00, fusion 1.00.
