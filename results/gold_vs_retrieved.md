# Full page against retrieved chunks

55 FinQA questions, 2 runs each, same model and answer format. Retrieved means the top 5 chunks of `fusion` on table_whole, inside the company. Differences are paired on the same questions, in points, and * means the interval excludes 0.

| questions | n | full page | retrieved | retrieved minus full page |
|---|---|---|---|---|
| all | 55 | 0.636 | 0.582 | -5.5 [-17.3, +6.4] |
| all evidence in the top 5 | 46 | 0.641 | 0.652 | +1.1 [-10.9, +13.0] |
| some evidence missing | 9 | 0.611 | 0.222 | -38.9 [-72.2, -11.1] * |

## Same question and run

| full page | retrieved | answers |
|---|---|---|
| right | right | 53 |
| right | wrong | 17 |
| wrong | right | 11 |
| wrong | wrong | 29 |

## What broke, share of answers

| label | full page | retrieved |
|---|---|---|
| correct | 0.636 | 0.582 |
| extraction | 0.164 | 0.264 |
| format | 0.000 | 0.009 |
| formula | 0.164 | 0.118 |
| sign | 0.036 | 0.027 |
