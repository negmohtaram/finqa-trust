Paired comparison on the same 100 questions (model alone = temperature 1.0, one answer).

| approach | correct | 95% CI | vs alone | 95% CI of difference | alone wrong → right | alone right → wrong | calls | tokens | latency serial / parallel (s) | cost per 1,000 questions | cost per 1,000 correct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| alone | 77% | [69%, 85%] |  |  | 0 | 0 | 1 | 1,039 | 1.3 / 1.3 | $0.22 | $0.28 |
| code | 77% | [69%, 85%] | +0% | [+0%, +0%] | 0 | 0 | 1 | 1,039 | 1.3 / 1.3 | $0.22 | $0.28 |
| temp0 | 76% | [67%, 84%] | -1% | [-5%, +2%] | 1 | 2 | 1 | 1,040 | 1.2 / 1.2 | $0.22 | $0.29 |

Answers by the first step that broke:

| approach | correct | sign | extraction | formula | arithmetic | unreadable |
|---|---|---|---|---|---|---|
| alone | 77 | 3 | 10 | 10 | 0 | 0 |
| code | 77 | 3 | 10 | 10 | 0 | 0 |
| temp0 | 76 | 3 | 10 | 9 | 1 | 1 |

What each approach did (fallbacks, ties, unreadable votes):

- code: computed 93, fallback 7
