Paired comparison on the same 100 questions (model alone = temperature 1.0, one answer). Runs used the free tier; the price column is Groq's list price for the same tokens.

| approach | correct | 95% CI | vs alone | 95% CI of difference | alone wrong → right | alone right → wrong | calls | tokens per question | tokens per correct answer | latency serial / parallel (s) | list price per 1,000 correct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| alone | 77% | [69%, 85%] |  |  | 0 | 0 | 1 | 1,039 | 1,350 | 1.3 / 1.3 | $0.28 |
| code | 77% | [69%, 85%] | +0% | [+0%, +0%] | 0 | 0 | 1 | 1,039 | 1,350 | 1.3 / 1.3 | $0.28 |
| temp0 | 76% | [67%, 84%] | -1% | [-5%, +2%] | 1 | 2 | 1 | 1,040 | 1,368 | 1.2 / 1.2 | $0.29 |
| vote k=3 | 78% | [69%, 86%] | +1% | [-2%, +4%] | 2 | 1 | 3 | 3,131 | 4,014 | 4.1 / 1.6 | $0.84 |

Answers by the first step that broke:

| approach | correct | sign | extraction | formula | arithmetic | unreadable |
|---|---|---|---|---|---|---|
| alone | 77 | 3 | 10 | 10 | 0 | 0 |
| code | 77 | 3 | 10 | 10 | 0 | 0 |
| temp0 | 76 | 3 | 10 | 9 | 1 | 1 |
| vote k=3 | 78 | 3 | 10 | 9 | 0 | 0 |

What each approach did (fallbacks, ties, unreadable votes):

- code: computed 93, fallback 7
- vote k=3: 1/3, 2 unreadable 1, 2/3 8, 3/3 90, no votes 1
