# Check-set results (100 tickets)

Labelled by: **Claude Code (Opus 5.5), blind to model outputs; row 61 was seen once earlier in the model comparison**

Unsure even for the labeller: 3 of 100. LLM share in the hybrid: 2% of tickets.

| Method | Correct | Accuracy | 95% range | Error rate | Accuracy on sure-only |
|---|---|---|---|---|---|
| bot_tag | 74/100 | 74% | 65%-82% | 26% | 73% (n=97) |
| keyword | 78/100 | 78% | 69%-85% | 22% | 78% (n=97) |
| tfidf | 100/100 | 100% | 96%-100% | 0% | 100% (n=97) |
| llm | 99/100 | 99% | 95%-100% | 1% | 100% (n=97) |
| hybrid_no_key | 99/100 | 99% | 95%-100% | 1% | 100% (n=97) |
| hybrid | 99/100 | 99% | 95%-100% | 1% | 100% (n=97) |

## Repeat-claim flag
| Method | Accuracy |
|---|---|
| llm | 97% |
| keyword | 82% |

## Where the hybrid is wrong (1)
| ticket | true | hybrid | via |
|---|---|---|---|
| TK-250383 | Battery drain / not charging | Audio fault (one side, distortion, mic) | llm |

## Most common confusions
- Battery drain / not charging -> Audio fault (one side, distortion, mic): 1
