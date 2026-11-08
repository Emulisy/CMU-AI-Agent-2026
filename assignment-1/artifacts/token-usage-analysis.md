# Context compaction versus full-context baseline

## Runs and method

Compared the two saved runs of `django__django-15368` using `deepseek-flash`:

- Compacted: `django__django-15368-trajectory.json`, intended threshold 6,000 tokens, with 4 recorded compactions.
- Baseline: `django__django-15368-baseline-trajectory.json`, with no recorded compactions.

Both runs start with identical message prompts. Each recorded model response has token-usage data. Action usage is summed from `responses[*].usage`; summarization usage is summed separately from `compactions[*].compaction_response.usage` and included in the compacted run's overall total. Completion tokens already include reported reasoning tokens, so reasoning tokens are not added again. These measurements cover generation requests, not separate evaluation sandboxes or monetary charges.

## Token usage

| Metric | Compacted | Baseline |
| --- | ---: | ---: |
| Action model requests | 18 | 11 |
| Summary model requests | 4 | 0 |
| All model requests | 22 | 11 |
| Action prompt tokens | 76,387 | 115,822 |
| Action completion tokens | 3,148 | 2,374 |
| Summary prompt tokens | 21,083 | 0 |
| Summary completion tokens | 2,273 | 0 |
| All prompt tokens | 97,470 | 115,822 |
| All completion tokens | 5,421 | 2,374 |
| **All tokens, including summaries** | **102,891** | **118,196** |
| Mean prompt tokens per action request | 4,243.7 | 10,529.3 |
| Maximum prompt tokens in an action request | 5,761 | 18,371 |

The compacted run used **15,305 fewer total tokens (12.95%)**, including summarization overhead. Action prompt usage fell by **34.05%**; after including summary requests, total prompt usage fell by **15.85%**. Summarization itself consumed **23,356 tokens**, partially offsetting the reduction in action-request usage.

The baseline's action prompt grew from 1,487 to 18,371 tokens. Compaction kept the action-request peak at 5,761 tokens, **68.64% lower** than the baseline peak. However, the compacted run took 18 action requests rather than 11. Total completion usage was higher because there were more action requests and four additional summary generations.

## Recorded context reductions

| After action step | Estimated tokens before | Estimated tokens after | Reduction |
| --- | ---: | ---: | ---: |
| 6 | 8,181 | 4,100 | 49.88% |
| 7 | 6,814 | 3,658 | 46.32% |
| 9 | 5,943 | 1,638 | 72.44% |
| 14 | 5,945 | 2,277 | 61.70% |

All four compactions materially reduced the active message context. Comparing each event's saved pre-compaction prompt with the next action request confirms that the original system/task messages and the latest complete assistant turn, including its tool observations, were retained unchanged.

The table uses the harness's rough serialized-message token estimates. These differ from provider-reported prompt tokens and from the calibrated estimate used to trigger compaction. Consequently, a recorded "before" value below 6,000 is not by itself evidence that the threshold setting was ignored. The threshold is a trigger for checking context, not an exact maximum request size.

## Caching and other tradeoffs

| Prompt-cache metric, including summary requests | Compacted | Baseline |
| --- | ---: | ---: |
| Cached prompt tokens | 60,288 | 100,608 |
| Uncached prompt tokens | 37,182 | 15,214 |
| Cached share of prompt tokens | 61.85% | 86.86% |

The baseline reused a growing, stable prefix, while compaction replaced older messages and issued separate summary requests. Summary requests alone used 19,163 uncached prompt tokens. Overall uncached prompt usage was **2.44 times** the baseline's, despite lower total token usage. Cached-input discounts and output-token prices matter: these results demonstrate token reduction, but do **not** establish lower monetary cost.

Compaction trades detailed history for shorter working memory. This can help avoid context-window limits, but summaries can omit details and require additional investigation. The compacted trajectory re-inspected source/test snippets after early compactions; this is consistent with a need to recover detail, but does not prove that compaction caused the extra steps. The runs also chose different command groupings and regression tests. With one stochastic run per condition, differences in step count cannot be attributed solely to compaction, and latency was not measured.

## Patch and test evidence

Both trajectories contain actual tool output reporting **30 tests run successfully** in `queries.test_bulk_update` inside the generation sandbox, after the agents added their own regression coverage. Both ended with `send_message` and exported a patch containing only `django/db/models/query.py`.

Both patches replace `isinstance(attr, Expression)` with `hasattr(attr, 'resolve_expression')`, allowing plain `F()` expressions to be resolved instead of wrapped in `Value`. The baseline also removes the now-unused `Expression` import; the compacted patch leaves it in place.

Both exported patches pass a read-only `git apply --numstat` syntax check from the repository root: the compacted patch has 1 insertion/1 deletion, and the baseline has 2 insertions/2 deletions. Neither contains the doubled `CR CR LF` sequences seen in the earlier Part 1 patch.

The user's supplied terminal output confirms that both patches were subsequently evaluated by `make check-swebench` in separate fresh Modal sandboxes using the published vendored Django image. Both the candidate patch and the supplied test patch applied successfully in each evaluation.

| Fresh-sandbox evaluation | Compacted | Baseline |
| --- | ---: | ---: |
| FAIL_TO_PASS (`test_f_expression`) | 1/1 passed | 1/1 passed |
| PASS_TO_PASS | 29/29 passed | 29/29 passed |
| Final result | `RESOLVED: yes` | `RESOLVED: yes` |

The evaluations passed despite a preceding Windows shell warning: the Makefile's Unix `test -f` command was unavailable, resulting in a misleading "Patch not found" message. In the supplied output, the evaluator nevertheless ran, loaded the patches, applied them, and passed every required test. That warning does not indicate evaluation failure for these runs.

## Findings

The compacted run satisfies the specified experiment requirements: four compactions occurred, each substantially reduced context, and its patch passed fresh-sandbox evaluation with all required FAIL_TO_PASS and PASS_TO_PASS tests passing. The baseline also passed evaluation. Total generation tokens were 12.95% lower with compaction after including summary overhead, while requests, completion tokens, and uncached prompt tokens increased. This pair demonstrates successful context reduction while retaining task correctness, but does not establish a general cost or step-count advantage.
