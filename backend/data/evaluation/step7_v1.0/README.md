# Step 7.0 — Evaluation correction and freeze

Status: PRE-REGISTERED. No official baseline, system, ablation, error-injection, or statistical run has been performed.

This directory records the pre-result evaluation decisions. InsightFlow-Bench v1.0 remains unchanged; hashes for its separate questions, ground truth, and dataset snapshots are in `step7_freeze_v1.0.json`.

## M1 — Numerical Accuracy applicability

M1 applies only when `ground_truth.value` is a finite numeric scalar or a non-empty list/mapping whose leaves are all finite numeric scalars. Boolean, textual, mixed categorical/numeric, null, and empty targets are non-applicable. This excludes metric-definition text, clarification targets, unsupported targets, and mixed ranking/category answers rather than scoring them as numeric errors. Exact integers/counts remain exact comparisons; floating-point targets still require the tolerance frozen by the applicable V2 contracts/configuration.

On the frozen benchmark, this rule yields **57 applicable targets and 43 excluded/non-applicable targets**. The evaluation code records the numerator (`numerical_correct_count`), denominator (`numerical_evaluated_count`), and excluded count (`numerical_excluded_count`). The denominator is recomputable from the frozen `ground_truth.json` with the rule above and is regression-tested.

## A1 — V2 without Planner

**Not executable under the current frozen implementation.** V2.3 `MultiAgentRequest` requires an `AnalysisPlan`; the Supervisor routes steps from that plan and does not derive steps from the question. Bypassing V2.2 therefore leaves no execution sequence. The previous hard-coded revenue plan was removed and A1 now fails closed. Substituting the V1 classifier or benchmark ground truth would introduce another decision mechanism and confound the planner ablation. The minimum future change is an explicit research-owner-approved A1 decision rule that supplies a plan without using V2.2 while preserving the question, controlled tools, and downstream stages; until that is frozen and implemented, A1 is N/A and no A1 result may be reported.

## Error-injection schedule

`error_injection_schedule_v1.0.json` pre-registers one `incorrect_intermediate_result` injection for each of four cases. The schedule spans aggregation, time analysis, and adversarial/error-oriented categories, with Easy and Medium difficulty. It targets the last successful Analysis Agent result for `count` or `distinct_count`, injects `-999999`, and expects V2.4's `numerical_verification` check to identify the affected step. The runtime-resolved step ID is retained in case evidence. If a scheduled case does not produce an eligible result, its injection is recorded as not applied and does not enter M5's denominator. Repetition is one per scheduled case.

V2.5 correction is expected to be inapplicable: its frozen correction strategy handles only the exact single-step identifier mismatch, not numerical-result repair. M5 and M6 use their actual applicable denominators; zero denominators remain N/A.

The schedule uses only an injection type already supported by the evaluation hook and V2.4 can independently check only direct `count`/`distinct_count` operations without unsupported dependent semantics. This is why no hard case is forced into the injection set.

## Frozen execution notes

- Baseline B is N/A because no canonical V1-tool to V2.2-operation mapping is frozen.
- A5 is the same complete configuration as System D and is an alias of that run, not a separate independent observation.
- The Step 6 statistical procedure is frozen in the JSON file. No inferential analysis occurs until valid results exist.
- This correction pass does not alter benchmark files, production V1/V2.2–V2.6 behavior, research definitions, or Git history.
