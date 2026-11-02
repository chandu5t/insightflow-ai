# InsightFlow-Bench v1.0

This is the 100-case benchmark snapshot for the V2.8 Step 6 design. It is not an experiment result. The question set and its ground truth are stored separately so evaluator-only labels are never part of the question artifact.

## Snapshot files

- `questions.json`: stable case ID, dataset filename, question, V2.1 category, and difficulty only.
- `ground_truth.json`: matching stable ID, structured result, expected operation sequence, formula, intermediate results, tolerance field, and provenance.
- `datasets/module3_sales.csv` and `datasets/module3_sales_direct.csv`: byte-for-byte snapshots of the existing fixed evaluation CSVs.

The files are joined in memory by `app.evaluation.official_benchmark.load_official_benchmark`. The adapter checks IDs, required fields, registered operation names, exactly 100 cases, all 12 categories, and all three difficulty levels, then passes the joined records through V2.7's existing `load_benchmark`. The returned `BenchmarkCase` objects work with `EvaluationRunner`; `sanitize_case_for_execution` continues to remove expected answers and ground truth before execution. No V2.7 schema or runner was changed.

For an experiment, upload/register these exact dataset snapshots through the existing dataset interface and supply the resulting dataset ID mapping in the experiment configuration. The evaluator does not read these repository paths as a production dataset source.

## Coverage

| V2.1 category | Cases |
|---|---:|
| A Aggregation | 12 |
| B Filtering | 10 |
| C Grouping | 10 |
| D Ranking | 8 |
| E Comparison | 8 |
| F Time Analysis | 8 |
| G Missing-Value Reasoning | 8 |
| H Metric Definition | 8 |
| I Multi-Step Reasoning | 10 |
| J Ambiguous Questions | 6 |
| K Unsupported Questions | 6 |
| L Adversarial/Error-Oriented | 6 |
| **Total** | **100** |

| Difficulty | Cases |
|---|---:|
| Easy | 34 |
| Medium | 42 |
| Hard | 24 |

The design does not require equal category-by-difficulty cells; this observed distribution is recorded for reporting.

## Ground-truth provenance and limits

Numerical answers were calculated deterministically from the two versioned CSV snapshots, independently of any evaluated planner, agent, baseline, or model output. Formula and source notes are stored on each ground-truth record. Metric-definition answers cite the committed V1 knowledge seed (`backend/app/services/knowledge_seed_data.py`) as their source. Ambiguous and unsupported cases have explicit structured clarification/unsupported targets rather than guessed answers.

This initial snapshot is narrow: both datasets are small sales samples, one is reused from the V1 fixed evaluation data, and the time coverage is January 2025 only. It does not establish broad business-domain or temporal generalization. Some questions deliberately exercise V2.3 operations whose execution semantics are undefined; V2.3/V2.4 safe failure remains the expected system boundary. M3 dependency correctness has no expected dependency graph in this snapshot and must not be claimed as measured; the V2.7 diagnostic remains separate from official M3.

No Baseline/System condition, ablation, official experiment, or statistical analysis was run to create or validate this benchmark. This snapshot contains ground truth and test/validation evidence only.
