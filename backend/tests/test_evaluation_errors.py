"""Unit tests for the official V2.1 E1–E15 error taxonomy."""

from app.evaluation.errors import ErrorCategory, classify_error


def test_all_15_error_categories_exist():
    categories = [e.value for e in ErrorCategory]
    assert len(categories) == 15
    for i in range(1, 16):
        assert any(c.startswith(f"E{i}_") for c in categories)


def test_classify_error_mapping():
    # Direct code mappings
    e_intent = classify_error("planner", "INVALID_INTENT", "Intent was not recognized")
    assert e_intent.error_category == ErrorCategory.E1_INTENT_MISUNDERSTANDING.value

    e_schema = classify_error("execution", "MISSING_COLUMN", "Column not found")
    assert e_schema.error_category == ErrorCategory.E4_WRONG_COLUMN_SELECTION.value

    e_num = classify_error("verification", "NUMERICAL_MISMATCH", "Values did not match")
    assert e_num.error_category == ErrorCategory.E6_NUMERICAL_CALCULATION_ERROR.value

    e_verif = classify_error("verification", "VERIFICATION_FAILED", "Checks failed")
    assert e_verif.error_category == ErrorCategory.E10_VERIFICATION_FAILURE.value

    e_corr = classify_error("self_correction", "CORRECTION_FAILED", "Could not reconcile")
    assert e_corr.error_category == ErrorCategory.E12_CORRECTION_FAILURE.value

    e_ground = classify_error("explanation", "UNGROUNDED_NUMBER", "Hallucinated digit")
    assert e_ground.error_category == ErrorCategory.E13_EXPLANATION_GROUNDING_FAILURE.value

    e_rag = classify_error("knowledge", "METRIC_NOT_FOUND", "Metric not in pgvector")
    assert e_rag.error_category == ErrorCategory.E3_METRIC_DEFINITION_ERROR.value


def test_classify_fallback_by_stage():
    e_plan = classify_error("planner", "UNKNOWN_CODE", "Something broke")
    assert e_plan.error_category == ErrorCategory.E1_INTENT_MISUNDERSTANDING.value

    e_verif = classify_error("verification", "SOME_OTHER_ERROR", "Check failure")
    assert e_verif.error_category == ErrorCategory.E10_VERIFICATION_FAILURE.value
