from app.core.config import Settings
from app.evaluation.official_benchmark import verify_official_benchmark_hashes
from app.evaluation.run_step7 import build_experiment_config, load_injection_schedule
from app.evaluation.schemas import SystemCondition
from app.evaluation.official_benchmark import OfficialBenchmarkDatasetRepository


def test_step7_entrypoint_builds_official_configs_without_live_services():
    freeze = verify_official_benchmark_hashes()
    repository = OfficialBenchmarkDatasetRepository()
    settings = Settings(gemini_model="test-configured-model")

    config = build_experiment_config(
        SystemCondition.SYSTEM_D_FULL_V2,
        settings=settings,
        dataset_mapping=repository.dataset_mapping,
        freeze=freeze,
        run_suffix="unit-test",
    )

    assert config.benchmark_id == "InsightFlow-Bench"
    assert config.benchmark_version == freeze["benchmark"]["benchmark_version"]
    assert config.dataset_version == freeze["benchmark"]["dataset_version"]
    assert config.model_name == "test-configured-model"
    assert config.evaluation_configuration["development_fixture"] is False
    assert config.evaluation_configuration["retriever_mode"] == "knowledge_base"
    assert config.evaluation_configuration["dataset_mapping"] == repository.dataset_mapping


def test_step7_entrypoint_loads_only_the_frozen_four_case_injections():
    case_ids, injections = load_injection_schedule()

    assert case_ids == ["IFB-002", "IFB-012", "IFB-053", "IFB-096"]
    assert set(injections) == set(case_ids)
    assert all(item.error_type == "incorrect_intermediate_result" for item in injections.values())
    assert all(item.target_step_id is None and item.injected_value == -999999 for item in injections.values())
