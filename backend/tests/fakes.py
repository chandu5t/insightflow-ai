"""Test doubles for Gemini. No test ever calls the real Gemini API."""

import json


class FakeLlm:
    """Returns a fixed reply, or raises a fixed error, and records what it was asked."""

    model_name = "fake-gemini"

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def plan_json(**overrides: object) -> str:
    """A valid plan (total revenue) as JSON text. Override any field."""
    plan: dict[str, object] = {
        "intent": "total_revenue", "tool_name": "revenue", "metric": None, "aggregation": None,
        "group_by": None, "sort_order": None, "limit": None, "confidence": 0.95,
        "reasoning": "Asks for the total revenue.",
    }
    plan.update(overrides)
    return json.dumps(plan)