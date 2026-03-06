from __future__ import annotations

from typing import Any, Callable, Mapping

from constraints.steps_jsonschema import build_steps_jsonschema


def build_prefix_allowed_tokens_fn(tokenizer, catalog: Mapping[str, Any]) -> Callable[[int, Any], list[int]]:
    try:
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn  # type: ignore
        from lmformatenforcer import JsonSchemaParser  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("lm-format-enforcer not installed. Install it to use constrained decoding.") from exc

    schema = build_steps_jsonschema(catalog)
    parser = JsonSchemaParser(schema)
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
