from __future__ import annotations

from typing import Any, Dict, Mapping

from catalog_io import all_param_names, allowed_skills


def build_steps_jsonschema(catalog: Mapping[str, Any]) -> Dict[str, Any]:
    skills = sorted(allowed_skills(catalog).keys())
    ports_union = sorted({p for s in all_param_names(catalog).values() for p in s})

    params_props: Dict[str, Any] = {}
    for p in ports_union:
        params_props[p] = {"type": ["number", "boolean", "string", "integer"]}

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "required": ["skill", "params"],
            "additionalProperties": False,
            "properties": {
                "skill": {"type": "string", "enum": skills},
                "params": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": params_props,
                },
                "comment": {"type": "string"},
            },
        },
    }
