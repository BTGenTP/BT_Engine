from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from catalog_io import allowed_skills, required_param_names


def strip_markdown_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


@dataclass(frozen=True)
class StepsParseResult:
    ok: bool
    steps: Optional[List[Dict[str, Any]]]
    errors: Dict[str, int]
    error_message: Optional[str] = None


def _normalize_params_aliases(skill: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not params:
        return {}
    out = dict(params)
    aliases_by_skill: Dict[str, Dict[str, str]] = {
        "Wait": {"duration": "wait_duration", "seconds": "wait_duration", "sec": "wait_duration"},
        "Spin": {"angle": "spin_dist", "radians": "spin_dist", "rad": "spin_dist"},
        "BackUp": {"distance": "backup_dist", "dist": "backup_dist", "speed": "backup_speed"},
        "DriveOnHeading": {
            "distance": "dist_to_travel",
            "dist": "dist_to_travel",
            "speed_mps": "speed",
            "timeout": "time_allowance",
            "time": "time_allowance",
            "time_limit": "time_allowance",
        },
    }
    amap = aliases_by_skill.get(skill) or {}
    for src, dst in amap.items():
        if src in out and dst not in out:
            out[dst] = out[src]
    return out


def parse_steps_strict(raw: str, catalog: Mapping[str, Any]) -> StepsParseResult:
    counters = {"non_json": 0, "skill_not_allowed": 0, "missing_required_port": 0, "other": 0}
    cleaned = strip_markdown_fences(raw)
    try:
        data = json.loads(cleaned)
    except Exception as exc:
        counters["non_json"] += 1
        return StepsParseResult(ok=False, steps=None, errors=counters, error_message=f"Non-JSON output: {exc}")

    if not isinstance(data, list) or not data:
        counters["other"] += 1
        return StepsParseResult(ok=False, steps=None, errors=counters, error_message="Expected a non-empty JSON list of steps.")

    allowed = allowed_skills(catalog)
    required = required_param_names(catalog)

    steps: List[Dict[str, Any]] = []
    try:
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Step #{idx} must be a JSON object.")
            skill = item.get("skill")
            if not isinstance(skill, str) or skill not in allowed:
                counters["skill_not_allowed"] += 1
                raise ValueError(f"Skill '{skill}' not allowed.")

            params = item.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise ValueError(f"Step '{skill}': params must be a JSON object.")
            params = _normalize_params_aliases(skill, params)

            for port in sorted(required.get(skill, set())):
                if port not in params:
                    counters["missing_required_port"] += 1
                    raise ValueError(f"Step '{skill}': missing required port: {port}")

            out_step: Dict[str, Any] = {"skill": skill, "params": params}
            if "comment" in item and item["comment"] is not None:
                if not isinstance(item["comment"], str):
                    raise ValueError(f"Step '{skill}': comment must be a string.")
                out_step["comment"] = item["comment"]
            steps.append(out_step)
    except Exception as exc:
        counters["other"] += 1
        return StepsParseResult(ok=False, steps=None, errors=counters, error_message=str(exc))

    return StepsParseResult(ok=True, steps=steps, errors=counters)
