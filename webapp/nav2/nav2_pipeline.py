from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from bt_validation import compute_bt_structure_metrics, validate_bt_xml
from catalog_io import default_catalog_path, load_catalog
from json_to_xml import build_bt_xml, steps_from_dicts
from run_artifacts import create_run_dir, next_run_id, now_iso_z, write_json, write_text
from steps_parsing import StepsParseResult, parse_steps_strict


def default_nav2_catalog_path() -> Path:
    return default_catalog_path()


def load_nav2_catalog(catalog_path: Optional[str | Path] = None) -> Dict[str, Any]:
    return load_catalog(catalog_path or default_nav2_catalog_path())


def _validator_messages(report: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for issue in report.get("issues", []) or []:
        if not isinstance(issue, Mapping):
            continue
        level = str(issue.get("level") or "error").lower()
        code = str(issue.get("code") or "unknown")
        message = str(issue.get("message") or code)
        line = f"[{code}] {message}"
        if level == "warning":
            warnings.append(line)
        else:
            errors.append(line)
    return errors, warnings


def _parse_messages(parsed: StepsParseResult) -> list[str]:
    lines: list[str] = []
    if parsed.error_message:
        lines.append(parsed.error_message)
    for key, value in (parsed.errors or {}).items():
        if value:
            lines.append(f"{key}={value}")
    return lines


def parse_steps_payload(raw_steps: str, *, catalog: Mapping[str, Any]) -> Dict[str, Any]:
    parsed = parse_steps_strict(raw_steps, catalog)
    return {
        "ok": parsed.ok,
        "steps": parsed.steps or [],
        "steps_json": json.dumps(parsed.steps or [], ensure_ascii=False, indent=2),
        "error_message": parsed.error_message,
        "error_counters": parsed.errors,
        "errors": _parse_messages(parsed),
        "_parsed": parsed,
    }


def build_xml_from_steps(
    steps: list[dict[str, Any]],
    *,
    catalog: Mapping[str, Any],
    strict_attrs: bool = True,
    strict_blackboard: bool = True,
) -> Dict[str, Any]:
    xml_tree = build_bt_xml(steps_from_dicts(steps), catalog=catalog)
    with tempfile.NamedTemporaryFile("w+", suffix=".xml", delete=False, encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
    try:
        xml_tree.write(tmp_path, encoding="utf-8", xml_declaration=False)
        xml = tmp_path.read_text(encoding="utf-8")
        report = validate_bt_xml(xml_path=tmp_path, strict_attrs=strict_attrs, strict_blackboard=strict_blackboard)
        structure = compute_bt_structure_metrics(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    errors, warnings = _validator_messages(report)
    valid = bool(report.get("ok"))
    summary = report.get("summary") or {}
    summary_text = "Strict validator passed" if valid else f"Strict validator failed: errors={summary.get('errors', 0)} warnings={summary.get('warnings', 0)}"
    return {
        "xml": xml,
        "validation_report": report,
        "valid": valid,
        "score": 1.0 if valid else 0.0,
        "errors": errors,
        "warnings": warnings,
        "summary": summary_text,
        "structure": structure,
    }


def write_nav2_run_artifacts(
    *,
    mission: str,
    prompt: str,
    llm_raw: str,
    parsed: StepsParseResult,
    provider: str,
    model_name: Optional[str],
    temperature: float,
    constraints_kind: str,
    strict_attrs: bool,
    strict_blackboard: bool,
    latency_ms: int,
    xml_payload: Optional[Dict[str, Any]] = None,
) -> str:
    run_id = next_run_id()
    rp = create_run_dir(run_id)

    experiment = {
        "id": run_id,
        "llm": {"provider": provider, "api_base": None, "model": model_name, "temperature": float(temperature)},
        "generator": {"mode": "fail-fast", "constraints": {"enabled": constraints_kind != "off", "kind": constraints_kind}},
        "validation": {"strict_attrs": bool(strict_attrs), "strict_blackboard": bool(strict_blackboard)},
        "inputs": {"mission": mission},
    }

    write_text(rp.mission_txt, mission)
    write_json(rp.experiment_json, experiment)
    write_text(rp.prompt_rendered_txt, prompt)
    write_text(rp.llm_steps_raw_txt, llm_raw)

    metrics: Dict[str, Any] = {
        "schema_version": "0.1",
        "run_id": run_id,
        "timestamps": {"run_started_at": now_iso_z(), "run_finished_at": None},
        "llm": {
            "provider": provider,
            "api_base": None,
            "model": model_name,
            "temperature": float(temperature),
            "latency_ms": int(latency_ms),
            "tokens": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "errors": parsed.errors,
        },
        "bt": {"xml_valid": False, "validator": {"errors": None, "warnings": None, "issues_total": None}, "structure": {}},
        "simulation": {"enabled": False, "nav2_result": "UNKNOWN", "mission_success": False, "duration_s": None, "recovery_events": None, "replan_events": None, "distance_m": None},
    }

    if not parsed.ok or not parsed.steps:
        write_json(rp.llm_steps_json, [])
        write_json(rp.validation_report_json, {"ok": False, "issues": [{"level": "error", "code": "steps_invalid", "message": parsed.error_message}]})
    else:
        write_json(rp.llm_steps_json, parsed.steps)
        if xml_payload is None:
            xml_payload = build_xml_from_steps(parsed.steps, catalog=load_nav2_catalog(), strict_attrs=strict_attrs, strict_blackboard=strict_blackboard)
        write_text(rp.generated_bt_xml, xml_payload["xml"])
        write_json(rp.validation_report_json, xml_payload["validation_report"])
        metrics["bt"]["xml_valid"] = bool(xml_payload["valid"])
        summary = xml_payload["validation_report"].get("summary") or {}
        if isinstance(summary, dict):
            metrics["bt"]["validator"] = {
                "errors": summary.get("errors"),
                "warnings": summary.get("warnings"),
                "issues_total": summary.get("issues_total"),
            }
        metrics["bt"]["structure"] = xml_payload["structure"]

    metrics["timestamps"]["run_finished_at"] = now_iso_z()
    write_json(rp.metrics_json, metrics)
    return str(rp.run_dir)
