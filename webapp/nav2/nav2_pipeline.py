from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from bt_validation import compute_bt_structure_metrics, validate_bt_xml
from catalog_io import default_catalog_path, load_catalog
from run_artifacts import create_run_dir, next_run_id, now_iso_z, write_json, write_text


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


def build_xml_payload(
    xml: str,
    *,
    strict_attrs: bool = True,
    strict_blackboard: bool = True,
    catalog_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile("w+", suffix=".xml", delete=False, encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(xml or "")
        tmp.flush()
    try:
        report = validate_bt_xml(
            xml_path=tmp_path,
            strict_attrs=strict_attrs,
            strict_blackboard=strict_blackboard,
            catalog_path=Path(catalog_path).expanduser().resolve() if catalog_path else None,
        )
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


def write_nav2_xml_run_artifacts(
    *,
    mission: str,
    prompt: str,
    llm_raw: str,
    provider: str,
    model_name: Optional[str],
    temperature: float,
    constraints_kind: str,
    strict_attrs: bool,
    strict_blackboard: bool,
    latency_ms: int,
    xml_payload: Dict[str, Any],
    tokens: Optional[Dict[str, int]] = None,
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
    write_text(rp.llm_xml_raw_txt, llm_raw)
    write_text(rp.generated_bt_xml, xml_payload["xml"])
    write_json(rp.validation_report_json, xml_payload["validation_report"])

    summary = xml_payload["validation_report"].get("summary") or {}
    tok = tokens or {}
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
            "tokens": {
                "prompt_tokens": tok.get("prompt_tokens"),
                "completion_tokens": tok.get("completion_tokens"),
                "total_tokens": tok.get("total_tokens"),
            },
            "errors": {},
        },
        "bt": {
            "xml_valid": bool(xml_payload["valid"]),
            "validator": {
                "errors": (summary.get("errors") if isinstance(summary, dict) else None),
                "warnings": (summary.get("warnings") if isinstance(summary, dict) else None),
                "issues_total": (summary.get("issues_total") if isinstance(summary, dict) else None),
            },
            "structure": xml_payload.get("structure") or {},
        },
        "simulation": {"enabled": False, "nav2_result": "UNKNOWN", "mission_success": False},
    }

    metrics["timestamps"]["run_finished_at"] = now_iso_z()
    write_json(rp.metrics_json, metrics)
    return str(rp.run_dir)

