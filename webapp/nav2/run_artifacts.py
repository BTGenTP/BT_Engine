from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def project_root() -> Path:
    return Path(__file__).resolve().parent


def runs_root() -> Path:
    return project_root() / "runs"


def _today_ymd() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d")


def next_run_id(*, date_ymd: Optional[str] = None) -> str:
    d = date_ymd or _today_ymd()
    base = runs_root()
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob(f"{d}_exp*"))
    max_n = 0
    for p in existing:
        name = p.name
        if not name.startswith(f"{d}_exp"):
            continue
        suf = name[len(f"{d}_exp") :]
        try:
            n = int(suf)
        except Exception:
            continue
        max_n = max(max_n, n)
    return f"{d}_exp{max_n + 1:03d}"


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    mission_txt: Path
    experiment_json: Path
    prompt_rendered_txt: Path
    llm_xml_raw_txt: Path
    generated_bt_xml: Path
    validation_report_json: Path
    metrics_json: Path


def create_run_dir(run_id: str) -> RunPaths:
    rd = runs_root() / run_id
    if rd.exists():
        raise FileExistsError(f"Run directory already exists: {rd}")
    rd.mkdir(parents=True, exist_ok=False)
    return RunPaths(
        run_dir=rd,
        mission_txt=rd / "mission.txt",
        experiment_json=rd / "experiment.json",
        prompt_rendered_txt=rd / "prompt_rendered.txt",
        llm_xml_raw_txt=rd / "llm_xml_raw.txt",
        generated_bt_xml=rd / "generated_bt.xml",
        validation_report_json=rd / "validation_report.json",
        metrics_json=rd / "metrics.json",
    )


def write_text(path: Path, content: str) -> None:
    path.write_text((content or "") + ("\n" if not (content or "").endswith("\n") else ""), encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now_iso_z() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

