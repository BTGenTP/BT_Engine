from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def data_root() -> Path:
    return Path(__file__).resolve().parent / "data"


def default_catalog_path() -> Path:
    return (data_root() / "bt_nodes_catalog.json").resolve()


def load_catalog(path: str | Path | None = None) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve() if path is not None else default_catalog_path()
    return json.loads(p.read_text(encoding="utf-8"))


def iter_atomic_skills(catalog: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for item in catalog.get("atomic_skills", []) or []:
        if isinstance(item, dict):
            yield item


def allowed_skills(catalog: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in iter_atomic_skills(catalog):
        sid = item.get("id")
        if isinstance(sid, str) and sid.strip():
            out[sid] = dict(item)
    return out


def all_param_names(catalog: Mapping[str, Any]) -> Dict[str, set[str]]:
    internal_ports = {"ID", "__shared_blackboard"}
    out: Dict[str, set[str]] = {}
    for sid, item in allowed_skills(catalog).items():
        names: set[str] = set()
        input_ports = item.get("input_ports") or {}
        output_ports = item.get("output_ports") or {}
        if isinstance(input_ports, dict):
            for k in input_ports.keys():
                if isinstance(k, str) and k not in internal_ports:
                    names.add(k)
        if isinstance(output_ports, dict):
            for k in output_ports.keys():
                if isinstance(k, str) and k not in internal_ports:
                    names.add(k)
        out[sid] = names
    return out


def required_param_names(catalog: Mapping[str, Any]) -> Dict[str, set[str]]:
    internal_ports = {"ID", "__shared_blackboard"}
    out: Dict[str, set[str]] = {}
    for sid, item in allowed_skills(catalog).items():
        req: set[str] = set()
        input_ports = item.get("input_ports") or {}
        if isinstance(input_ports, dict):
            for port_name, port_desc in input_ports.items():
                if not isinstance(port_name, str) or port_name in internal_ports:
                    continue
                is_optional = isinstance(port_desc, str) and ("optional" in port_desc.lower())
                if not is_optional:
                    req.add(port_name)
        out[sid] = req
    return out


@dataclass(frozen=True)
class CatalogSummary:
    catalog_path: Path
    skill_ids: list[str]
    required_ports_by_skill: dict[str, list[str]]
    all_ports_by_skill: dict[str, list[str]]


def summarize_catalog(path: str | Path | None = None) -> CatalogSummary:
    p = Path(path).expanduser().resolve() if path is not None else default_catalog_path()
    cat = load_catalog(p)
    skills = allowed_skills(cat)
    required_ports = required_param_names(cat)
    all_ports = all_param_names(cat)
    return CatalogSummary(
        catalog_path=p,
        skill_ids=sorted(skills.keys()),
        required_ports_by_skill={k: sorted(v) for k, v in required_ports.items()},
        all_ports_by_skill={k: sorted(v) for k, v in all_ports.items()},
    )

