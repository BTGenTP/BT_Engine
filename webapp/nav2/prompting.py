from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Tuple

from catalog_io import allowed_skills, required_param_names


def render_catalog_compact(catalog: Mapping[str, Any]) -> str:
    allowed = allowed_skills(catalog)
    required = required_param_names(catalog)
    lines: List[str] = []
    lines.append("Skills autorisés (Nav2 proxy) :")
    for sid in sorted(allowed.keys()):
        req = sorted(required.get(sid, set()))
        ports = allowed[sid].get("input_ports") or {}
        ports_list = ", ".join([str(k) for k in ports.keys() if isinstance(k, str)]) if isinstance(ports, dict) else ""
        req_list = ", ".join(req) if req else "(aucun port requis)"
        lines.append(f"- {sid}: ports=[{ports_list}] requis=[{req_list}]")
    return "\n".join(lines)


def system_prompt_base() -> str:
    return (
        "Tu es un assistant spécialisé Nav2 / BehaviorTree.CPP.\n"
        "Ta tâche: convertir une mission en une liste JSON STRICTE d'étapes.\n"
        "\n"
        "Règles STRICTES:\n"
        "- La sortie doit être UNIQUEMENT un JSON valide (aucun texte autour).\n"
        "- Le JSON est une liste d'objets: {\"skill\": <string>, \"params\": <object>, \"comment\": <string optional>}.\n"
        "- N'invente jamais de skills.\n"
        "- Utilise les unités: Wait=secondes, Spin=radians, BackUp=meters+m/s, DriveOnHeading=meters+m/s+seconds.\n"
    )


def build_mistral_inst_prompt(*, mission: str, catalog: Mapping[str, Any]) -> Tuple[str, str]:
    sys = system_prompt_base()
    cat = render_catalog_compact(catalog)
    instruction = f"{sys}\n{cat}\n\nMission: {mission}\n\nRéponds UNIQUEMENT avec la liste JSON."
    prompt = f"<s>[INST] {instruction} [/INST]\n### Steps JSON:\n"
    return prompt, "[/INST]"


def build_phi2_prompt(*, mission: str, catalog: Mapping[str, Any]) -> Tuple[str, str]:
    sys = system_prompt_base()
    cat = render_catalog_compact(catalog)
    prompt = f"{sys}\n{cat}\n\nMission: {mission}\n\n### Steps JSON:\n"
    return prompt, "\n### Steps JSON:"


def build_chat_messages(*, mission: str, catalog: Mapping[str, Any]) -> List[Dict[str, str]]:
    sys = system_prompt_base() + "\n" + render_catalog_compact(catalog)
    user = f"Mission: {mission}\n\nRéponds UNIQUEMENT avec la liste JSON."
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def serialize_steps_json(steps: List[Dict[str, Any]]) -> str:
    return json.dumps(steps, ensure_ascii=False, separators=(",", ":"))
