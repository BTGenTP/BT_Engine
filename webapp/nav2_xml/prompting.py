from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from catalog_io import allowed_skills, data_root, required_param_names


def _default_reference_bt_path() -> Path:
    return data_root() / "reference_behavior_trees" / "navigate_then_spin.xml"


def render_catalog_compact(
    catalog: Mapping[str, Any],
    reference_bt_path: Optional[Path] = None,
) -> str:
    """Render full catalog for the prompt: skills (bt_tag, node_type, ports, description, examples), control nodes, reference BT."""
    allowed = allowed_skills(catalog)
    required = required_param_names(catalog)
    lines: List[str] = []

    lines.append("Skills autorisés (Nav2 proxy) :")
    for sid in sorted(allowed.keys()):
        skill = allowed[sid]
        bt_tag = skill.get("bt_tag", sid)
        node_type = skill.get("node_type", "Action")
        req = sorted(required.get(sid, set()))
        input_ports = skill.get("input_ports") or {}
        output_ports = skill.get("output_ports") or {}
        input_typed = ", ".join(f"{k}: {v}" if isinstance(v, str) else str(k) for k, v in input_ports.items() if isinstance(k, str))
        req_list = ", ".join(req) if req else "(aucun port requis)"
        output_typed = ""
        if output_ports and isinstance(output_ports, dict):
            output_typed = ", ".join(f"{k}: {v}" if isinstance(v, str) else str(k) for k, v in output_ports.items() if isinstance(k, str))
        desc = (skill.get("semantic_description") or "").strip()
        examples = skill.get("examples") or []
        lines.append(f"- id={sid} | bt_tag={bt_tag!r} | node_type={node_type}")
        lines.append(f"  input_ports: {input_typed or '—'} | requis=[{req_list}]")
        if output_typed:
            lines.append(f"  output_ports: {output_typed}")
        if desc:
            lines.append(f"  description: {desc}")
        if examples:
            examples_str = "; ".join(json.dumps(ex, ensure_ascii=False) for ex in examples[:3])
            if len(examples) > 3:
                examples_str += " ..."
            lines.append(f"  exemples: {examples_str}")
    lines.append("")

    control = catalog.get("control_nodes_allowed") or []
    if control:
        lines.append("Noeuds de contrôle autorisés :")
        for c in control:
            if isinstance(c, dict):
                tag = c.get("bt_tag", "?")
                attrs = c.get("attributes") or []
                lines.append(f"- {tag}" + (f" attributs=[{', '.join(attrs)}]" if attrs else ""))
        lines.append("")

    ref_path = reference_bt_path or _default_reference_bt_path()
    if ref_path and ref_path.is_file():
        lines.append("Behavior Tree de référence (structure et tags à respecter) :")
        lines.append(ref_path.read_text(encoding="utf-8").strip())
        lines.append("")

    return "\n".join(lines)


def system_prompt_base() -> str:
    return (
        "Tu es un assistant spécialisé Nav2 / BehaviorTree.CPP.\n"
        "Ta tâche: convertir une mission en un Behavior Tree XML complet, compatible BehaviorTree.CPP v4.\n"
        "\n"
        "Règles STRICTES:\n"
        "- La sortie doit être UNIQUEMENT du XML (aucun markdown, aucun texte autour).\n"
        "- Structure obligatoire: <root main_tree_to_execute=\"MainTree\"> puis <BehaviorTree ID=\"MainTree\">.\n"
        "- N'invente jamais de tags/skills.\n"
        "- Utilise les unités: Wait=secondes, Spin=radians, BackUp=meters+m/s, DriveOnHeading=meters+m/s+seconds.\n"
    )


def build_mistral_inst_prompt(*, mission: str, catalog: Mapping[str, Any]) -> Tuple[str, str]:
    sys = system_prompt_base()
    cat = render_catalog_compact(catalog)
    instruction = f"{sys}\n{cat}\n\nMission: {mission}\n\nRéponds UNIQUEMENT avec le BT XML."
    prompt = f"<s>[INST] {instruction} [/INST]\n### BT XML:\n"
    return prompt, "[/INST]"


def build_phi2_prompt(*, mission: str, catalog: Mapping[str, Any]) -> Tuple[str, str]:
    sys = system_prompt_base()
    cat = render_catalog_compact(catalog)
    prompt = f"{sys}\n{cat}\n\nMission: {mission}\n\n### BT XML:\n"
    return prompt, "\n### BT XML:"


def build_chat_messages(*, mission: str, catalog: Mapping[str, Any]) -> List[Dict[str, str]]:
    sys = system_prompt_base() + "\n" + render_catalog_compact(catalog)
    user = f"Mission: {mission}\n\nRéponds UNIQUEMENT avec le BT XML."
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

