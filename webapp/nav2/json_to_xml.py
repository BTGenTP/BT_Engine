from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from xml.etree.ElementTree import Comment, Element, ElementTree, SubElement

import xml.etree.ElementTree as ET

from catalog_io import allowed_skills


def reference_tree_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "reference_behavior_trees" / "navigate_then_spin.xml"


@dataclass(frozen=True)
class MissionStep:
    skill: str
    params: Dict[str, Any]
    comment: Optional[str] = None


def _indent_xml(elem: Element, level: int = 0) -> None:
    indent_str = "  "
    i = "\n" + level * indent_str
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent_str
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():  # type: ignore[name-defined]
            child.tail = i  # type: ignore[name-defined]
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def _add_nav_subtree_definition(root: Element) -> None:
    src_tree = ET.parse(str(reference_tree_path()))
    src_root = src_tree.getroot()
    found = None
    for bt in src_root.findall("BehaviorTree"):
        if bt.get("ID") == "NavigateToPoseWithReplanningAndRecovery":
            found = bt
            break
    if found is None:
        raise RuntimeError("Missing NavigateToPoseWithReplanningAndRecovery subtree in local reference BT")
    copied = ET.fromstring(ET.tostring(found, encoding="utf-8"))
    root.append(copied)


def build_bt_xml(steps: List[MissionStep], catalog: Mapping[str, Any]) -> ElementTree:
    allowed = allowed_skills(catalog)
    tag_by_skill = {sid: allowed[sid]["bt_tag"] for sid in allowed.keys()}

    root = Element("root", main_tree_to_execute="MainTree")
    bt = SubElement(root, "BehaviorTree", ID="MainTree")
    seq = SubElement(bt, "Sequence", name="TurtlebotMission")

    needs_nav_subtree = False
    for step in steps:
        if step.comment:
            seq.append(Comment(f" {step.comment} "))
        if step.skill == "NavigateToGoalWithReplanningAndRecovery":
            needs_nav_subtree = True
            SubElement(
                seq,
                "SubTree",
                ID="NavigateToPoseWithReplanningAndRecovery",
                __shared_blackboard="true",
            )
            continue

        bt_tag = tag_by_skill.get(step.skill)
        if not isinstance(bt_tag, str) or not bt_tag:
            raise RuntimeError(f"Unknown skill tag mapping for: {step.skill}")

        attrs: Dict[str, str] = {}
        for k, v in (step.params or {}).items():
            if isinstance(v, bool):
                attrs[k] = "true" if v else "false"
            else:
                attrs[k] = str(v)
        SubElement(seq, bt_tag, **attrs)

    if needs_nav_subtree:
        _add_nav_subtree_definition(root)

    _indent_xml(root)
    return ElementTree(root)


def steps_from_dicts(items: List[Dict[str, Any]]) -> List[MissionStep]:
    out: List[MissionStep] = []
    for it in items:
        out.append(
            MissionStep(
                skill=str(it.get("skill")),
                params=dict(it.get("params") or {}),
                comment=(str(it["comment"]) if it.get("comment") is not None else None),
            )
        )
    return out
