from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from xml.etree import ElementTree as ET

from catalog_io import load_catalog

BB_VAR_RE = re.compile(r"\{([^{}]+)\}")
CONTROL_ATTR_TYPES: Dict[str, Dict[str, str]] = {
    "RateController": {"hz": "float"},
    "DistanceController": {"distance": "float"},
    "SpeedController": {"max_speed": "float"},
    "Repeat": {"num_cycles": "int"},
    "RecoveryNode": {"number_of_retries": "int"},
}


def _infer_type(desc: Any) -> str:
    if not isinstance(desc, str):
        return "string"
    low = desc.lower()
    if "bool" in low:
        return "bool"
    if "int" in low:
        return "int"
    if "float" in low or "double" in low:
        return "float"
    return "string"


def _load_allowlists(catalog: Mapping[str, Any], reference_dir: Path) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]], dict[str, dict[str, str]]]:
    allowed_tags = {"root", "BehaviorTree", "SubTree"}
    allowed_attrs = {
        "root": {"main_tree_to_execute"},
        "BehaviorTree": {"ID"},
        "SubTree": {"ID", "__shared_blackboard"},
    }
    required_attrs = {
        "root": {"main_tree_to_execute"},
        "BehaviorTree": {"ID"},
        "SubTree": {"ID"},
    }
    attr_types = {
        "root": {"main_tree_to_execute": "string"},
        "BehaviorTree": {"ID": "string"},
        "SubTree": {"ID": "string", "__shared_blackboard": "bool"},
    }

    for item in catalog.get("control_nodes_allowed", []) or []:
        tag = item.get("bt_tag")
        attrs = item.get("attributes") or []
        if isinstance(tag, str) and tag:
            allowed_tags.add(tag)
            allowed_attrs.setdefault(tag, set()).update(a for a in attrs if isinstance(a, str))
            attr_types.setdefault(tag, {}).update(CONTROL_ATTR_TYPES.get(tag, {}))

    for item in catalog.get("atomic_skills", []) or []:
        tag = item.get("bt_tag")
        if not isinstance(tag, str) or not tag:
            continue
        allowed_tags.add(tag)
        input_ports = item.get("input_ports") or {}
        output_ports = item.get("output_ports") or {}
        allowed_attrs.setdefault(tag, set())
        attr_types.setdefault(tag, {})
        if isinstance(input_ports, dict):
            for key, desc in input_ports.items():
                if not isinstance(key, str):
                    continue
                allowed_attrs[tag].add(key)
                attr_types[tag][key] = _infer_type(desc)
                is_optional = isinstance(desc, str) and "optional" in desc.lower()
                if key not in {"ID", "__shared_blackboard"} and not is_optional:
                    required_attrs.setdefault(tag, set()).add(key)
        if isinstance(output_ports, dict):
            for key in output_ports.keys():
                if isinstance(key, str):
                    allowed_attrs[tag].add(key)

    for xml_path in reference_dir.glob("*.xml"):
        tree = ET.parse(str(xml_path))
        for el in tree.getroot().iter():
            allowed_tags.add(el.tag)
            allowed_attrs.setdefault(el.tag, set()).update(el.attrib.keys())

    return allowed_tags, allowed_attrs, required_attrs, attr_types


def _check_type(expected: str, value: str) -> bool:
    low = (value or "").strip().lower()
    if BB_VAR_RE.search(value or ""):
        return True
    if expected == "bool":
        return low in {"true", "false"}
    if expected == "int":
        return re.fullmatch(r"-?\d+", value or "") is not None
    if expected == "float":
        try:
            float(value)
            return True
        except Exception:
            return False
    return True


def validate_bt_xml(
    *,
    xml_path: Path,
    strict_attrs: bool,
    strict_blackboard: bool,
    catalog_path: Optional[Path] = None,
    reference_dir: Optional[Path] = None,
    external_bb_vars: Optional[list[str]] = None,
) -> Dict[str, Any]:
    issues: list[Dict[str, Any]] = []
    catalog = load_catalog(catalog_path)
    ref_dir = reference_dir or (Path(__file__).resolve().parent / "data" / "reference_behavior_trees")
    allowed_tags, allowed_attrs, required_attrs, attr_types = _load_allowlists(catalog, ref_dir)

    try:
        tree = ET.parse(str(xml_path))
    except Exception as exc:
        return {"ok": False, "issues": [{"level": "error", "code": "xml_parse", "message": str(exc)}], "summary": {"issues_total": 1, "errors": 1, "warnings": 0}}

    root = tree.getroot()
    if root.tag != "root":
        issues.append({"level": "error", "code": "root_tag", "message": "Root tag must be <root>"})
    main_tree = root.get("main_tree_to_execute")
    if not main_tree:
        issues.append({"level": "error", "code": "root_main_tree", "message": "root@main_tree_to_execute missing"})

    bt_ids = []
    produced_bb_vars = set(external_bb_vars or ["goal"])
    consumed_bb_vars = set()

    for el in root.iter():
        tag = el.tag
        if tag not in allowed_tags:
            issues.append({"level": "error", "code": "tag_not_allowed", "message": f"Tag '{tag}' not allowed", "tag": tag})
            continue
        if tag == "BehaviorTree":
            bt_id = el.get("ID")
            if not bt_id:
                issues.append({"level": "error", "code": "bt_missing_id", "message": "BehaviorTree without ID"})
            else:
                bt_ids.append(bt_id)

        for req in required_attrs.get(tag, set()):
            if req not in el.attrib:
                issues.append({"level": "error", "code": "missing_required_attr", "message": f"Missing required attr '{req}' on <{tag}>", "tag": tag})

        if strict_attrs:
            for attr_name, attr_value in el.attrib.items():
                if attr_name not in allowed_attrs.get(tag, set()):
                    issues.append({"level": "error", "code": "unknown_attr", "message": f"Unknown attr '{attr_name}' on <{tag}>", "tag": tag})
                expected_type = attr_types.get(tag, {}).get(attr_name)
                if expected_type and not _check_type(expected_type, attr_value):
                    issues.append({"level": "error", "code": "attr_type", "message": f"Invalid {expected_type} for {tag}.{attr_name}: {attr_value}", "tag": tag})
                for var in BB_VAR_RE.findall(attr_value or ""):
                    consumed_bb_vars.add(var)
                # Nav2 blackboard production heuristics:
                # - ComputePathToPose produces `{path}` (and similar) via its `path` output port.
                if tag in {"ComputePathToPose", "ComputePathThroughPoses"} and attr_name == "path":
                    for var in BB_VAR_RE.findall(attr_value or ""):
                        produced_bb_vars.add(var)

        if tag in {"Sequence", "Fallback", "ReactiveSequence", "ReactiveFallback", "RoundRobin", "PipelineSequence", "KeepRunningUntilFailure", "Repeat", "Inverter", "RecoveryNode"} and len(list(el)) == 0:
            issues.append({"level": "error", "code": "empty_control_node", "message": f"Control node <{tag}> cannot be empty", "tag": tag})

    if len(bt_ids) != len(set(bt_ids)):
        issues.append({"level": "error", "code": "bt_duplicate_id", "message": "BehaviorTree IDs must be unique"})
    if main_tree and main_tree not in set(bt_ids):
        issues.append({"level": "error", "code": "missing_main_tree_def", "message": f"BehaviorTree '{main_tree}' missing"})

    if strict_blackboard:
        missing_vars = sorted(var for var in consumed_bb_vars if var not in produced_bb_vars)
        for var in missing_vars:
            issues.append({"level": "error", "code": "blackboard_unproduced", "message": f"Blackboard variable '{var}' is consumed but not declared"})

    errors = sum(1 for issue in issues if issue["level"] == "error")
    warnings = sum(1 for issue in issues if issue["level"] == "warning")
    return {
        "ok": errors == 0,
        "file": str(xml_path),
        "blackboard": {"external_vars": sorted(produced_bb_vars)},
        "summary": {"issues_total": len(issues), "errors": errors, "warnings": warnings},
        "issues": issues,
    }


def compute_bt_structure_metrics(xml_path: Path) -> Dict[str, Any]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    tags = [el.tag for el in root.iter()]
    subtree_count = sum(1 for t in tags if t == "SubTree")
    base = {"root", "BehaviorTree"}
    control = {
        "Sequence", "Fallback", "ReactiveSequence", "ReactiveFallback", "RoundRobin",
        "PipelineSequence", "RateController", "DistanceController", "SpeedController",
        "KeepRunningUntilFailure", "Repeat", "Inverter", "RecoveryNode",
    }
    control_node_count = sum(1 for t in tags if t in control or t == "SubTree")
    atomic_node_count = sum(1 for t in tags if t not in base and t not in control and t != "SubTree")

    def depth(el: ET.Element, d: int) -> int:
        children = list(el)
        if not children:
            return d
        return max(depth(child, d + 1) for child in children)

    return {
        "bt_depth": depth(root, 1),
        "node_count": len(tags),
        "subtree_count": subtree_count,
        "control_node_count": control_node_count,
        "atomic_node_count": atomic_node_count,
    }
