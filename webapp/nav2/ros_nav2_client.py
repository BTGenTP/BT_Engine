from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
from urllib import error, request


class RosNav2Client:
    def __init__(self, api_base: Optional[str] = None) -> None:
        self.api_base = (api_base or os.getenv("ROS2_CONTROL_API_BASE", "http://localhost:8001")).rstrip("/")

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        req = request.Request(f"{self.api_base}{path}", data=body, headers=headers, method=method.upper())
        try:
            with request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8") if exc.fp else ""
            try:
                details = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                details = {"error": raw or str(exc)}
            raise RuntimeError(details.get("detail") or details.get("error") or str(exc)) from exc
        except error.URLError as exc:
            raise RuntimeError(f"ROS2 control API unreachable at {self.api_base}") from exc

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/api/health")

    def upload_bt(self, *, xml: str, filename: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"xml": xml}
        if filename:
            payload["filename"] = filename
        return self._request("POST", "/api/bt/upload", payload)

    def execute_bt(
        self,
        *,
        xml: Optional[str],
        filename: Optional[str],
        goal_pose: Optional[str],
        goal_name: Optional[str],
        initial_pose: Optional[str],
        allow_invalid: bool,
        start_stack_if_needed: bool,
        restart_navigation: bool,
    ) -> Dict[str, Any]:
        payload = {
            "xml": xml,
            "filename": filename,
            "goal_pose": goal_pose,
            "goal_name": goal_name,
            "initial_pose": initial_pose,
            "allow_invalid": allow_invalid,
            "start_stack_if_needed": start_stack_if_needed,
            "restart_navigation": restart_navigation,
        }
        return self._request("POST", "/api/bt/execute", payload)
