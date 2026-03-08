"""
Standalone Nav2 XML-direct web application.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from bt_validation import validate_bt_xml
from inference import TEST_MISSIONS_NAV2_XML, build_nav2_xml_generator_from_env
from nav2_pipeline import load_nav2_catalog
from ros_nav2_client import RosNav2Client

APP_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Nav2 BT Generator (XML direct)")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

nav2_generator = build_nav2_xml_generator_from_env()
nav2_catalog = load_nav2_catalog()
ros_nav2_client = RosNav2Client()

LOG_DIR = APP_DIR / "runs" / "_server_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
_logger = logging.getLogger("nav2_xml_webapp")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "requests.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    _logger.addHandler(fh)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    req_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        dt_ms = int((time.perf_counter() - t0) * 1000)
        _logger.exception("rid=%s %s %s -> EXC %dms err=%s", req_id, request.method, request.url.path, dt_ms, str(exc))
        raise
    dt_ms = int((time.perf_counter() - t0) * 1000)
    _logger.info("rid=%s %s %s -> %s %dms", req_id, request.method, request.url.path, getattr(response, "status_code", "?"), dt_ms)
    response.headers["X-Request-Id"] = req_id
    return response


class GenerateRequest(BaseModel):
    mission: str
    constrained: str = "regex"
    max_new_tokens: int = 1024
    temperature: float = 0.0
    write_run: bool = True
    strict_attrs: bool = True
    strict_blackboard: bool = True


class ValidateRequest(BaseModel):
    xml: str


class ExecuteRequest(BaseModel):
    xml: Optional[str] = None
    filename: Optional[str] = None
    goal_pose: Optional[str] = None
    goal_name: Optional[str] = None
    initial_pose: Optional[str] = "0.0,0.0,0.0"
    allow_invalid: bool = False
    start_stack_if_needed: bool = True
    restart_navigation: bool = True


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def status():
    payload = nav2_generator.status()
    payload["ros2_control_api_base"] = ros_nav2_client.api_base
    return payload


@app.get("/api/examples")
async def examples():
    return {"missions": TEST_MISSIONS_NAV2_XML}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    try:
        return await asyncio.to_thread(
            nav2_generator.generate,
            req.mission,
            constrained=req.constrained,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            write_run=req.write_run,
            strict_attrs=req.strict_attrs,
            strict_blackboard=req.strict_blackboard,
        )
    except Exception as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})


@app.post("/api/validate/xml")
async def validate_xml(req: ValidateRequest):
    import tempfile

    with tempfile.NamedTemporaryFile("w+", suffix=".xml", delete=False, encoding="utf-8") as tmp:
        tmp.write(req.xml)
        tmp.flush()
        tmp_path = Path(tmp.name)
    try:
        report = validate_bt_xml(xml_path=tmp_path, strict_attrs=True, strict_blackboard=True)
    finally:
        tmp_path.unlink(missing_ok=True)

    errors = []
    warnings = []
    for issue in report.get("issues", []) or []:
        line = f"[{issue.get('code', 'unknown')}] {issue.get('message', '')}".strip()
        if str(issue.get("level", "error")).lower() == "warning":
            warnings.append(line)
        else:
            errors.append(line)

    return {
        "xml": req.xml,
        "valid": bool(report.get("ok")),
        "score": 1.0 if bool(report.get("ok")) else 0.0,
        "errors": errors,
        "warnings": warnings,
        "summary": "Strict validator passed" if bool(report.get("ok")) else "Strict validator failed",
        "validation_report": report,
    }


@app.post("/api/transfer")
async def transfer(req: ExecuteRequest):
    if not req.xml:
        return JSONResponse(status_code=422, content={"error": "xml is required for transfer"})
    try:
        return await asyncio.to_thread(ros_nav2_client.upload_bt, xml=req.xml, filename=req.filename)
    except Exception as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})


@app.post("/api/execute")
async def execute(req: ExecuteRequest):
    try:
        return await asyncio.to_thread(
            ros_nav2_client.execute_bt,
            xml=req.xml,
            filename=req.filename,
            goal_pose=req.goal_pose,
            goal_name=req.goal_name,
            initial_pose=req.initial_pose,
            allow_invalid=req.allow_invalid,
            start_stack_if_needed=req.start_stack_if_needed,
            restart_navigation=req.restart_navigation,
        )
    except Exception as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})

