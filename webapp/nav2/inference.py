from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from model_registry import MODELS
from nav2_pipeline import build_xml_from_steps, load_nav2_catalog, parse_steps_payload, write_nav2_run_artifacts
from prompting import build_chat_messages, build_mistral_inst_prompt, build_phi2_prompt


DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent / "models" / "lora_adapter"
DEFAULT_HF_CACHE_DIR = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
DEFAULT_REMOTE_TIMEOUT_S = 120.0

TEST_MISSIONS_NAV2 = [
    "Navigue vers le goal (Nav2), puis attends 2.0 s.",
    "Navigue vers le goal (Nav2), puis tourne de 180° (3.14 rad).",
    "Navigue vers le goal (Nav2), puis recule de 0.3 m à 0.08 m/s.",
    "Attends 1.0 s puis tourne de 90° (1.57 rad).",
    "Navigue vers le goal (Nav2), puis efface la costmap locale, puis attends 0.5 s.",
]


def _make_prompt(model_key: str, tokenizer, catalog: Dict[str, Any], mission: str) -> str:
    spec = MODELS[model_key]
    if spec.chat_template:
        msgs = build_chat_messages(mission=mission, catalog=catalog)
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "\n### Steps JSON:\n"
    if model_key == "phi2":
        prompt, _ = build_phi2_prompt(mission=mission, catalog=catalog)
        return prompt
    prompt, _ = build_mistral_inst_prompt(mission=mission, catalog=catalog)
    return prompt


def _make_prompt_remote(model_key: str, *, catalog: Dict[str, Any], mission: str) -> str:
    """
    Tokenizer-free prompt builder for remote inference.

    Notes:
    - For chat-template models, we would normally need `tokenizer.apply_chat_template()`.
      To avoid any HF downloads at runtime, remote mode currently supports only non-chat-template specs.
    """
    spec = MODELS[model_key]
    if spec.chat_template:
        raise RuntimeError(f"Remote inference does not support chat_template models yet (model_key={model_key}). Use NAV2_MODEL_KEY=mistral7b.")
    if model_key == "phi2":
        prompt, _ = build_phi2_prompt(mission=mission, catalog=catalog)
        return prompt
    prompt, _ = build_mistral_inst_prompt(mission=mission, catalog=catalog)
    return prompt


class Nav2Generator:
    def __init__(
        self,
        *,
        model_key: str = "mistral7b",
        adapter_dir: str | Path | None = None,
        base_model_dir: str | Path | None = None,
        hf_cache_dir: str | Path | None = None,
        load_in_4bit: bool = True,
        allow_downloads: bool = False,
    ) -> None:
        self.model_key = model_key
        self.adapter_dir = Path(adapter_dir or DEFAULT_ADAPTER_DIR).expanduser().resolve()
        self.base_model_dir = Path(base_model_dir).expanduser().resolve() if base_model_dir else None
        self.hf_cache_dir = Path(hf_cache_dir or DEFAULT_HF_CACHE_DIR).expanduser().resolve()
        self.load_in_4bit = bool(load_in_4bit)
        self.allow_downloads = bool(allow_downloads)
        self.catalog = load_nav2_catalog()
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._load_error: Optional[str] = None

    @property
    def configured(self) -> bool:
        return self.adapter_weights_path is not None

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def adapter_weights_path(self) -> Optional[Path]:
        for name in ("adapter_model.safetensors", "adapter_model.bin"):
            candidate = self.adapter_dir / name
            if candidate.exists():
                return candidate
        return None

    @property
    def adapter_config_path(self) -> Path:
        return self.adapter_dir / "adapter_config.json"

    @property
    def local_files_only(self) -> bool:
        return not self.allow_downloads

    def _spec(self):
        return MODELS[self.model_key]

    def _cached_model_root(self) -> Path:
        return self.hf_cache_dir / self._spec().cache_dir_name

    def _base_model_reference(self) -> str:
        return str(self.base_model_dir) if self.base_model_dir else self._spec().hf_id

    def _base_model_available(self) -> bool:
        if self.base_model_dir is not None:
            return self.base_model_dir.exists() and (self.base_model_dir / "config.json").exists()
        return self._cached_model_root().exists()

    def _offline_load_hint(self) -> str:
        return (
            "Provide NAV2_BASE_MODEL_DIR=/path/to/the/base-model directory, "
            "or pre-download the Hugging Face cache, "
            "or set NAV2_ALLOW_DOWNLOADS=1 for a one-time bootstrap."
        )

    def status(self) -> Dict[str, Any]:
        return {
            "configured": self.configured,
            "loaded": self.loaded,
            "provider": "hf_local_peft",
            "model_key": self.model_key,
            "model_id": self._spec().hf_id,
            "adapter_dir": str(self.adapter_dir),
            "adapter_ready": self.configured,
            "base_model_dir": str(self.base_model_dir) if self.base_model_dir else None,
            "base_model_source": self._base_model_reference(),
            "base_model_available": self._base_model_available(),
            "hf_cache_dir": str(self.hf_cache_dir),
            "local_files_only": self.local_files_only,
            "downloads_allowed": self.allow_downloads,
            "load_error": self._load_error,
        }

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            adapter_weights_path = self.adapter_weights_path
            if adapter_weights_path is None or not self.adapter_config_path.exists():
                raise FileNotFoundError(
                    "Nav2 LoRA adapter is incomplete. "
                    f"Expected {self.adapter_config_path} and an adapter weight file in {self.adapter_dir}."
                )
            try:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            except Exception as exc:  # pragma: no cover
                self._load_error = str(exc)
                raise RuntimeError("Missing HF/LoRA dependencies for Nav2 mode. Install torch, transformers, peft and bitsandbytes.") from exc

            spec = self._spec()
            model_ref = self._base_model_reference()
            load_kwargs = {
                "local_files_only": self.local_files_only,
                "cache_dir": str(self.hf_cache_dir),
            }
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_ref, use_fast=True, **load_kwargs)
            except OSError as exc:
                self._load_error = str(exc)
                if self.local_files_only:
                    raise RuntimeError(
                        "Base tokenizer is not available locally for offline loading. "
                        + self._offline_load_hint()
                    ) from exc
                raise
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_kwargs: Dict[str, Any] = dict(load_kwargs)
            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs.update(
                    {
                        "quantization_config": bnb_config,
                        "device_map": "auto",
                        "dtype": torch.float16,
                    }
                )
            else:
                if torch.cuda.is_available():
                    model_kwargs.update({"device_map": "auto", "dtype": torch.float16})
            try:
                base = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
            except OSError as exc:
                self._load_error = str(exc)
                if self.local_files_only:
                    raise RuntimeError(
                        "Base model weights are not available locally for offline loading. "
                        + self._offline_load_hint()
                    ) from exc
                raise

            model = PeftModel.from_pretrained(base, self.adapter_dir)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._load_error = None

    def generate(
        self,
        mission: str,
        *,
        constrained: str = "jsonschema",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        write_run: bool = True,
        strict_attrs: bool = True,
        strict_blackboard: bool = True,
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        tokenizer = self._tokenizer
        model = self._model
        prompt = _make_prompt(self.model_key, tokenizer, self.catalog, mission)

        prefix_fn = None
        if constrained == "jsonschema":
            from constraints.steps_prefix_fn import build_prefix_allowed_tokens_fn

            prefix_fn = build_prefix_allowed_tokens_fn(tokenizer, self.catalog)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with self._lock:
            t0 = time.perf_counter()
            output = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(float(temperature) > 0.0),
                temperature=float(temperature) if float(temperature) > 0.0 else 1.0,
                pad_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_fn,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000.0)

        gen_ids = output[0][inputs["input_ids"].shape[1] :]
        raw_steps = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        parsed_payload = parse_steps_payload(raw_steps, catalog=self.catalog)

        result: Dict[str, Any] = {
            "provider": "hf_local_peft",
            "model": MODELS[self.model_key].hf_id,
            "model_key": self.model_key,
            "mission": mission,
            "prompt": prompt,
            "raw_steps": raw_steps,
            "steps": parsed_payload["steps"],
            "steps_json": parsed_payload["steps_json"],
            "steps_valid": parsed_payload["ok"],
            "parse_error": parsed_payload["error_message"],
            "parse_error_counters": parsed_payload["error_counters"],
            "generation_time_s": round(latency_ms / 1000.0, 2),
            "constraint_mode": constrained,
        }

        parsed = parsed_payload["_parsed"]
        if parsed.ok and parsed.steps:
            xml_payload = build_xml_from_steps(parsed.steps, catalog=self.catalog, strict_attrs=strict_attrs, strict_blackboard=strict_blackboard)
            result.update(xml_payload)
            if write_run:
                result["run_dir"] = write_nav2_run_artifacts(
                    mission=mission,
                    prompt=prompt,
                    llm_raw=raw_steps,
                    parsed=parsed,
                    provider="hf_local_peft",
                    model_name=MODELS[self.model_key].hf_id,
                    temperature=float(temperature),
                    constraints_kind=constrained,
                    strict_attrs=strict_attrs,
                    strict_blackboard=strict_blackboard,
                    latency_ms=latency_ms,
                    xml_payload=xml_payload,
                )
        else:
            result.update(
                {
                    "xml": "",
                    "validation_report": {"ok": False, "issues": []},
                    "valid": False,
                    "score": 0.0,
                    "errors": parsed_payload["errors"],
                    "warnings": [],
                    "summary": parsed_payload["error_message"] or "Steps JSON invalid",
                    "structure": {},
                }
            )
            if write_run:
                result["run_dir"] = write_nav2_run_artifacts(
                    mission=mission,
                    prompt=prompt,
                    llm_raw=raw_steps,
                    parsed=parsed,
                    provider="hf_local_peft",
                    model_name=MODELS[self.model_key].hf_id,
                    temperature=float(temperature),
                    constraints_kind=constrained,
                    strict_attrs=strict_attrs,
                    strict_blackboard=strict_blackboard,
                    latency_ms=latency_ms,
                )
        return result


class Nav2RemoteGenerator:
    """
    Remote inference (Hugging Face Inference API) for a merged LoRA model.

    Goal: preserve LoRA behavior while avoiding local VRAM/weight downloads.
    """

    def __init__(
        self,
        *,
        model_id: str,
        token: str,
        model_key: str = "mistral7b",
        timeout_s: float = DEFAULT_REMOTE_TIMEOUT_S,
        max_retries: int = 2,
    ) -> None:
        self.model_id = (model_id or "").strip()
        self.token = (token or "").strip()
        self.model_key = model_key
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.catalog = load_nav2_catalog()

    @property
    def configured(self) -> bool:
        return bool(self.model_id) and bool(self.token)

    @property
    def loaded(self) -> bool:
        # No local weights to load; treat as ready when configured.
        return self.configured

    def status(self) -> Dict[str, Any]:
        return {
            "configured": self.configured,
            "loaded": self.loaded,
            "provider": "hf_inference_api",
            "model_key": self.model_key,
            "model": self.model_id,
            "remote_timeout_s": self.timeout_s,
            "remote_max_retries": self.max_retries,
        }

    def _text_generation(self, *, prompt: str, max_new_tokens: int, temperature: float) -> str:
        try:
            from huggingface_hub import InferenceClient
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("huggingface_hub is required for remote inference. Install huggingface_hub>=0.26.") from exc

        client = InferenceClient(model=self.model_id, token=self.token, timeout=self.timeout_s)
        do_sample = bool(float(temperature) > 0.0)
        kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "return_full_text": False,
        }
        if do_sample:
            kwargs["temperature"] = float(temperature)
        return str(client.text_generation(prompt, **kwargs)).strip()

    def _repair_prompt(self, *, base_prompt: str, parse_error: str | None) -> str:
        err = (parse_error or "").strip()
        tail = (
            "\n\nIMPORTANT:\n"
            "- Output ONLY a JSON array (no markdown fences, no prose).\n"
            "- Each item must be an object: {\"skill\": <string>, \"params\": <object>}.\n"
        )
        if err:
            tail += f"- Previous parse error: {err}\n"
        tail += "\nJSON:\n"
        return base_prompt + tail

    def generate(
        self,
        mission: str,
        *,
        constrained: str = "jsonschema",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        write_run: bool = True,
        strict_attrs: bool = True,
        strict_blackboard: bool = True,
    ) -> Dict[str, Any]:
        if not self.configured:
            raise RuntimeError("Remote inference is not configured. Provide NAV2_REMOTE_MODEL_ID and HF_TOKEN (or NAV2_HF_TOKEN).")

        # Remote APIs do not support our local prefix constraints; keep arg for compatibility.
        _ = constrained

        base_prompt = _make_prompt_remote(self.model_key, catalog=self.catalog, mission=mission)
        prompt = base_prompt

        attempts = 0
        raw_steps = ""
        latency_ms = 0
        parsed_payload: Dict[str, Any] = {}
        while True:
            attempts += 1
            t0 = time.perf_counter()
            raw_steps = self._text_generation(prompt=prompt, max_new_tokens=int(max_new_tokens), temperature=float(temperature))
            latency_ms = int((time.perf_counter() - t0) * 1000.0)
            parsed_payload = parse_steps_payload(raw_steps, catalog=self.catalog)
            if parsed_payload.get("ok"):
                break
            if attempts > self.max_retries + 1:
                break
            prompt = self._repair_prompt(base_prompt=prompt, parse_error=parsed_payload.get("error_message"))

        result: Dict[str, Any] = {
            "provider": "hf_inference_api",
            "model": self.model_id,
            "model_key": self.model_key,
            "mission": mission,
            "prompt": base_prompt,
            "raw_steps": raw_steps,
            "steps": parsed_payload.get("steps", []),
            "steps_json": parsed_payload.get("steps_json", ""),
            "steps_valid": bool(parsed_payload.get("ok")),
            "parse_error": parsed_payload.get("error_message"),
            "parse_error_counters": parsed_payload.get("error_counters"),
            "generation_time_s": round(latency_ms / 1000.0, 2),
            "constraint_mode": constrained,
            "remote_attempts": attempts,
        }

        parsed = parsed_payload.get("_parsed")
        if getattr(parsed, "ok", False) and getattr(parsed, "steps", None):
            xml_payload = build_xml_from_steps(parsed.steps, catalog=self.catalog, strict_attrs=strict_attrs, strict_blackboard=strict_blackboard)  # type: ignore[arg-type]
            result.update(xml_payload)
            if write_run:
                result["run_dir"] = write_nav2_run_artifacts(
                    mission=mission,
                    prompt=base_prompt,
                    llm_raw=raw_steps,
                    parsed=parsed,
                    provider="hf_inference_api",
                    model_name=self.model_id,
                    temperature=float(temperature),
                    constraints_kind=constrained,
                    strict_attrs=strict_attrs,
                    strict_blackboard=strict_blackboard,
                    latency_ms=latency_ms,
                    xml_payload=xml_payload,
                )
        else:
            result.update(
                {
                    "xml": "",
                    "validation_report": {"ok": False, "issues": []},
                    "valid": False,
                    "score": 0.0,
                    "errors": parsed_payload.get("errors", []),
                    "warnings": [],
                    "summary": parsed_payload.get("error_message") or "Steps JSON invalid",
                    "structure": {},
                }
            )
            if write_run and parsed is not None:
                result["run_dir"] = write_nav2_run_artifacts(
                    mission=mission,
                    prompt=base_prompt,
                    llm_raw=raw_steps,
                    parsed=parsed,
                    provider="hf_inference_api",
                    model_name=self.model_id,
                    temperature=float(temperature),
                    constraints_kind=constrained,
                    strict_attrs=strict_attrs,
                    strict_blackboard=strict_blackboard,
                    latency_ms=latency_ms,
                )
        return result


def _env_truthy(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val not in {"0", "false", "False", ""}


def build_nav2_generator_from_env():
    provider = (os.getenv("NAV2_PROVIDER") or "auto").strip().lower()
    model_key = os.getenv("NAV2_MODEL_KEY", "mistral7b")

    # Remote (recommended)
    remote_model_id = os.getenv("NAV2_REMOTE_MODEL_ID", "").strip()
    token = (os.getenv("NAV2_HF_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
    remote_timeout_s = float(os.getenv("NAV2_REMOTE_TIMEOUT_S", str(DEFAULT_REMOTE_TIMEOUT_S)))
    remote_max_retries = int(os.getenv("NAV2_REMOTE_MAX_RETRIES", "2"))

    remote_ready = bool(remote_model_id) and bool(token)
    if provider in {"auto", "hf_inference_api", "remote"} and remote_ready:
        return Nav2RemoteGenerator(
            model_id=remote_model_id,
            token=token,
            model_key=model_key,
            timeout_s=remote_timeout_s,
            max_retries=remote_max_retries,
        )
    if provider in {"hf_inference_api", "remote"} and not remote_ready:
        return Nav2RemoteGenerator(
            model_id=remote_model_id,
            token=token,
            model_key=model_key,
            timeout_s=remote_timeout_s,
            max_retries=remote_max_retries,
        )

    # Local HF + PEFT (fallback)
    adapter_dir = os.getenv("NAV2_ADAPTER_DIR") or str(DEFAULT_ADAPTER_DIR)
    base_model_dir = os.getenv("NAV2_BASE_MODEL_DIR")
    hf_cache_dir = os.getenv("NAV2_HF_CACHE_DIR") or str(DEFAULT_HF_CACHE_DIR)
    load_in_4bit = _env_truthy("NAV2_LOAD_IN_4BIT", default=True)
    allow_downloads = _env_truthy("NAV2_ALLOW_DOWNLOADS", default=False)
    return Nav2Generator(
        model_key=model_key,
        adapter_dir=adapter_dir,
        base_model_dir=base_model_dir,
        hf_cache_dir=hf_cache_dir,
        load_in_4bit=load_in_4bit,
        allow_downloads=allow_downloads,
    )
