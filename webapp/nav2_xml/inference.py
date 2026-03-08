from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from model_registry import MODELS
from nav2_pipeline import build_xml_payload, load_nav2_catalog, write_nav2_xml_run_artifacts
from prompting import build_chat_messages, build_mistral_inst_prompt, build_phi2_prompt
from xml_extraction import extract_root_xml


DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent / "models" / "lora_adapter"
DEFAULT_HF_CACHE_DIR = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"

TEST_MISSIONS_NAV2_XML = [
    "Navigue vers le goal (Nav2), puis attends 2.0 s.",
    "Navigue vers le goal (Nav2), puis tourne de 180° (3.14 rad).",
    "Navigue vers le goal (Nav2), puis efface la costmap locale, puis attends 0.5 s.",
    "Construis un BT de navigation avec recovery: clear costmap, backup et spin si besoin.",
]


def _make_prompt(model_key: str, tokenizer, catalog: Dict[str, Any], mission: str) -> str:
    spec = MODELS[model_key]
    if spec.chat_template:
        msgs = build_chat_messages(mission=mission, catalog=catalog)
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "\n### BT XML:\n"
    if model_key == "phi2":
        prompt, _ = build_phi2_prompt(mission=mission, catalog=catalog)
        return prompt
    prompt, _ = build_mistral_inst_prompt(mission=mission, catalog=catalog)
    return prompt


class Nav2XmlGenerator:
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
            p = self.adapter_dir / name
            if p.exists():
                return p
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
            if self.adapter_weights_path is None or not self.adapter_config_path.exists():
                raise FileNotFoundError(
                    "Nav2 XML LoRA adapter is incomplete. "
                    f"Expected {self.adapter_config_path} and an adapter weight file in {self.adapter_dir}."
                )
            try:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            except Exception as exc:  # pragma: no cover
                self._load_error = str(exc)
                raise RuntimeError("Missing HF/LoRA dependencies. Install torch, transformers, peft and bitsandbytes.") from exc

            spec = self._spec()
            model_ref = self._base_model_reference()
            load_kwargs = {"local_files_only": self.local_files_only, "cache_dir": str(self.hf_cache_dir)}
            tokenizer = AutoTokenizer.from_pretrained(model_ref, use_fast=True, **load_kwargs)
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
                model_kwargs.update({"quantization_config": bnb_config, "device_map": "auto", "dtype": torch.float16})
            else:
                if torch.cuda.is_available():
                    model_kwargs.update({"device_map": "auto", "dtype": torch.float16})

            base = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
            model = PeftModel.from_pretrained(base, self.adapter_dir)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._load_error = None

    def generate(
        self,
        mission: str,
        *,
        constrained: str = "regex",
        max_new_tokens: int = 1024,
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
        if constrained == "regex":
            from constraints.xml_prefix_fn import build_prefix_allowed_tokens_fn

            prefix_fn, _ = build_prefix_allowed_tokens_fn(
                tokenizer,
                self.catalog,
                reference_dir=(Path(__file__).resolve().parent / "data" / "reference_behavior_trees"),
            )

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
        llm_raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        xml = extract_root_xml(llm_raw) or llm_raw

        payload = build_xml_payload(xml, strict_attrs=strict_attrs, strict_blackboard=strict_blackboard)

        result: Dict[str, Any] = {
            "provider": "hf_local_peft",
            "model": self._spec().hf_id,
            "temperature": float(temperature),
            "constraints": {"enabled": constrained != "off", "kind": constrained},
            "prompt": prompt,
            "raw_xml": llm_raw,
            "xml": payload["xml"],
            "valid": bool(payload["valid"]),
            "score": float(payload["score"]),
            "errors": payload["errors"],
            "warnings": payload["warnings"],
            "summary": payload["summary"],
            "validation_report": payload["validation_report"],
            "structure": payload.get("structure") or {},
            "generation_time_s": float(latency_ms) / 1000.0,
            "run_dir": None,
        }

        if write_run:
            run_dir = write_nav2_xml_run_artifacts(
                mission=mission,
                prompt=prompt,
                llm_raw=llm_raw,
                provider="hf_local_peft",
                model_name=self._spec().hf_id,
                temperature=float(temperature),
                constraints_kind=str(constrained),
                strict_attrs=bool(strict_attrs),
                strict_blackboard=bool(strict_blackboard),
                latency_ms=int(latency_ms),
                xml_payload=payload,
            )
            result["run_dir"] = run_dir
        return result


def build_nav2_xml_generator_from_env() -> Nav2XmlGenerator:
    model_key = os.getenv("NAV2_MODEL_KEY", "mistral7b")
    adapter_dir = os.getenv("NAV2_ADAPTER_DIR", None)
    base_model_dir = os.getenv("NAV2_BASE_MODEL_DIR", None)
    hf_cache_dir = os.getenv("NAV2_HF_CACHE_DIR", None)
    load_in_4bit = os.getenv("NAV2_LOAD_IN_4BIT", "1").strip() not in {"0", "false", "False"}
    allow_downloads = os.getenv("NAV2_ALLOW_DOWNLOADS", "0").strip() in {"1", "true", "True"}
    return Nav2XmlGenerator(
        model_key=model_key,
        adapter_dir=adapter_dir,
        base_model_dir=base_model_dir,
        hf_cache_dir=hf_cache_dir,
        load_in_4bit=load_in_4bit,
        allow_downloads=allow_downloads,
    )

