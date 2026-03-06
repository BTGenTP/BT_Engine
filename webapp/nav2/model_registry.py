from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    lora_targets: List[str]
    lora_r: int
    lora_alpha: int
    max_seq_len: int
    batch_size: int
    grad_accum: int
    epochs: int
    lr: float
    chat_template: bool
    response_anchor: str

    @property
    def cache_dir_name(self) -> str:
        namespace, model_name = self.hf_id.split("/", 1)
        return f"models--{namespace}--{model_name}"


MODELS: dict[str, ModelSpec] = {
    "mistral7b": ModelSpec(
        key="mistral7b",
        hf_id="mistralai/Mistral-7B-Instruct-v0.2",
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_r=16,
        lora_alpha=32,
        max_seq_len=1536,
        batch_size=4,
        grad_accum=16,
        epochs=10,
        lr=2e-4,
        chat_template=False,
        response_anchor="[/INST]",
    ),
    "llama3_8b": ModelSpec(
        key="llama3_8b",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_r=16,
        lora_alpha=32,
        max_seq_len=1536,
        batch_size=2,
        grad_accum=16,
        epochs=6,
        lr=2e-4,
        chat_template=True,
        response_anchor="\n### Steps JSON:",
    ),
    "phi2": ModelSpec(
        key="phi2",
        hf_id="microsoft/phi-2",
        lora_targets=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        lora_r=8,
        lora_alpha=16,
        max_seq_len=1024,
        batch_size=4,
        grad_accum=8,
        epochs=8,
        lr=3e-4,
        chat_template=False,
        response_anchor="\n### Steps JSON:",
    ),
}
