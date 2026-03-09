#!/usr/bin/env python3
"""
Test minimal d'inférence HF via InferenceClient (huggingface_hub).
Même client que la webapp : router ou HF_INFERENCE_ENDPOINT selon l'env.

En cas de 404 : le modèle n'est souvent pas sur le serverless. Pour un 7B,
déployer un Inference Endpoint sur le Hub puis définir HF_INFERENCE_ENDPOINT
(pointant vers l'URL de l'endpoint).
"""
import os
import sys

try:
    from huggingface_hub import InferenceClient
    from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError
except ImportError:
    print("pip install huggingface_hub")
    sys.exit(1)

MODEL_ID = os.environ.get(
    "NAV2_XML_REMOTE_MODEL_ID",
    "mlatoundji/Mistral-7B-Instruct-v0.2-Nav2BT-XML-merged",
)
TOKEN = os.environ.get("NAV2_XML_HF_TOKEN", "").strip() or os.environ.get("HF_TOKEN", "").strip()
TIMEOUT_S = float(os.environ.get("NAV2_XML_REMOTE_TIMEOUT_S", "300"))
# Si défini, utiliser l'URL de l'endpoint comme "model" (client appelle cette URL directement)
ENDPOINT_URL = os.environ.get("HF_INFERENCE_ENDPOINT", "https://xdxj73b4sj2u0ojn.us-east-1.aws.endpoints.huggingface.cloud").strip() or None

if not TOKEN:
    print("Définir HF_TOKEN ou NAV2_XML_HF_TOKEN")
    sys.exit(1)

# Comme la webapp : model = URL d'endpoint si fournie, sinon repo_id (router/serverless)
model = ENDPOINT_URL if ENDPOINT_URL else MODEL_ID
print(f"InferenceClient(model={model!r}, timeout={TIMEOUT_S}s)")

client = InferenceClient(model=model, token=TOKEN, timeout=TIMEOUT_S)
prompt = "Écris un BT Nav2 minimal : NavigateToPose puis Wait."

try:
    out = client.text_generation(
        prompt,
        max_new_tokens=256,
        return_full_text=False,
    )
    print("Réponse (extrait):", (out[:400] + "..." if len(out) > 400 else out))
    print("OK")
except HfHubHTTPError as e:
    status = getattr(e.response, "status_code", None)
    body = (getattr(e.response, "text", None) or "")[:500]
    print("HfHubHTTPError:", status, body or "(empty)")
    if status == 404:
        print(
            "404: Modèle introuvable sur le router. Déployer un Inference Endpoint et définir HF_INFERENCE_ENDPOINT."
        )
    sys.exit(1)
except InferenceTimeoutError as e:
    print("InferenceTimeoutError:", e)
    sys.exit(1)
except StopIteration:
    print(
        "StopIteration: le modèle n'a pas de 'inference provider mapping' sur le Hub (router serverless)."
        " Déployer un Inference Endpoint puis définir HF_INFERENCE_ENDPOINT, ou utiliser un modèle listé sur le router."
    )
    sys.exit(1)
except Exception as e:
    print(type(e).__name__, ":", e)
    sys.exit(1)
