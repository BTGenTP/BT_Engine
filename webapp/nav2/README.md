# BT_Engine — Nav2 XML Webapp

Projet autonome pour l’inférence **mission -> BT XML direct -> validation stricte** avec un modèle orienté Nav2 (LoRA local, GGUF local, ou API HF).

## Choix du mode dans l’interface

L’utilisateur choisit le **mode de génération** dans la webapp (sélecteur en haut du panneau Mission). Les options sont :

- **Local (base+adapter)** — LoRA local (HF + adapter)
- **Local (GGUF)** — Modèle GGUF chargé en local
- **Remote (Hugging Face)** — Inférence via l’API Hugging Face (Inference API / Endpoints)
- **Remote (OpenAI-compatible)** — Inférence via une API compatible OpenAI (ex. Mistral)

Les options **non disponibles** (prérequis ou variables d’environnement manquants, ou dépendances non installées) sont **grisées** et une courte raison s’affiche. Seuls les modes disponibles sont sélectionnables.

## Dépendances (uv)

Le projet utilise **uv** pour la gestion des dépendances. Les dépendances sont réparties par mode via des **extras** :

| Extra | Mode | Dépendances |
|-------|------|-------------|
| *(aucun)* | Base | `fastapi`, `uvicorn`, `jinja2`, `huggingface_hub` |
| `lora` | Local (base+adapter) | `torch`, `transformers`, `peft`, `bitsandbytes`, `lm-format-enforcer` |
| `gguf` | Local (GGUF) | `llama-cpp-python` |
| `remote` | Remote (HF) | *(utilise `huggingface_hub` du core)* |
| `openai` | Remote (OpenAI-compatible) | `openai` |

**Installation :**

```bash
pip install uv
cd repositories/BT_Engine/webapp/nav2_xml
uv sync --extra lora --extra gguf --extra remote --extra openai   # tous les modes
# ou pour un seul mode :
uv sync --extra lora
uv sync --extra gguf
uv sync --extra remote
uv sync --extra openai
```

Lancer avec uv : `uv run uvicorn app:app --host 0.0.0.0 --port 8000`

## Modes de génération (config)

| Mode | Variables d’environnement | Prérequis |
|------|---------------------------|-----------|
| **Local LoRA** | `NAV2_ADAPTER_DIR`, `NAV2_BASE_MODEL_DIR` (opt.), `NAV2_MODEL_KEY`, `NAV2_LOAD_IN_4BIT`, `NAV2_ALLOW_DOWNLOADS` | Adapter LoRA + base ; installer avec `uv sync --extra lora` |
| **Local GGUF** | `NAV2_XML_GGUF_PATH` (fichier `.gguf`), `NAV2_MODEL_KEY` | Fichier GGUF ; installer avec `uv sync --extra gguf` |
| **Remote (HF)** | `NAV2_XML_REMOTE_MODEL_ID`, `HF_TOKEN` ou `NAV2_XML_HF_TOKEN`, optionnel : `NAV2_XML_REMOTE_TIMEOUT_S`, `NAV2_XML_REMOTE_MAX_RETRIES` | Modèle mergé sur HF ; core suffit. Par défaut, le modèle est `mlatoundji/Mistral-7B-Instruct-v0.2-Nav2BT-XML-merged` et le timeout à 300 s. |
| **Remote (OpenAI-compatible)** | `NAV2_XML_OPENAI_API_KEY` ou `OPENAI_API_KEY`, optionnel : `NAV2_XML_OPENAI_MODEL`, `NAV2_XML_OPENAI_BASE_URL`, `NAV2_XML_OPENAI_TIMEOUT_S`, `NAV2_XML_OPENAI_MAX_RETRIES` | Backend compatible OpenAI (ex. Mistral) ; installer avec `uv sync --extra openai` |

- L’endpoint `/api/status` renvoie la liste `modes` (disponibilité et raison si indisponible) et le `provider` du générateur par défaut.
- `POST /api/generate` accepte un champ optionnel `mode` (`"lora"` | `"gguf"` | `"remote"` | `"openai"`) pour forcer le mode pour cette requête.

### Activer l’inférence Remote (Hugging Face) avec `mlatoundji/Mistral-7B-Instruct-v0.2-Nav2BT-XML-merged`

Le mode **Remote (Hugging Face)** utilise `huggingface_hub.InferenceClient`. Pour activer l’inférence sur le modèle mergé déployé sur Hugging Face :

```bash
export NAV2_XML_REMOTE_MODEL_ID="mlatoundji/Mistral-7B-Instruct-v0.2-Nav2BT-XML-merged"
export HF_TOKEN="hf_xxx"                         # token avec accès au modèle
export NAV2_XML_REMOTE_TIMEOUT_S="300"           # timeout requête inference (secondes)
export NAV2_XML_REMOTE_MAX_RETRIES="2"
```

- L’ancien endpoint `api-inference.huggingface.co` renvoie **410** (déprécié). La librairie `huggingface_hub` utilise par défaut le **router** (`https://router.huggingface.co/hf-inference`). En cas de **404**, le modèle n’est souvent pas disponible en serverless (ex. Mistral 7B) : déployer un **Inference Endpoint** et définir `HF_INFERENCE_ENDPOINT`.
- Pour un **Inference Endpoint** dédié, configurez `HF_INFERENCE_ENDPOINT` :

```bash
export HF_INFERENCE_ENDPOINT="https://your-endpoint.us-east-1.aws.endpoints.huggingface.cloud"
```

Ensuite, dans la webapp, choisissez le mode **Remote (Hugging Face)**. Les réponses sont validées et journalisées comme pour les autres modes.

### Inférence OpenAI-compatible (ex. Mistral)

Le mode **Remote (OpenAI-compatible)** s’appuie sur le client `openai` configuré avec un `base_url` et un `api_key`. Exemple de configuration pour utiliser l’API Mistral :

```bash
export OPENAI_API_KEY="mistral-xxxxx"                            # ou NAV2_XML_OPENAI_API_KEY
export NAV2_XML_OPENAI_BASE_URL="https://api.mistral.ai/v1"
export NAV2_XML_OPENAI_MODEL="mistral-large-latest"
export NAV2_XML_OPENAI_TIMEOUT_S="300"
export NAV2_XML_OPENAI_MAX_RETRIES="2"
```

- Dans l’UI, sélectionnez le mode **Remote (OpenAI-compatible)**.
- En API, envoyez `mode: "openai"` dans le corps de `/api/generate`.

## Conversion merged → GGUF (job SLURM)

Pour obtenir un fichier GGUF à utiliser avec le mode Local GGUF :

1. Exécuter d’abord le job de merge LoRA : `repositories/FineTuningOnTelecomCluster/finetune_Nav2_XML/slurm/job_merge_nav2_xml_mistral7b_lora_adapter.sh`.
2. Puis lancer le job de conversion : `repositories/FineTuningOnTelecomCluster/finetune_Nav2_XML/slurm/job_convert_nav2_xml_merged_to_gguf.sh` (même `WORK_DIR` / `OUT_DIR` que le merge).
3. Récupérer le fichier GGUF produit (ex. `nav2_xml_mistral7b_q4_k_m.gguf`) et définir `NAV2_XML_GGUF_PATH` vers ce fichier.

## Architecture

```text
webapp/nav2_xml/
├── pyproject.toml      # uv, optional-dependencies par mode (lora, gguf, remote)
├── app.py
├── inference.py
├── nav2_pipeline.py
├── bt_validation.py
├── model_registry.py
├── prompting.py
├── run_artifacts.py
├── ros_nav2_client.py
├── constraints/
├── data/
│   ├── bt_nodes_catalog.json
│   └── reference_behavior_trees/
├── templates/
├── static/
└── runs/
```

## Endpoints API

| Route | Méthode | Description |
| --- | --- | --- |
| `/` | GET | Interface HTML |
| `/api/status` | GET | État du backend (provider actif : hf_local_peft / gguf_local / hf_inference_api) |
| `/api/examples` | GET | Exemples de missions Nav2 |
| `/api/generate` | POST | Génère un BT XML direct + rapport strict |
| `/api/validate/xml` | POST | Valide un XML BT |
| `/api/transfer` | POST | Transfère un BT XML au `ROS2_Container` |
| `/api/execute` | POST | Demande l’exécution dans `ROS2_Container` |

## Lancer la webapp

**Avec uv (recommandé) :**

```bash
cd repositories/BT_Engine/webapp/nav2_xml
uv sync --extra lora --extra gguf   # selon les modes souhaités
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

**Avec pip (fallback) :**  
`pip install -r requirements.txt` installe toutes les dépendances (tous modes). Puis `python -m uvicorn app:app --host 0.0.0.0 --port 8000`.

Dans l’interface, choisir le mode de génération (les options non disponibles sont grisées). Pour qu’un mode soit disponible : configurer les variables d’environnement correspondantes et installer l’extra uv associé si besoin.

