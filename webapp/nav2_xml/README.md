# BT_Engine — Nav2 XML Webapp

Projet autonome pour l’inférence **mission -> BT XML direct -> validation stricte** avec un modèle orienté Nav2 (LoRA local, GGUF local, ou API HF).

## Modes de génération

Un seul mode est actif à la fois, choisi par variables d’environnement (priorité : GGUF > Remote > LoRA).

| Mode | Classe | Variables d’environnement | Prérequis |
|------|--------|---------------------------|-----------|
| **Local LoRA** | `Nav2XmlGenerator` | `NAV2_ADAPTER_DIR`, `NAV2_BASE_MODEL_DIR` (optionnel), `NAV2_MODEL_KEY`, `NAV2_LOAD_IN_4BIT`, `NAV2_ALLOW_DOWNLOADS` | Adapter LoRA + base (HF ou répertoire local), torch, transformers, peft, bitsandbytes |
| **Local GGUF** | `Nav2XmlGeneratorGGUF` | `NAV2_XML_GGUF_PATH` (chemin vers le fichier `.gguf`), `NAV2_MODEL_KEY` | Fichier GGUF (ex. produit par le job SLURM ci‑dessous), `llama-cpp-python` |
| **Remote (HF)** | `Nav2XmlRemoteGenerator` | `NAV2_XML_REMOTE_MODEL_ID`, `HF_TOKEN` ou `NAV2_XML_HF_TOKEN`, optionnel : `NAV2_XML_REMOTE_TIMEOUT_S`, `NAV2_XML_REMOTE_MAX_RETRIES` | Modèle mergé sur Hugging Face, `huggingface_hub` |

- **Priorité** : si `NAV2_XML_GGUF_PATH` pointe vers un fichier existant → mode GGUF ; sinon si `NAV2_XML_REMOTE_MODEL_ID` et token sont renseignés → mode Remote ; sinon → mode Local LoRA.
- L’endpoint `/api/status` renvoie le `provider` actif : `hf_local_peft`, `gguf_local` ou `hf_inference_api`.

## Conversion merged → GGUF (job SLURM)

Pour obtenir un fichier GGUF à utiliser avec le mode Local GGUF :

1. Exécuter d’abord le job de merge LoRA : `repositories/FineTuningOnTelecomCluster/finetune_Nav2_XML/slurm/job_merge_nav2_xml_mistral7b_lora_adapter.sh`.
2. Puis lancer le job de conversion : `repositories/FineTuningOnTelecomCluster/finetune_Nav2_XML/slurm/job_convert_nav2_xml_merged_to_gguf.sh` (même `WORK_DIR` / `OUT_DIR` que le merge).
3. Récupérer le fichier GGUF produit (ex. `nav2_xml_mistral7b_q4_k_m.gguf`) et définir `NAV2_XML_GGUF_PATH` vers ce fichier.

## Architecture

```text
webapp/nav2_xml/
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

```bash
cd repositories/BT_Engine/webapp/nav2_xml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Pour le mode GGUF, placer le fichier `.gguf` et définir par exemple :  
`export NAV2_XML_GGUF_PATH=/chemin/vers/nav2_xml_mistral7b_q4_k_m.gguf`

Pour le mode Remote, définir :  
`export NAV2_XML_REMOTE_MODEL_ID=org/repo-merged` et `export HF_TOKEN=...`

