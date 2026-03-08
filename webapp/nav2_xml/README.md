# BT_Engine — Nav2 XML Webapp

Projet autonome pour l’inférence **mission -> BT XML direct -> validation stricte** avec un modèle HF + LoRA orienté Nav2.

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
| `/api/status` | GET | État du backend HF + LoRA |
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

