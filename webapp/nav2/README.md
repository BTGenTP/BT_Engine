# BT_Engine — Nav2 Webapp

Projet autonome pour l’inférence `mission -> steps JSON -> BT XML -> validation stricte` avec un modèle HF + LoRA orienté Nav2.

## Architecture

```text
webapp/nav2/
├── app.py
├── inference.py
├── nav2_pipeline.py
├── bt_validation.py
├── json_to_xml.py
├── steps_parsing.py
├── catalog_io.py
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
├── models/
│   └── lora_adapter/
└── runs/
```

## Endpoints API

| Route | Méthode | Description |
| --- | --- | --- |
| `/` | GET | Interface HTML |
| `/api/status` | GET | État du backend HF + LoRA |
| `/api/examples` | GET | Exemples de missions Nav2 |
| `/api/generate` | POST | Génère `steps JSON`, BT XML et rapport strict |
| `/api/validate/steps` | POST | Valide des `steps JSON` |
| `/api/steps-to-xml` | POST | Convertit des `steps JSON` vers XML |
| `/api/validate/xml` | POST | Valide un XML BT |
| `/api/transfer` | POST | Transfère un BT XML au `ROS2_Container` |
| `/api/execute` | POST | Demande l’exécution dans `ROS2_Container` |

## Déploiement du modèle + LoRA

Le projet charge directement l’adapter LoRA avec `transformers + peft`.
Par défaut, le backend fonctionne maintenant en mode hors-ligne:
- aucun téléchargement Hugging Face n'est autorisé au premier appel
- il faut fournir un modèle de base déjà présent localement, soit via un dossier explicite, soit via un cache HF prérempli

Il n'y a pas de conversion GGUF dans le flux par défaut. Une conversion GGUF ne doit etre envisagée qu'en plan B si le backend HF local reste trop lourd.

### Étape 1 — Précharger le modèle de base hors-ligne

Option recommandée: télécharger une fois le modèle de base dans un dossier dédié hors du repo.

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    local_dir="/opt/models/Mistral-7B-Instruct-v0.2",
    local_dir_use_symlinks=False,
)
PY
```

Ensuite, pointer explicitement ce dossier:

```bash
export NAV2_BASE_MODEL_DIR=/opt/models/Mistral-7B-Instruct-v0.2
```

Alternative:
- laisser `NAV2_BASE_MODEL_DIR` vide
- préremplir `~/.cache/huggingface/hub`
- garder `NAV2_ALLOW_DOWNLOADS=0`

### Étape 2 — Placer l’adapter LoRA hors du repo

Depuis votre dépôt d’entraînement ou depuis le cluster :

```bash
mkdir -p /opt/nav2/lora_adapter
rsync -av /chemin/vers/lora_adapter/ /opt/nav2/lora_adapter/
```

Puis :

```bash
export NAV2_ADAPTER_DIR=/opt/nav2/lora_adapter
```

Le dossier `models/lora_adapter/` du repo peut rester vide, avec seulement `.gitkeep`.

### Étape 3 — Installer les dépendances

```bash
cd repositories/BT_Engine/webapp/nav2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Étape 4 — Variables d’environnement

```bash
export NAV2_MODEL_KEY=mistral7b
export NAV2_BASE_MODEL_DIR=/opt/models/Mistral-7B-Instruct-v0.2
export NAV2_ADAPTER_DIR=/opt/nav2/lora_adapter
export NAV2_HF_CACHE_DIR="${HOME}/.cache/huggingface/hub"
export NAV2_LOAD_IN_4BIT=1
export NAV2_ALLOW_DOWNLOADS=0
export ROS2_CONTROL_API_BASE=http://localhost:8001
```

Le mode par défaut est hors-ligne. Pour un bootstrap ponctuel seulement, vous pouvez autoriser un téléchargement:

```bash
export NAV2_ALLOW_DOWNLOADS=1
```

### Étape 5 — Lancer la webapp

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

L’interface est accessible sur [http://localhost:8000](http://localhost:8000).

### Vérification rapide

L’endpoint `GET /api/status` expose l’état du backend:
- `adapter_ready`
- `base_model_available`
- `local_files_only`
- `downloads_allowed`

## Goal Nav2

Le `goal Nav2` n’est pas généré par le modèle. Le modèle produit seulement l’intention de navigation via la step `NavigateToGoalWithReplanningAndRecovery`.

La cible réelle est injectée à l’exécution par Nav2 via la blackboard `{goal}` :

- soit comme pose explicite `x,y,theta`
- soit comme nom logique résolu par `ROS2_Container/runtime/BT_Navigator/config/locations.yaml`

## Contrat avec `ROS2_Container`

Le contrat avec le conteneur ROS est strictement HTTP :

- transfert de BT XML
- démarrage ou réinitialisation de la simulation
- redémarrage sélectif de la navigation
- envoi d’un goal

Le projet n’importe aucun code Python depuis `ROS2_Container`.
