# CropAI — Agricultural Intelligence Platform

Precision agriculture platform for crop phenotyping and plant disease diagnostics, built for the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) (Physical AI Track).

Uses **NVIDIA Cosmos Transfer 2.5** for synthetic training data generation and a trained **ResNet-50** vision model (exported to ONNX/TensorRT) for real-time crop classification. Integrates with [Learner's Permit](https://github.com/thiagopari/learners-permit) for Bayesian skill licensing — the model is only trusted within the conditions it was tested under.

## Architecture

```
TRAINING (Nebius GPU)
┌───────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  PlantVillage     │────▶│  Cosmos Transfer 2.5 │────▶│  train.py        │
│  Seed Dataset     │     │  Synthetic Variants   │     │  ResNet-50       │
│  (50-200 images)  │     │  (5k-10k images)      │     │  Fine-tune       │
└───────────────────┘     └─────────────────────┘     └────────┬─────────┘
                                                               │
                                                    phenotype_model.onnx
                                                               │
INFERENCE (Edge / Nebius Endpoint)                             │
┌─────────────────────┐     ┌──────────────────────────┐       │
│   Dashboard (HTML)  │────▶│  FastAPI Backend          │◀──────┘
│   Alpine.js + Chart │ REST│  ONNX Runtime inference   │
│                     │◀────│  Score mapper → 6 cats    │
└─────────────────────┘     └────────────┬─────────────┘
                                         │
LICENSING (Learner's Permit)             │
┌─────────────────────┐     ┌────────────▼─────────────┐
│  Bayesian Gate      │◀────│  phenotype_skill.py      │
│  P(rate≥0.8)≥0.95   │     │  Each inference = trial   │
│  EARN/REFUSE/REVOKE │     │  pass/fail → gate         │
└─────────────────────┘     └──────────────────────────┘
```

## NVIDIA Stack

| Component | Role |
|-----------|------|
| **Cosmos Transfer 2.5** | Generate synthetic crop image variants (lighting, weather, cultivar shifts) from a small seed dataset |
| **ResNet-50 → ONNX → TensorRT** | Trained phenotype classifier, edge-deployable on Jetson |
| **Nebius Serverless Jobs** | Model training at scale |
| **Nebius Serverless Endpoints** | Real-time inference serving |
| **Nemotron** | Planner + explainer in Learner's Permit (upstream repo) |
| **GR00T N1.7** | Pick-and-place skill in Learner's Permit (upstream repo) |

## Features

### Crop Phenotype Analyzer

Trained vision model classifies crops into 3 phenotype classes:

| Class | Dashboard Mapping | Score Range |
|-------|-------------------|-------------|
| **Healthy** | Genetically Desirable | 0.75 – 0.95 |
| **Stressed** | Standard | 0.40 – 0.65 |
| **Diseased** | Suboptimal | 0.10 – 0.35 |

The score mapper expands the 3-class prediction into 6 official judging categories:

1. Plant Architecture & Morphology — leaf angle, stalk strength, root system, branching habit
2. Yield & Reproductive Traits — harvest index, kernel density, fertility efficiency
3. Biotic Stress Resistance — necrotic threshold, pest deterrence, hypersensitive response
4. Abiotic Stress Resilience — stomatal regulation, stay-green trait, salinity tolerance
5. Quality & Phenology — flowering synchrony, photoperiod sensitivity, pericarp toughness
6. Harvest & Processing Suitability — non-shattering pods, detachment force

**Trait preference filtering** — compare AI results against user-defined minimum thresholds to approve or reject candidates for breeding programs.

### Plant Pathology & Disease Diagnostics

Same model, different output mapping — identifies pathogens, deficiencies, and environmental stress across 4 diagnostic vectors:

- **Fungal pathogens** — rust, mildew, leaf spot, root rot
- **Bacterial & viral infections** — blight, mosaic distortion, systemic necrosis
- **Pest & insect damage** — chewing defoliation, sucking pest chlorosis, borer wounds
- **Abiotic & nutritional stress** — N/P deficiency, sunscald, chemical drift burn

### Learner's Permit Integration

The phenotype model is a **licensed skill** in the Learner's Permit framework. Each inference is a trial — the Bayesian gate decides whether the model is trustworthy under specific conditions.

**Licence scope dimensions:**
- Cultivar (mandatory) — e.g., `tomato_truss`
- Light window (mandatory) — e.g., `noon_diffuse`
- Camera rig + model hash (mandatory)
- Occlusion class, canopy stage (optional)

**Four licensing beats demonstrated:**

| Beat | What happens |
|------|-------------|
| **EARN** | Model scores 90%+ on in-scope tomato data → licence granted |
| **REFUSE** | Same model tested on out-of-scope crop (pepper) → accuracy drops → licence refused |
| **BLOCK** | Request for a cultivar outside the licensed scope → Cedar denies |
| **REVOKE** | Performance degrades under dusk lighting or late-season canopy → licence revoked |

### Dashboard

Single-page interface built with Tailwind CSS, Alpine.js, and Chart.js:
- Image upload (drag-and-drop) and sample crop presets
- Telemetry metadata form (GPS, NDVI, soil moisture, temperature)
- Trait threshold sliders for phenotype selection criteria
- Radar chart (phenotype) and bar chart (pathology) visualizations
- Historical evaluation log with search and reload
- Falls back to mock data when API is unavailable

## Project Structure

```
nebius/
├── training/                          # Runs on Nebius GPU
│   ├── cosmos_datagen.py              # Cosmos Transfer 2.5 synthetic data generation
│   ├── train.py                       # ResNet-50 fine-tuning (PlantVillage → 3-class)
│   ├── setup_nebius.sh                # One-shot VM setup (deps + dataset + train)
│   └── Dockerfile                     # Nebius Serverless Job container
│
├── inference/                         # Runs locally or on Nebius Endpoint
│   ├── main.py                        # FastAPI server (REST endpoints)
│   ├── phenotype_detector.py          # ONNX model inference
│   ├── disease_diagnostics.py         # Pathology detection (shared model)
│   ├── evaluator.py                   # Trait preference filtering
│   └── score_mapper.py                # 3-class → 6-category dashboard format
│
├── permit_integration/                # Bridge to Learner's Permit
│   ├── phenotype_skill.py             # Wraps model as a licensed skill
│   └── evaluation_sweep.py            # Runs all 4 beats (EARN/REFUSE/BLOCK/REVOKE)
│
├── dashboard/                         # Frontend
│   └── dashboard.html                 # Tailwind + Alpine.js + Chart.js
│
├── requirements.txt
└── README.md
```

## Setup

### Option A: Train on Nebius GPU VM (recommended)

**1. Provision a Nebius VM** with GPU (RTX PRO 6000 or similar), Ubuntu 24.04 + CUDA.

**2. Run the setup script** — downloads PlantVillage dataset, trains the model, exports ONNX:

```bash
scp training/setup_nebius.sh user@<nebius-vm-ip>:~
scp training/train.py user@<nebius-vm-ip>:~
ssh user@<nebius-vm-ip> bash setup_nebius.sh
```

Training takes ~30 minutes on a single GPU.

**3. Copy the trained model back:**

```bash
mkdir -p inference/models
scp user@<nebius-vm-ip>:~/phenotype_training/models/phenotype_model.onnx inference/models/
scp user@<nebius-vm-ip>:~/phenotype_training/models/training_meta.json inference/models/
scp user@<nebius-vm-ip>:~/phenotype_training/models/class_names.json inference/models/
```

**4. (Optional) Generate Cosmos synthetic data** — uncomment step 6 in `setup_nebius.sh` or run manually:

```bash
python cosmos_datagen.py \
    --seed_dir data/plantvillage_raw \
    --output_dir data/cosmos_synthetic \
    --variants_per_image 10
```

### Option B: Nebius Serverless Job

```bash
docker build -t phenotype-training:latest -f training/Dockerfile .
# Push to Nebius Container Registry and submit as Serverless Job
```

### Run the inference server

```bash
pip install -r requirements.txt
cd inference
uvicorn main:app --reload --port 8000
```

API docs at `http://localhost:8000/docs`.

### Open the dashboard

Open `dashboard/dashboard.html` in a browser. It calls the FastAPI backend at `localhost:8000` and falls back to mock data if the API is unavailable.

### Run the Learner's Permit licensing sweep

```bash
python permit_integration/evaluation_sweep.py \
    --test_dir data/test \
    --scope_cultivar tomato_truss \
    --scope_light noon_diffuse
```

Output shows all 4 beats (EARN, REFUSE, BLOCK, REVOKE) with Bayesian gate results.

## API Endpoints

### `POST /api/v1/evaluate/phenotype`

Run phenotypic evaluation with optional trait filtering.

```json
{
  "image_url": "https://example.com/crop-image.jpg",
  "metadata": {
    "target_id": "FIELD-NORTH-04",
    "gps_coordinates": [-121.4944, 38.5815],
    "ndvi_score": 0.62,
    "temperature_delta_c": 2.4
  },
  "user_preferences": {
    "target_classification": "Genetically Desirable",
    "priority_level": "High",
    "target_thresholds": {
      "1_plant_architecture_and_morphology": {
        "erect_leaf_angle_score": 0.75,
        "stalk_strength_and_diameter_score": 0.80
      }
    }
  }
}
```

### `POST /api/v1/evaluate/diagnostics`

Run plant pathology diagnostics.

```json
{
  "image_url": "https://example.com/leaf-sample.jpg",
  "metadata": {
    "target_id": "FIELD-SOUTH-02",
    "ambient_humidity_percent": 84.5,
    "soil_moisture_volumetric": 0.22,
    "temperature_celsius": 29.1
  }
}
```

### `GET /health`

Returns service health and model load status.

## Training Data Pipeline

```
PlantVillage (Kaggle)          Cosmos Transfer 2.5
┌─────────────────────┐       ┌──────────────────────────────┐
│ Tomato_healthy      │──────▶│ Lighting: dawn/noon/dusk/LED  │
│ Tomato_Early_blight │       │ Weather: clear/rain/fog       │
│ Tomato_Late_blight  │       │ Soil: dry/moist/mulched       │
│ Tomato_Leaf_Mold    │       │                               │
│ Tomato_Bacterial_   │       │ 50 seed → 5,000+ synthetic    │
│   spot (+ 5 more)   │       │ Labels carry over from seed   │
└─────────┬───────────┘       └───────────────┬──────────────┘
          │                                   │
          ▼                                   ▼
    ┌─────────────────────────────────────────────┐
    │  3-class mapping:                           │
    │    Tomato_healthy           → Healthy        │
    │    Early_blight, Leaf_Mold  → Stressed       │
    │    Bacterial_spot, Late_    → Diseased        │
    │      blight, Septoria, etc                   │
    └─────────────────────────────────────────────┘
```

## Related Projects

- **[Learner's Permit](https://github.com/thiagopari/learners-permit)** — Bayesian skill licensing framework for robot autonomy. CropAI's phenotype model integrates as a licensed skill.
- **[NVIDIA Cosmos Cookbook — Ag Scenarios](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/transfer2_5/agtec_scenarios_single_view/post_training.html)** — Recipe for post-training Cosmos Transfer on agricultural fleet video.
- **[NVIDIA Aigen Case Study](https://www.nvidia.com/en-us/case-studies/aigen/)** — Production example of 1% real + 99% Cosmos-generated synthetic training data for crop perception.

## Tech Stack

- **Synthetic Data**: NVIDIA Cosmos Transfer 2.5
- **Model**: ResNet-50 fine-tuned → ONNX → TensorRT
- **Training**: PyTorch on Nebius GPU (Serverless Jobs / VM)
- **Inference**: ONNX Runtime (CPU/GPU), FastAPI, Pydantic
- **Licensing**: Learner's Permit Bayesian gate + Cedar RBAC
- **Frontend**: Tailwind CSS, Alpine.js, Chart.js, Font Awesome
- **Infrastructure**: Nebius AI Cloud (RTX PRO 6000), Nebius Serverless Endpoints
