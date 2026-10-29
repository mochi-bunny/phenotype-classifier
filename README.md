# CropAI — Agricultural Intelligence Platform

Precision agriculture platform for crop phenotyping and plant disease diagnostics, built for the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) (Physical AI Track).

Two inference paths: a trained **ResNet-50** vision model (ONNX/TensorRT) for real-time edge classification, and a **Qwen2.5-VL-72B** VLM on Nebius AI Studio for rich multi-category evaluation with few-shot visual prompting. Uses **NVIDIA Cosmos Transfer 2.5** for synthetic training data generation. Integrates with [Learner's Permit](https://github.com/thiagopari/learners-permit) for Bayesian skill licensing — the model is only trusted within the conditions it was tested under.

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
| **Qwen2.5-VL-72B-Instruct** | VLM on Nebius AI Studio — scores all 6 judging categories directly from the image |
| **Nebius Serverless Jobs** | Model training at scale |
| **Nebius Serverless Endpoints** | Real-time inference serving |
| **Nebius AI Studio** | VLM inference via OpenAI-compatible API |
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

## Inference Approaches

CropAI supports two inference paths, each suited to different deployment scenarios. Both produce the same 6-category judging output consumed by the dashboard and Learner's Permit.

### Approach A: Trained Model (ResNet-50 → ONNX)

**Best for:** Edge deployment, real-time field robotics, deterministic licensing

The trained model pipeline classifies crop images into 3 phenotype classes (Healthy / Stressed / Diseased), then the score mapper expands this into all 6 judging categories with bounded noise to simulate sub-trait variance.

```
Image → ResNet-50 (ONNX) → 3-class softmax → score_mapper → 6 categories → dashboard
                                                                          → Learner's Permit
```

**Strengths:**
- ~5ms inference on GPU, ~50ms on CPU — real-time capable
- Deterministic: same image always produces the same scores
- Edge-deployable via TensorRT on Jetson
- Model hash tracked by Learner's Permit for version-specific licensing
- Works offline — no API calls needed

**Limitation:** The 6-category sub-scores are *derived* from the 3-class prediction, not independently measured. The score mapper adds bounded noise to simulate variance, but a Healthy classification always produces high scores across all categories. The model cannot distinguish, for example, a plant with excellent architecture but poor yield traits.

### Approach B: VLM Direct Evaluation (Qwen2.5-VL-72B)

**Best for:** Rich evaluation, auto-labeling, auditing, research

The VLM evaluates each of the 6 judging categories *independently* from visual cues in the image. It can distinguish leaf curl (abiotic stress) from leaf spots (biotic stress), assess plant architecture, and evaluate yield traits — all from a single image.

```
Image + Prompt → Qwen2.5-VL-72B (Nebius AI Studio) → 6 independent category scores → dashboard
         ↑                                                                           → Learner's Permit
    (optional)
  Reference images
  (few-shot anchors)
```

**Strengths:**
- Scores each judging category independently — can identify a plant with strong architecture but poor biotic resistance
- Returns natural-language observations and assessment limitations
- Supports few-shot visual prompting with reference images for calibration
- No training required — works out of the box with a good prompt

**Limitation:** ~3-5s latency per image (API call), not suitable for real-time edge robotics. Scores can vary slightly between calls. Requires network connectivity to Nebius AI Studio.

#### VLM Phenotype Evaluation Prompt

The VLM receives a structured prompt that covers every sub-trait across all 6 official judging categories. Each trait includes a description of what to look for visually:

**Category 1 — Plant Architecture & Morphology:**
- `erect_leaf_angle_score` — Are leaves angled upward (efficient light capture) or drooping?
- `stalk_strength_and_diameter_score` — Is the stem thick, rigid, and upright?
- `root_system_architecture_score` — If roots are visible, are they well-branched and deep? (null if not visible)
- `branching_habit` — "Erect Canopy" / "Semi-Erect" / "Spreading" / "Irregular / Compromised"

**Category 2 — Yield & Reproductive Traits:**
- `harvest_index_estimate` — Proportion of harvestable fruit/grain vs vegetative biomass
- `kernel_fruit_density_count` — Number of visible fruits, flowers, or grain heads
- `fertility_efficiency_score` — Are flowers/fruits developing uniformly?

**Category 3 — Biotic Stress Resistance:**
- `necrotic_chlorotic_threshold_score` — How much brown/dead or yellow tissue is visible?
- `pest_deterrence_score` — Visible insect damage (holes, mines, webbing)?
- `hypersensitive_response_detected` — Small well-defined lesions suggesting active immune response?

**Category 4 — Abiotic Stress Resilience:**
- `stomatal_regulation_leaf_rolling_score` — Leaf rolling, curling, or wilting from water stress?
- `stay_green_trait_active` — Is the plant maintaining green color despite stress conditions?
- `salinity_aluminum_tolerance_score` — Marginal leaf burns or tip necrosis?

**Category 5 — Quality, Phenology & Post-Harvest:**
- `synchronous_flowering_ripening_score` — Are fruits at a uniform developmental stage?
- `photoperiod_insensitivity_score` — Appropriate flowering for growth stage?
- `pericarp_skin_toughness_score` — Are fruits firm with intact skin, or cracked/split?

**Category 6 — Harvest & Processing Suitability:**
- `non_shattering_seed_pods_score` — Are seed pods/fruits retained on the plant?
- `detachment_force_score` — Do fruits appear firmly attached?

Each trait is scored 0.0–1.0 (0.0–0.3 = Poor, 0.3–0.6 = Below average, 0.6–0.8 = Good, 0.8–1.0 = Excellent). The VLM also returns an overall classification ("Genetically Desirable" / "Standard" / "Suboptimal"), collector queue priority, and a recommended breeder action.

The full prompt and JSON schema are in [`inference/phenotype_vlm.py`](inference/phenotype_vlm.py).

#### Few-Shot Visual Prompting with Reference Images

Providing verified reference images alongside the query image significantly improves VLM consistency:

```
Message 1 (references):
  "Here are verified reference specimens at known quality levels."
  [IMAGE: verified healthy plant]     → "REFERENCE — HEALTHY: vibrant green, erect architecture, no lesions"
  [IMAGE: verified stressed plant]    → "REFERENCE — STRESSED: mild leaf curl, slight yellowing"
  [IMAGE: verified diseased plant]    → "REFERENCE — DISEASED: necrotic lesions, chlorotic halos"

Message 2 (query):
  "Now evaluate this new image across all 6 judging categories."
  [IMAGE: unknown crop to classify]
```

**Why reference images help:**
- **Calibration** — The VLM sees what "healthy" looks like for *this specific cultivar*, not just from pretraining
- **Consistency** — Scores become anchored to the reference baseline, reducing inter-call variance
- **Domain adaptation** — A tomato reference set calibrates differently than corn or wheat
- **Confidence** — The VLM can state "this specimen shows more necrosis than the healthy reference but less than the diseased reference"

**How to select reference images:**
1. Pick 1 image per class (Healthy, Stressed, Diseased) from your seed dataset
2. Choose clear, well-lit, representative examples — avoid edge cases
3. Match the cultivar to your target crop (tomato references for tomato evaluation)
4. Store them in a `references/` directory and pass their URLs to the VLM

```bash
# Zero-shot (no references)
python inference/phenotype_vlm.py https://example.com/crop.jpg

# Few-shot with reference images (recommended)
python inference/phenotype_vlm.py https://example.com/crop.jpg \
    --ref_healthy references/tomato_healthy.jpg \
    --ref_stressed references/tomato_stressed.jpg \
    --ref_diseased references/tomato_diseased.jpg
```

#### Comparison: Trained Model vs VLM

| Dimension | Trained Model (ResNet-50) | VLM (Qwen2.5-VL-72B) |
|-----------|--------------------------|----------------------|
| **Latency** | ~5ms GPU / ~50ms CPU | ~3-5s (API) |
| **Scores** | 3-class → derived 6-cat | 6 independent categories |
| **Determinism** | Identical on repeat | Slight variance |
| **Edge deploy** | Yes (ONNX/TensorRT) | No (needs API) |
| **Training required** | Yes (PlantVillage + Cosmos) | No (prompt only) |
| **New cultivars** | Retrain | Update prompt/references |
| **Learner's Permit** | Model hash tracked | Model ID tracked |
| **Best role** | Primary runtime classifier | Auto-labeler, auditor, research |

**Recommended hybrid approach:** Use the trained model for real-time field inference and the VLM as an auto-labeler for Cosmos-generated synthetic images and as an auditor that spot-checks trained model decisions.

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
│   ├── phenotype_vlm.py               # VLM inference via Nebius AI Studio (Qwen2.5-VL)
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
- **Trained Model**: ResNet-50 fine-tuned → ONNX → TensorRT
- **VLM**: Qwen2.5-VL-72B-Instruct on Nebius AI Studio (few-shot visual prompting)
- **Training**: PyTorch on Nebius GPU (Serverless Jobs / VM)
- **Inference**: ONNX Runtime (CPU/GPU) + Nebius AI Studio API, FastAPI, Pydantic
- **Licensing**: Learner's Permit Bayesian gate + Cedar RBAC
- **Frontend**: Tailwind CSS, Alpine.js, Chart.js, Font Awesome
- **Infrastructure**: Nebius AI Cloud (RTX PRO 6000), Nebius Serverless Endpoints
