## Text Commands Classification & Voice → ROS pipeline

This package provides a complete text classification and voice-driven command system. It includes:

- from-scratch Transformer training (`train.py`, `search.py`) and grid search support
- fine-tuning of pretrained models and an ensemble workflow (`train_ensemble.py`, `trainothermodels.py`, `predict_ensemble.py`)
- SentencePiece (SPM) tooling and fine-tuned SPM support (`spm.py`, `big_spm.py`)
- evaluation/test utilities (`evaluate_models.py`, `test_*.py`, `plot_accuracy_vs_config.py`)
- voice recording + Whisper integration (`whisper_mic.py`, `predict_voice_ensemble.py`, `predict_voice_ros.py`) that convert speech → text → intent
- ROS publishing helpers for real-time robot control (optional; requires ROS Noetic / `rospy`) so your intent can be sent to `unitree_ros` or other ROS consumers

Key files and what they do
- `dataset.py`       — Data loading, tokenization helpers, `TextCommandsDataset` and `Vocab` classes (supports word vocab and SPM)
- `model.py`         — Transformer encoder classifier used for from-scratch training
- `train.py`         — Train a transformer from scratch (word-vocab or SPM), with augmentation
- `search.py`        — Hyperparameter sampling / grid search over transformer hyperparameters
- `train_ensemble.py`— Fine-tune multiple pretrained transformer-based models (Hugging Face) for an ensemble
- `trainothermodels.py` / `othermodels.py` — Alternative model families (CNN, BiLSTM, etc.) and training scripts
- `predict.py`       — Simple text-only predictor (expects vocab in checkpoint)
- `spmpredict.py`    — Predictor that can fall back to SentencePiece when checkpoint lacks `vocab`
- `predict_ensemble.py` / `predict_voice_ensemble.py` — Ensemble prediction and an interactive voice -> ensemble loop
- `predict_voice_ros.py` — Interactive voice -> transcription -> SPM prediction with optional ROS publishing (used for robot control)
- `whisper_mic.py`   — Record audio and transcribe with OpenAI Whisper
- `spm.py`, `big_spm.py` — Build/fine-tune SentencePiece models for your dataset

Which CSV is used by default
- Many training/fine-tuning scripts in this repo default to `data/updated_data2.csv`. Use `--csv` to override if needed.

Dependencies (what to install)

The existing `requirements.txt` in this folder lists core Python libs, but the project uses several more packages (Whisper, audio I/O, Hugging Face Transformers, plotting, ROS runtime). Below is a recommended set to get the full functionality.

Quick install (recommended minimal + extras):

```bash
# create venv and activate
python -m venv .venv
source .venv/bin/activate

# core python deps (replace with pinned versions you prefer)
pip install --upgrade pip
pip install torch pandas numpy tqdm sentencepiece transformers scikit-learn matplotlib seaborn

# audio + Whisper + convenience
pip install -U openai-whisper sounddevice soundfile keyboard

# optional extras used by some scripts
pip install sentencepiece tqdm

# If you use HuggingFace fine-tuning (train_ensemble.py), also install:
pip install transformers accelerate datasets
```

Notes about ROS and platform requirements
- The voice→ROS runtime (`predict_voice_ros.py`) optionally publishes predictions to a ROS topic using `rospy`. That requires a system with ROS Noetic (Ubuntu 20.04 recommended) and Python packages provided by the ROS installation. Install ROS Noetic following official instructions: https://wiki.ros.org/noetic/Installation/Ubuntu
- After installing ROS, ensure your environment sources the setup script in every shell that will run ROS-enabled scripts:

```bash
source /opt/ros/noetic/setup.bash
# if you use a catkin workspace, source that too:
source ~/catkin_ws/devel/setup.bash
```

- `rospy` is not installed via pip; it comes from the ROS apt packages. If you run `predict_voice_ros.py` with `--ros-publish` enabled, the script will attempt to import `rospy` and gracefully continue without ROS if it is not available.
- `unitree_ros` (for sending commands to the Unitree Go1 robot) is a separate ROS package and must be built in your catkin workspace (outside this repo). Ensure `unitree_ros` and its messages are installed/built and your ROS_MASTER_URI is correctly set when running the publisher.

Files that change/are written by training
- `models/.../best.pth` — saved checkpoint (contains `model_state`, `label_map`, and usually `vocab` or `spm_model`)
- `models/.../config.txt` — text copy of arguments used to train (useful to reconstruct model at inference)
- `models/.../label_map.txt` — text mapping of label -> id
- `models/.../metrics.json` — search/grid-run metrics (when using `search.py` / ensemble runs)

How to train (examples)

From-scratch transformer (word vocab built from CSV; default CSV often used in repo is `data/updated_data2.csv`):

```bash
python text_classification/train.py --csv data/updated_data2.csv \
  --text-col text --label-col label \
  --out-dir models/transformer_from_scratch --epochs 8 --batch-size 64
```

Train with a SentencePiece tokenizer (pass `--spm-model`):

```bash
python text_classification/train.py --csv data/updated_data2.csv \
  --spm-model text_classification/spm_en_ar_joint_finetuned.model \
  --out-dir models/transformer_spm --epochs 8
```

Hyperparameter search / grid sampling
- Use `search.py` to sample combinations from a hyperparameter grid and run multiple short experiments; useful to find reasonable hyperparameters before final training. `search.py` contains a default grid and samples up to 30 combinations (can be adjusted).

Fine-tuning pretrained models (ensemble)
- Use `train_ensemble.py` to fine-tune Hugging Face pretrained models (BERT, DistilBERT, RoBERTa, DeBERTa, etc.). The script supports freezing encoder layers, warmup schedules, mixed precision, and saving tokenizers. Example:

```bash
python text_classification/train_ensemble.py --csv data/updated_data2.csv --out-dir models/ensemble --models bert distilbert roberta --epochs 3
```

Testing & evaluation
- Use `evaluate_models.py` and `test_*.py` scripts to compute confusion matrices, accuracy, and other diagnostics. Example:

```bash
python text_classification/evaluate_models.py --model-dir models/run_spm --csv data/test_set.csv
```

Voice → intent → ROS runtime (Whisper + predictor)

1. Record & transcribe with Whisper locally (quick test):

```bash
python -m text_classification.whisper_mic --model small --duration 5 --language en
```

2. Interactive voice → ensemble (Whisper + ensemble predictor, prints voting breakdown):

```bash
python -m text_classification.predict_voice_ensemble --custom-model models/run1/best.pth --ensemble-dir models/ensemble --whisper-model small
```

3. Interactive voice → SPM predictor → optional ROS publish (robot control):

```bash
python -m text_classification.predict_voice_ros --ros-publish --ros-topic command
```

Notes for production/robot integration
- To publish to ROS you must run on a system with ROS Noetic (Ubuntu 20.04 is the tested target). The script will try to import `rospy` and notify you if ROS isn't available. Building `unitree_ros` and configuring ROS networking is out of this repo's scope — do so in your ROS workspace and source the workspace before running the publisher scripts.
- For low-latency deployments consider running a small Whisper model (tiny/base) on a CPU or small GPU; for best accuracy use `small`/`medium` on a GPU.



