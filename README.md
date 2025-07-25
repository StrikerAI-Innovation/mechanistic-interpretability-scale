# 🚀 Scaling Mechanistic Interpretability to Large Language Models (In Dev)

[![Tests](https://github.com/yourusername/mechanistic-interpretability-scale/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/mechanistic-interpretability-scale/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-IEEE-green.svg)](docs/papers/ieee_paper.pdf)

This repository contains the official implementation of our paper:

> **Scaling Mechanistic Interpretability to Production-Scale Language Models**  
> 🧪 IEEE Conference on AI Systems 2025  
> 📄 [Read the paper](docs/papers/ieee_paper.pdf)

---

## ✨ Key Contributions

- **HybridSAE**: A novel sparse autoencoder architecture for interpretable and efficient feature extraction.
- **MEGA-Bench**: A benchmark suite evaluating interpretability across model families and SAE types.
- **Unified Scaling Laws**: Predictive models estimating interpretability cost vs. model scale.
- **Full Stack Tooling**: Modular training pipeline, circuit discovery, attribution tools, and metric reporting.

---

## 🧱 Repository Structure

<pre lang="markdown">

<details>

mechanistic-interpretability-scale/
├── src/                    # Core implementation
│   ├── models/             # SAE architectures
│   ├── training/           # Training loop and optimizers
│   ├── analysis/           # Feature analysis and circuit tracing
│   ├── benchmarks/         # Benchmark interfaces and metrics
│   └── utils/              # Helper utilities
├── experiments/            # Reproducible experiments
├── notebooks/              # Jupyter notebooks for analysis and figures
├── tests/                  # Unit tests
├── scripts/                # Automation scripts
├── docs/                   # Documentation and paper
├── configs/                # Config files
├── data/                   # Gitignored cache, weights, results
├── requirements.txt
├── setup.py
└── README.md


</details>

</pre>



---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mechanistic-interpretability-scale
cd mechanistic-interpretability-scale

# Create environment
conda create -n mech-interp python=3.10 -y
conda activate mech-interp

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Download pretrained models and activations (optional)
bash scripts/download_models.sh

## ⚡ Quick Start

```python
from src.models import HybridSAE
from src.training import train_sae
from src.analysis import analyze_features
from src.utils.model_loading import load_model

# Load a pre-trained transformer
model = load_model("gpt2-small")

# Initialize Hybrid Sparse Autoencoder
sae = HybridSAE(
    d_model=768,
    n_features=32768,
    k_sparse=128
)

# Train on transformer activations
trained_sae = train_sae(sae, model, dataset)

# Analyze discovered features
features = analyze_features(trained_sae, model)
```

---

## 🧪 Experiments

You can reproduce all experiments via:

```bash
bash scripts/run_all_experiments.sh
```

Or run them individually:

```bash
python experiments/01_sae_comparison/run_comparison.py
python experiments/02_scaling_laws/derive_scaling_laws.py
python experiments/03_hybrid_architecture/train_hybrid_sae.py
```

---

## 📊 Results Snapshot

| Method               | Recon. Error | Active Features | Train Time | Memory   |
| -------------------- | ------------ | --------------- | ---------- | -------- |
| K-Sparse SAE         | ---          | ---             | ---        | ---      |
| Gated SAE            | ---          | ---             | ---        | ---      |
| **HybridSAE (Ours)** | ---          | ---             | ---        | ---      |

More detailed analysis and figures will be available in:

* 📓 `notebooks/03_results_analysis.ipynb`
* 📓 `notebooks/04_paper_figures.ipynb`

---

## 📚 Documentation

* [Installation Guide](docs/installation.md)
* [Quick Start](docs/quick_start.md)
* [Experiments Guide](docs/experiments_guide.md)
* [API Reference](docs/api_reference.md)

---

## 🤝 Contributing

We welcome contributions! To get started:

1. Fork the repo and create a new branch.
2. Run the test suite: `pytest tests/`
3. Submit a pull request with a clear description of your changes.

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## 📝 Citation

If you use this codebase in your work, please cite:

```bibtex
@inproceedings{strikerinnovations2025scaling,
  title=Scaling Mechanistic Interpretability to Production-Scale Language Models,
  author=Striker AI Innovations,
  booktitle=IEEE Conference on AI Systems,
  year=2025
}
```

---

## 🙏 Acknowledgments

* Anthropic (SAE research on Claude 3 Sonnet)
* OpenAI (TransformerLens and interpretability tools)
* DeepMind (Gemma and mechanistic insights)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
