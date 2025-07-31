{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SAE Comparison Experiments - Complete Workflow\n",
    "\n",
    "This notebook provides a complete workflow for running SAE experiments from your GitHub repository.\n",
    "\n",
    "**Steps:**\n",
    "1. Setup environment\n",
    "2. Clone repository\n",
    "3. Install dependencies\n",
    "4. Run experiments\n",
    "5. Visualize results\n",
    "6. Save to Google Drive"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup Environment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check GPU availability\n",
    "import torch\n",
    "import os\n",
    "\n",
    "if torch.cuda.is_available():\n",
    "    gpu_info = torch.cuda.get_device_name(0)\n",
    "    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
    "    print(f\"✓ GPU Available: {gpu_info}\")\n",
    "    print(f\"  Memory: {gpu_memory:.1f} GB\")\n",
    "else:\n",
    "    print(\"❌ No GPU detected!\")\n",
    "    print(\"Go to Runtime → Change runtime type → GPU\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Mount Google Drive (optional - for saving results)\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "# Create results directory in Drive\n",
    "!mkdir -p /content/drive/MyDrive/sae_experiments"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Clone Repository"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set your GitHub username\n",
    "GITHUB_USERNAME = \"YOUR_USERNAME\"  # <-- Change this!\n",
    "REPO_NAME = \"mechanistic-interpretability-scale\"\n",
    "\n",
    "# Clone repository\n",
    "!git clone https://github.com/{GITHUB_USERNAME}/{REPO_NAME}.git\n",
    "%cd {REPO_NAME}\n",
    "\n",
    "# Verify we're in the right directory\n",
    "!pwd\n",
    "!ls -la"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Install Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install from requirements.txt\n",
    "!pip install -r requirements.txt -q\n",
    "\n",
    "# Install package in development mode\n",
    "!pip install -e . -q\n",
    "\n",
    "print(\"✓ Dependencies installed!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Verify imports\n",
    "try:\n",
    "    from src.models import KSparseSAE, GatedSAE, HybridSAE, VanillaSAE\n",
    "    from src.training.trainer import SAETrainer\n",
    "    from src.utils.model_loading import load_model\n",
    "    print(\"✓ All imports successful!\")\n",
    "except Exception as e:\n",
    "    print(f\"❌ Import error: {e}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Configure Experiment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Detect GPU and set appropriate configuration\n",
    "if torch.cuda.is_available():\n",
    "    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
    "    \n",
    "    if gpu_memory > 30:  # A100\n",
    "        config = {\n",
    "            'model': 'gpt2-medium',\n",
    "            'n_features': 16384,\n",
    "            'batch_size': 32,\n",
    "            'epochs': 10\n",
    "        }\n",
    "    elif gpu_memory > 15:  # V100/T4\n",
    "        config = {\n",
    "            'model': 'gpt2',\n",
    "            'n_features': 8192,\n",
    "            'batch_size': 16,\n",
    "            'epochs': 5\n",
    "        }\n",
    "    else:  # Smaller GPUs\n",
    "        config = {\n",
    "            'model': 'gpt2',\n",
    "            'n_features': 4096,\n",
    "            'batch_size': 8,\n",
    "            'epochs': 3\n",
    "        }\n",
    "    \n",
    "    print(f\"Configuration for {gpu_memory:.1f}GB GPU:\")\n",
    "    for key, value in config.items():\n",
    "        print(f\"  {key}: {value}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Run Experiments"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Quick test run (5-10 minutes)\n",
    "!python experiments/01_sae_comparison/run_comparison.py \\\n",
    "    --model gpt2 \\\n",
    "    --layer 6 \\\n",
    "    --n_features 4096 \\\n",
    "    --batch_size 8 \\\n",
    "    --epochs 2 \\\n",
    "    --debug"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Full experiment run (1-3 hours)\n",
    "!python experiments/01_sae_comparison/run_comparison.py \\\n",
    "    --model {config['model']} \\\n",
    "    --layer 6 \\\n",
    "    --n_features {config['n_features']} \\\n",
    "    --batch_size {config['batch_size']} \\\n",
    "    --epochs {config['epochs']}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Visualize Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from pathlib import Path\n",
    "from IPython.display import Image, display\n",
    "\n",
    "# Find results\n",
    "results_dirs = sorted(Path('results').glob('sae_comparison_*'))\n",
    "if results_dirs:\n",
    "    latest_results = results_dirs[-1]\n",
    "    print(f\"Results found in: {latest_results}\")\n",
    "    \n",
    "    # Load and display summary\n",
    "    summary_path = latest_results / 'summary_results.csv'\n",
    "    if summary_path.exists():\n",
    "        df = pd.read_csv(summary_path)\n",
    "        display(df)\n",
    "    \n",
    "    # Display all plots\n",
    "    for img in sorted(latest_results.glob('*.png')):\n",
    "        print(f\"\\n{img.name}:\")\n",
    "        display(Image(str(img)))\n",
    "else:\n",
    "    print(\"No results found yet!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Save Results to Google Drive"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import shutil\n",
    "from datetime import datetime\n",
    "\n",
    "if results_dirs:\n",
    "    # Create timestamped folder\n",
    "    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n",
    "    drive_path = f'/content/drive/MyDrive/sae_experiments/run_{timestamp}'\n",
    "    \n",
    "    # Copy results\n",
    "    shutil.copytree(latest_results, drive_path)\n",
    "    print(f\"✓ Results saved to: {drive_path}\")\n",
    "    \n",
    "    # Also save the notebook\n",
    "    !cp /content/{REPO_NAME}/notebooks/*.ipynb {drive_path}/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Advanced: Run Multiple Configurations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run experiments with different settings\n",
    "configurations = [\n",
    "    {\"model\": \"gpt2\", \"layer\": 6, \"n_features\": 4096},\n",
    "    {\"model\": \"gpt2\", \"layer\": 8, \"n_features\": 8192},\n",
    "    {\"model\": \"gpt2\", \"layer\": 10, \"n_features\": 8192},\n",
    "]\n",
    "\n",
    "for i, conf in enumerate(configurations):\n",
    "    print(f\"\\nRunning configuration {i+1}/{len(configurations)}...\")\n",
    "    !python experiments/01_sae_comparison/run_comparison.py \\\n",
    "        --model {conf['model']} \\\n",
    "        --layer {conf['layer']} \\\n",
    "        --n_features {conf['n_features']} \\\n",
    "        --batch_size 16 \\\n",
    "        --epochs 3"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Clean Up (Optional)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clear GPU memory\n",
    "import gc\n",
    "gc.collect()\n",
    "torch.cuda.empty_cache()\n",
    "\n",
    "# Check memory usage\n",
    "!nvidia-smi"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}