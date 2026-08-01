# CAP – Contextualized Hate Speech Detection with Explainability

This repository contains the code to reproduce the results of the **CAP** model for hate speech detection, explainability, and bias analysis.

---

## Setup

1. Clone this repository to your local machine.

2. Create and activate a Python virtual environment (Python **3.9+** recommended).

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install the required dependencies.

```bash
pip install -r requirements.txt
```

---

## Reproducing the Results

The evaluation scripts automatically download the trained model checkpoints and the **HateXplain** dataset. No manual configuration is required.

### RoBERTa Backbone (λ = 2)

```bash
python "evaluate(Roberta lambda 2).py"
```

### HateBERT Backbone (λ = 1)

```bash
python "evaluate(HateBert lambda 1).py"
```

Both scripts report:

- Classification metrics
- Explainability metrics
- Bias metrics

---

## Additional Files

| File | Description |
|------|-------------|
| `Cap(HateBERT).ipynb` | Complete Jupyter notebook containing the HateBERT training and evaluation pipeline. |
| `train_script(hateBERT).py` | Training script for the HateBERT backbone. |
| `train_script(RoBERTa).py` | Training script for the RoBERTa backbone. |

---

## Training from Scratch

To train a model from scratch, simply execute the corresponding training script.

Example:

```bash
python "train_script(hateBERT).py"
```

or

```bash
python "train_script(RoBERTa).py"
```

The required datasets will be downloaded automatically during execution.