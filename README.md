# CAP – Contextualized Hate Speech Detection with Explainability

This repository contains the code to reproduce the results of the **CAP** model for hate speech detection, explainability, and bias analysis.

---

## Setup

> **Note:** If you prefer to use **Google Colab**, simply create a new notebook and paste the entire evaluation script (`evaluate_Roberta_Lambda2.py` or `evaluate_HateBert_Lambda2.py`) into a code cell.
>
> No additional setup is required. All required Python packages will be installed automatically, the pretrained model will be downloaded from **Hugging Face**, and the **HateXplain** dataset will be retrieved automatically from the original GitHub repository. Once the setup is complete, run the notebook cells in order to reproduce the evaluation results.

if you wanna test on your machine

1. Clone this repository to your local machine.

2. Create and activate a Python virtual environment (Python **3.9+** recommended).

### Linux 

```bash
python3 -m venv venv
source venv/bin/activate
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
python3 evaluate_Roberta_Lambda2.py
```

### HateBERT Backbone (λ = 1)

```bash
python3 evaluate_HateBert_Lambda1.py
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
| `train_script_hateBERT.py` | Training script for the HateBERT backbone. |
| `train_script_RoBERTa.py` | Training script for the RoBERTa backbone. |

---

## Training from Scratch

To train a model from scratch, run the corresponding training script:

```bash
python3 train_script_hateBERT.py
```

or

```bash
python3 train_script_RoBERTa.py
```

The required datasets will be downloaded automatically during execution.

### Using Google Colab

If you prefer to train the models in **Google Colab**, create a new notebook and mount your Google Drive by running:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then, create a new code cell and paste the full training script (`train_script_hateBERT.py` or `train_script_RoBERTa.py`) directly into the notebook. Run the cells in order to start training. The required datasets will be downloaded automatically during execution.


> **Note:** `LAMBDA_RAT` is the weighting coefficient for the rationale supervision objective. It can be set to any desired value to reproduce different experimental configurations.