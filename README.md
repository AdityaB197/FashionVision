# FashionVision

A CNN-based Fashion-MNIST clothing classifier built with TensorFlow/Keras and Streamlit.

## Project Structure

```
FashionVision/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── src/
│   ├── data.py             # Dataset loading and preprocessing
│   ├── model.py            # CNN model definition
│   ├── inference.py        # Prediction helpers
│   └── reporting.py        # Metrics and visualisation helpers
├── scripts/
│   ├── train_model.py      # Training entry-point
│   ├── evaluate_model.py   # Evaluation entry-point
│   └── verify_project.py   # Sanity-check script
├── notebooks/              # Exploratory notebooks
├── models/                 # Saved model checkpoints
├── results/                # Evaluation outputs (plots, CSVs)
├── tests/                  # Pytest test suite
└── docs/                   # Architecture notes and report outline
```

## Quick Start

```bash
pip install -r requirements.txt
python -m src.data          # verify dataset loading
```
