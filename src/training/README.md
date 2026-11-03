# CAMT-GNN

**Climate-Aware Multi-Modal Temporal Graph Neural Network for Marine Food Web Analysis and Ecosystem Monitoring**

## Overview
CAMT-GNN is a research project that uses Graph Neural Networks to analyze marine food-web interactions and incorporate environmental climate information.

## Technologies
- Python
- PyTorch and PyTorch Geometric
- NetworkX
- Pandas and NumPy
- Scikit-learn
- NOAA OISST
- Mangal database
- Streamlit

## Project Structure
- `src/data/` – Data collection and climate processing
- `src/preprocessing/` – Dataset preparation
- `src/graph/` – Graph construction
- `src/models/` – GNN models and experiments
- `data/` – Marine datasets and processed features
- `results/` – Experiment outputs
- `dashboard/` – Dashboard application
- `dashboard.py` – Dashboard entry point
- `requirements.txt` – Python dependencies

## Data Sources
- Mangal database: marine ecological networks and interactions
- NOAA OISST: historical sea-surface temperature data

## Run the Dashboard
```bash
pip install -r requirements.txt
python -m streamlit run dashboard.py
```

## Research Notes
The project uses historical marine food-web records and historical climate observations. Predictions and climate-related findings should be interpreted within the limitations of the available datasets and experiments.

## Status
Research and development in progress.