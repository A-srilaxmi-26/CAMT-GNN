# CAMT-GNN

Climate-Aware Multi-Modal Temporal Graph Neural Network for Marine Food Web Analysis and Ecosystem Monitoring.

## Overview
CAMT-GNN uses Graph Neural Networks to analyze marine food-web interactions and incorporate historical climate information.

## Technologies
- Python
- PyTorch
- PyTorch Geometric
- NetworkX
- Pandas and NumPy
- Scikit-learn
- Streamlit
- NOAA OISST
- Mangal database

## Project Structure
- src/data - Data collection and climate processing
- src/preprocessing - Dataset preparation
- src/graph - Graph construction
- src/models - GNN models and experiments
- data - Raw and processed datasets
- results - Experiment outputs
- dashboard - Dashboard interface

## Data Sources
- Mangal marine ecological networks
- NOAA Optimum Interpolation Sea Surface Temperature (OISST)

## Run Dashboard
Install dependencies:

pip install -r requirements.txt

Launch:

python -m streamlit run dashboard.py

## Research Status
Research and development in progress. Climate observations and marine interaction records are historical datasets, not live measurements.
