# Water Leak Forecasting Pipeline

**Author**: Nirvana Fanaelahi  
**Date**: 2025  
**Purpose**: Real-time **Minimum Night Flow (MNF)** forecasting using **LSTM + Attention** to detect leaks in water distribution networks.

## Features
- End-to-end pipeline: ingestion → validation → features → training → API
- MLflow tracking + DVC data versioning
- Deployed via FastAPI + Docker + AWS/GCP
- 30% water loss reduction potential via anomaly detection

## Setup
```bash
pip install -r requirements.txt
docker-compose up --build