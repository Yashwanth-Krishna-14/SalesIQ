# SalesIQ — AI-Powered Sales Lead Scoring & Forecasting Platform

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**End-to-End Machine Learning System for Predictive Lead Scoring, Customer Segmentation, and AI-Driven Sales Insights.**

Built with a **production-grade ML pipeline**, experiment tracking, LLM-powered analytics, and a polished multi-page Streamlit dashboard.

---

## 🎯 Project Highlights

- **End-to-End ML Pipeline**: From raw CRM data to deployed predictions
- **Advanced Lead Scoring**: Random Forest + XGBoost models with full evaluation
- **Customer Persona Segmentation**: KMeans clustering with business-friendly personas (Champion, Prospect, At-Risk)
- **MLOps Implementation**: Full experiment tracking and artifact logging with **MLflow**
- **Generative AI Layer**: LangChain + OpenAI for actionable, persona-specific sales insights
- **Production Ready**: Dockerized multi-container setup with Streamlit + MLflow UI
- **Clean Architecture**: Modular, testable, and well-documented code

---

## ✨ Features

- Realistic CRM dataset generation (1000 leads)
- Automated data preprocessing and feature engineering
- Model training, hyperparameter logging, and comparison
- Real-time lead scoring dashboard with filtering
- Interactive customer segmentation with persona insights
- LLM-generated strategic recommendations per segment
- Model performance visualization and experiment tracking

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[Raw CRM Data<br/>leads.csv] --> B[Data Loader & Validator]
    B --> C[Preprocessing & Feature Engineering]
    C --> D[Supervised Models<br/>RF + XGBoost]
    C --> E[Unsupervised Clustering<br/>KMeans]
    D --> F[MLflow Tracking & Artifacts]
    E --> F
    F --> G[Streamlit Multi-Page Dashboard]
    G --> H[LangChain + GPT-3.5<br/>AI Insights Engine]
