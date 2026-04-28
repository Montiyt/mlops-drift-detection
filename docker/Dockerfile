FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --timeout=300 \
    pandas==2.3.3 \
    numpy==2.2.6 \
    scikit-learn==1.7.2 \
    mlflow==3.11.1 \
    prometheus-client==0.25.0 \
    fastapi==0.136.1 \
    uvicorn==0.46.0 \
    pyarrow==23.0.1 \
    scipy==1.15.3 \
    matplotlib==3.10.9 \
    seaborn==0.13.2 \
    shap==0.49.1 \
    pydantic==2.13.3 \
    starlette==1.0.0 \
    python-dotenv==1.2.2

COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY mlflow/ ./mlflow/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]