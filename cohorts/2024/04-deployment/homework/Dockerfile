FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app
COPY ["starter.py", "."]

RUN pip install pandas pyarrow scikit-learn==1.5.0

ENTRYPOINT ["python","./starter.py"]

# docker run --rm mlzoomcamp-model  --year 2023 --month 05
