FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install pandas scikit-learn joblib

CMD ["python", "inference.py"]