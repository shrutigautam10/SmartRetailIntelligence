FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install pandas scikit-learn joblib streamlit matplotlib numpy

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]