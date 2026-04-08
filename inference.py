import pandas as pd
import joblib

# Load model
model = joblib.load("sales_model.pkl")

def predict(input_data: dict):
    df = pd.DataFrame([input_data])

    # Ensure all columns match model
    model_columns = model.feature_names_in_

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]

    prediction = model.predict(df)[0]

    return {"predicted_sales": float(prediction)}