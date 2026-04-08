import joblib
import pandas as pd

def load_model():
    model = joblib.load("sales_model.pkl")
    return model

def predict(category="Furniture", sub_category="Chairs", region="South", ship_mode="Standard Class"):
    """Predict sales using the trained model with proper one-hot encoded features."""
    model = load_model()

    # Build input with all features set to 0
    input_data = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)

    # Set the selected features to 1
    feature_map = {
        f"Category_{category}": 1,
        f"Sub-Category_{sub_category}": 1,
        f"Region_{region}": 1,
        f"Ship Mode_{ship_mode}": 1,
    }

    for col, val in feature_map.items():
        if col in input_data.columns:
            input_data[col] = val
        else:
            print(f"Warning: Feature '{col}' not found in model. Skipping.")

    result = model.predict(input_data)
    print(f"Predicted Sales: ${result[0]:,.2f}")
    return result[0]

if __name__ == "__main__":
    # Example: Furniture Chair in South region with Standard Class shipping
    predict("Furniture", "Chairs", "South", "Standard Class")
