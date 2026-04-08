import pickle

def load_model():
    with open("sales_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def predict():
    model = load_model()
    # dummy input
    result = model.predict([[1, 2, 3]])
    print(result)

if __name__ == "__main__":
    predict()