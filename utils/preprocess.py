import pandas as pd

def preprocess_input(df, label_encoders, scaler):
    df = df.copy()

    # Encode categorical columns
    for col in label_encoders:
        le = label_encoders[col]
        df[col] = le.transform(df[col])

    # Drop 'Credit amount' from input as it's used for target creation
    df = df.drop(columns=["Credit amount"], errors="ignore")

    # Scale numeric features
    scaled_input = scaler.transform(df)

    return scaled_input
