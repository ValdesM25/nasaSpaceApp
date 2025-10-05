def predict_exoplanet(df, model_name):
    # Generate additional features
    df = create_efficient_features(df)

    # Scale data
    X_scaled = scaler.transform(df)


    # Verify that model exists
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(models.keys())}")

    # Obtain model
    model = models[model_name]

    # Predict
    prob = model.predict_proba(X_scaled)[0][1]
    label = int(prob > 0.5)

    # Results
    result = {
        "model": model_name,
        "probability_confirmed": float(prob),
        "prediction": "CONFIRMED" if label == 1 else "FALSE POSITIVE"
    }

    return result