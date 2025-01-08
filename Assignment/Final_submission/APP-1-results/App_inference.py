import pandas as pd
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Load the trained model, selector, and label encoder
model = joblib.load("bagging_model.pkl")
selector = joblib.load("selector.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define the same feature preprocessing
def preprocess_features(df):
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    for col in non_numeric_cols:
        if col != 'label':  # skip 'label' if it appears
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

# Load CSV data
new_data_df = pd.read_csv("val_without_labels.csv")

# Preprocess features
new_data_df = preprocess_features(new_data_df)

# Transform with the same KBest selector
X_new_kbest = selector.transform(new_data_df)

# Model prediction (numeric labels)
predictions_numeric = model.predict(X_new_kbest)

# Inverse transform numeric predictions to original string labels
predictions_original = label_encoder.inverse_transform(predictions_numeric)

# Save predictions to CSV
output_df = pd.DataFrame({"predictions": predictions_original})
output_df.to_csv("App_predictions.csv", index=False)

print("Predictions saved to predictions.csv")
