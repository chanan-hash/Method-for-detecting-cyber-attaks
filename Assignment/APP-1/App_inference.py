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

def select_features(X, y, correlation_matrix, correlation_threshold=0.95):
    # Remove highly correlated features
    highly_correlated = np.where(np.abs(correlation_matrix) > correlation_threshold)
    highly_correlated = [(correlation_matrix.index[x], correlation_matrix.columns[y]) 
                        for x, y in zip(*highly_correlated) if x != y and x < y]
    
    features_to_drop = set()
    for feat1, feat2 in highly_correlated:
        if feat1 not in features_to_drop:
            features_to_drop.add(feat2)
    
    X = X.drop(columns=list(features_to_drop))
    
    # Select best features using mutual information
    # selector = SelectKBest(score_func=mutual_info_classif, k='all')
    selector = SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=42), k='all')
    selector.fit(X, y)
    
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    })
    
    print("\nTop 20 features by mutual information:")
    print(feature_scores.sort_values('Score', ascending=False).head(30))
    
    # Select top features
    k = 30  # Number of features to select
    best_features = feature_scores.nlargest(k, 'Score')['Feature'].tolist()
    
    return X[best_features]


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

# Taking the data into the same shape as the training data


# Transform with the same KBest selector
# X_new_kbest = selector.transform(new_data_df)

# Model prediction (numeric labels)
predictions_numeric = model.predict(X_new_kbest)

# Inverse transform numeric predictions to original string labels
predictions_original = label_encoder.inverse_transform(predictions_numeric)

# Save predictions to CSV
output_df = pd.DataFrame({"predictions": predictions_original})
output_df.to_csv("App_predictions.csv", index=False)

print("Predictions saved to predictions.csv")
