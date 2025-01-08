import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import joblib  # for saving the model

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

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


# ===========================
# STEP 1: DATA LOADING
# ===========================
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError as e:
    print("Error: ", e)
    exit()

# ===========================
# STEP 2: PREPROCESS FEATURES
# (But do NOT encode the 'label' here)
# ===========================
def preprocess_features(df):
    # Encode only feature columns that are objects, not the target
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    for col in non_numeric_cols:
        if col != 'label':  # Important: skip the target column
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

train_df = preprocess_features(train_df)
test_df = preprocess_features(test_df)

# ===========================
# STEP 2B: ENCODE THE LABEL (TARGET) SEPARATELY
# ===========================
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['label'].astype(str))
test_df['label'] = label_encoder.transform(test_df['label'].astype(str))

# Save the label encoder for inverse transform in predictions
joblib.dump(label_encoder, "label_encoder.pkl")

# Split data into features (X) and target (y)
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# ===========================
# STEP 3: SMOTE
# ===========================
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# selector = SelectKBest(score_func=f_classif, k=k)
# X_train_kbest = selector.fit_transform(X_train_res, y_train_res)
# X_test_kbest = selector.transform(X_test)

# ===========================
# STEP 4: FEATURE SELECTION
# ===========================
# Taking One minute
# print("\nSelecting features...")
# X_train_selected = select_features(X_train_res, y_train_res, X_train_res.corr())
# X_test_selected = X_test[X_train_selected.columns]

# ===========================
# STEP 5: NORMALIZE FEATURES
# ===========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)


# ===========================
# STEP 6: FINAL MODEL
# ===========================
bagging_clf_final = BaggingClassifier(
    n_estimators=800,
    max_samples=1.0,
    max_features=0.5,
    random_state=90, # 42
    n_jobs=-1
)

# bagging_clf_final.fit(X_train_kbest, y_train_res)
# acc_final = accuracy_score(y_test, bagging_clf_final.predict(X_test_kbest))
# print("Final Bagging Accuracy:", acc_final)

bagging_clf_final.fit(X_train_scaled, y_train_res)
acc_final = accuracy_score(y_test, bagging_clf_final.predict(X_test_scaled))
print("Final Bagging Accuracy:", acc_final)

# ===========================
# STEP 7: SAVE THE MODEL AND SELECTOR
# ===========================
model_filename = 'bagging_model.pkl'
joblib.dump(bagging_clf_final, model_filename)
print(f"Model saved to {model_filename}")

# selector_filename = 'selector.pkl'
# joblib.dump(selector, selector_filename)
# print(f"Selector saved to {selector_filename}")

scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to {scaler_filename}")

# ===========================
# Final Bagging Accuracy: 0.7859375
# ===========================