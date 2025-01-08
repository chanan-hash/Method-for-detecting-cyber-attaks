import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import joblib  # for saving the model

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

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

# ===========================
# STEP 4: FEATURE SELECTION
# ===========================
k = 30
selector = SelectKBest(score_func=f_classif, k=k)
X_train_kbest = selector.fit_transform(X_train_res, y_train_res)
X_test_kbest = selector.transform(X_test)

# ===========================
# STEP 5: FINAL MODEL
# ===========================
bagging_clf_final = BaggingClassifier(
    n_estimators=800,
    max_samples=1.0,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)

bagging_clf_final.fit(X_train_kbest, y_train_res)
acc_final = accuracy_score(y_test, bagging_clf_final.predict(X_test_kbest))
print("Final Bagging Accuracy:", acc_final)

# ===========================
# STEP 6: SAVE THE MODEL AND SELECTOR
# ===========================
model_filename = 'bagging_model.pkl'
joblib.dump(bagging_clf_final, model_filename)
print(f"Model saved to {model_filename}")

selector_filename = 'selector.pkl'
joblib.dump(selector, selector_filename)
print(f"Selector saved to {selector_filename}")


# ===========================
# Final Bagging Accuracy: 0.7859375
# ===========================