# email_spam.py
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Read
df = pd.read_csv("emails.csv")
print(df.shape)
print(df.columns)
print(df.head())

# Detect likely label column (common names) and text column.
possible_label_names = ['label', 'Label', 'prediction', 'Prediction', 'target', 'Target']
label_col = None
for name in possible_label_names:
    if name in df.columns:
        label_col = name
        break

# Fallback: if there's a column named 'Prediction' or the last column looks like a label, try that
if label_col is None:
    if 'Prediction' in df.columns:
        label_col = 'Prediction'
    else:
        # try last column if it seems to be binary-ish
        last = df.columns[-1]
        if df[last].nunique() <= 10:
            label_col = last
        else:
            # give up and raise a helpful error
            raise RuntimeError('Could not automatically find a label column. Please set label_col explicitly.')

# Detect text column if present
text_col = 'text' if 'text' in df.columns else None
# If a 'text' column exists but is not string-like, treat dataset as numeric features
if text_col is not None and not is_string_dtype(df[text_col].dtype):
    print("Note: 'text' column exists but is not string-like; treating dataset as precomputed numeric features.")
    text_col = None

# If we have a text column, X will be raw text; otherwise assume the frame already contains numeric features (bag-of-words)
if text_col is not None:
    X = df[text_col].fillna("")
else:
    # drop non-feature columns (Email No., label_col)
    drop_cols = [label_col]
    if 'Email No.' in df.columns:
        drop_cols.append('Email No.')
    X = df.drop(columns=drop_cols, errors='ignore')

# Prepare y. Map textual labels to 0/1 if necessary, otherwise keep as-is.
y = df[label_col]
if y.dtype == object:
    y = y.map(lambda x: 1 if str(x).lower().strip() in ['spam','1','yes','true'] else 0)

# Train/test split. Only stratify if every class has at least 2 members.
stratify_arg = y if y.value_counts().min() >= 2 else None
if stratify_arg is None:
    print('Warning: not stratifying train/test split because at least one class has fewer than 2 members.')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)

# Build pipelines: TF-IDF + classifier
# Build appropriate models depending on whether X is text or numeric features
if text_col is not None:
    knn_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9)),
        ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
    ])

    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9)),
        ('svm', SVC(kernel='linear', probability=True, random_state=42))
    ])
else:
    # X already numeric (bag-of-words / counts). Don't use TF-IDF vectorizer.
    knn_pipeline = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    svm_pipeline = SVC(kernel='linear', probability=True, random_state=42)

# Fit and predict
knn_pipeline.fit(X_train, y_train)
y_pred_knn = knn_pipeline.predict(X_test)

svm_pipeline.fit(X_train, y_train)
y_pred_svm = svm_pipeline.predict(X_test)

# Evaluate
def report(y_true, y_pred, name):
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d'); plt.title(name + " Confusion Matrix"); plt.show()

report(y_test, y_pred_knn, "KNN")
report(y_test, y_pred_svm, "SVM (linear)")

# Notes:
# - SVM with linear kernel often performs well for text.
# - KNN can be slow with large TF-IDF and doesn't scale well; consider Naive Bayes (MultinomialNB) for baseline.
plt.xlabel("True label"); plt.ylabel("Pred label")