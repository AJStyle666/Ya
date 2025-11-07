# email_spam.py
import pandas as pd
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

# Assume columns: 'text' and 'label' (label: 'spam'/'ham' or 1/0)
text_col = 'text' if 'text' in df.columns else df.columns[0]
label_col = 'label' if 'label' in df.columns else df.columns[1]

# Map labels to 0/1 if needed
if df[label_col].dtype == object:
    df[label_col] = df[label_col].map(lambda x: 1 if str(x).lower().strip() in ['spam','1','yes','true'] else 0)

X = df[text_col].fillna("")
y = df[label_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build pipelines: TF-IDF + classifier
knn_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9)),
    ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
])

svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.9)),
    ('svm', SVC(kernel='linear', probability=True, random_state=42))
])

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