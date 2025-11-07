# bank_churn_nn.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    USE_TF = True
except Exception:
    # TensorFlow not available — we'll fall back to scikit-learn's MLPClassifier
    USE_TF = False
    from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Read
df = pd.read_csv("Churn_Modelling.csv")
print(df.shape)
print(df.columns)

# 2. Feature/target separation
# Typical Kaggle columns: RowNumber, CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited
df = df.drop(columns=['RowNumber'], errors='ignore')

if 'CustomerId' in df.columns:
    df = df.drop(columns=['CustomerId'])
if 'Surname' in df.columns:
    df = df.drop(columns=['Surname'])

target = 'Exited'
X = df.drop(columns=[target])
y = df[target]

# Categorical encoding
X = pd.get_dummies(X, columns=[c for c in ['Geography','Gender'] if c in X.columns], drop_first=True)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Build model
if USE_TF:
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Early stopping to avoid overfitting
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train_scaled, y_train, validation_split=0.15, epochs=50, batch_size=64, callbacks=[es], verbose=1)

    # 6. Evaluate
    y_pred_prob = model.predict(X_test_scaled).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
else:
    # Fallback using scikit-learn MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                        early_stopping=True, validation_fraction=0.15, max_iter=200, random_state=42)
    history = None
    mlp.fit(X_train_scaled, y_train)
    if hasattr(mlp, 'predict_proba'):
        y_pred_prob = mlp.predict_proba(X_test_scaled)[:, 1]
    else:
        # predict_proba may not be available; use decision_function as proxy
        try:
            scores = mlp.decision_function(X_test_scaled)
            # scale to [0,1] with a simple sigmoid
            y_pred_prob = 1 / (1 + np.exp(-scores))
        except Exception:
            y_pred_prob = mlp.predict(X_test_scaled)
    y_pred = (y_pred_prob >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d'); plt.title("Confusion Matrix"); plt.show()

# Plot training history (only available when using TensorFlow)
if history is not None:
    try:
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend(); plt.show()
    except Exception:
        pass
