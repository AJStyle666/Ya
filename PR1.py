# uber_regression.py
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42

# 1. Read dataset
df = pd.read_csv("uber_fares.csv")

# Quick peek
print(df.shape)
print(df.columns)
print(df.head())

# 2. Preprocess
# - parse datetime if present
# Adapt column names depending on dataset. Common columns: pickup_datetime, pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, passenger_count, fare_amount

# Example safe parsing:
if 'pickup_datetime' in df.columns:
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

# Haversine distance function
def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))  # kilometers

# Create distance feature if coordinates exist
coord_cols = {'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'}
if coord_cols.issubset(df.columns):
    df['distance_km'] = df.apply(lambda r: haversine(
        r['pickup_longitude'], r['pickup_latitude'],
        r['dropoff_longitude'], r['dropoff_latitude']), axis=1)

# Extract datetime features
if 'pickup_datetime' in df.columns:
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month

# Drop rows with missing target or essential features
df = df.dropna(subset=['fare_amount'])
# Drop impossible passenger counts and fares
if 'passenger_count' in df.columns:
    df = df[df['passenger_count'] > 0]
df = df[df['fare_amount'] > 0]

# 3. Identify outliers (simple rule-based)
# Remove extreme fares (e.g., > 99.5 percentile) and extreme distances
fare_995 = df['fare_amount'].quantile(0.995)
dist_995 = df['distance_km'].quantile(0.995) if 'distance_km' in df.columns else None

df = df[df['fare_amount'] <= fare_995]
if dist_995:
    df = df[df['distance_km'] <= dist_995]

# Optional: visualize distributions
# sns.histplot(df['fare_amount'], bins=80); plt.show()

# 4. Correlation check (numeric features)
numeric = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric].corr()
print("Top correlations with fare_amount:\n", corr['fare_amount'].sort_values(ascending=False).head(10))

# 5. Feature selection
features = []
for col in ['distance_km','passenger_count','hour','day_of_week','month']:
    if col in df.columns:
        features.append(col)

X = df[features]
y = df['fare_amount']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Scale numeric features (important for linear regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# 7. Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)  # tree-based models usually don't need scaled inputs
y_pred_rf = rf.predict(X_test)

# 8. Evaluation
def eval_metrics(y_true, y_pred, name="Model"):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    print(f"{name} -> R2: {r2:.4f}, RMSE: {rmse:.4f}")

eval_metrics(y_test, y_pred_lr, "LinearRegression")
eval_metrics(y_test, y_pred_rf, "RandomForest")

# 9. Simple comparison plot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(y_test, y_pred_lr, alpha=0.3)
plt.title("LR: True vs Pred")
plt.xlabel("True fare"); plt.ylabel("Pred fare")
plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.title("RF: True vs Pred")
plt.xlabel("True fare"); plt.ylabel("Pred fare")
plt.tight_layout(); plt.show()
