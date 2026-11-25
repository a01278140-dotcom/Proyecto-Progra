import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.utils.set_random_seed(42)
df = pd.DataFrame(pd.read_csv("traffic (1).csv"))

df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values(by=["DateTime", "Junction"]).reset_index(drop=True)
df["Date"] = df["DateTime"].dt.date 
df["Mes"] = df["DateTime"].dt.month
df["Año"] = df["DateTime"].dt.year
df["Hora"] = df["DateTime"].dt.hour
df["Dia"] = df["DateTime"].dt.day
df["Dia_Semana"] = df["DateTime"].dt.dayofweek

# Número de vehículos por cada intersección
vehicles_per_junction = df.groupby(["Junction"])["Vehicles"].sum().reset_index()

# Número de vehículos por día e intersección
vehicles_per_day_junction = df.groupby(["Junction", "Date"])["Vehicles"].sum().reset_index()

# Número de vehículos por mes e intersección
vehicles_per_month_junction = df.groupby(["Junction", "Mes"])["Vehicles"].sum().reset_index()

# Número de vehículos por año e intersección
vehicles_per_year_junction = df.groupby(["Junction", "Año"])["Vehicles"].sum().reset_index()

# Número de vehículos por hora e intersección
vehicles_per_hour_junction = df.groupby(["Junction", "Hora"])["Vehicles"].sum().reset_index()

# Número de vehículos por día de la semana e intersección
vehicles_per_weekday_junction = df.groupby(["Junction", "Dia_Semana"])["Vehicles"].sum().reset_index()

"""
print("--- Total por intersección ---")
print(vehicles_per_junction)

print("\n--- Total por intersección y día ---")
print(vehicles_per_day_junction)

print("\n--- Total por intersección y mes ---")
print(vehicles_per_month_junction)

print("\n--- Total por intersección y año ---")
print(vehicles_per_year_junction)

print("\n--- Total por intersección y hora ---")
print(vehicles_per_hour_junction)

print("\n--- Total por intersección y día de la semana ---")
print(vehicles_per_weekday_junction)
"""

# Random Forest

X = df[["Junction", "Año", "Mes", "Dia", "Hora", "Dia_Semana"]]
y = df["Vehicles"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

modelo = RandomForestRegressor(
    n_estimators=110, 
    max_depth=9,
    random_state=43,
    n_jobs=-1)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Resultados Random Forest ---")
print(f"MAE: {mae:.2f} %")
print(f"MSE: {mse:.2f} ")
print(f"R2: {r2:.4f}")



# Red Neuronal (Keras)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo_nn = Sequential([
    Dense(32, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    #Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1)      
])

modelo_nn.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

historial = modelo_nn.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

y_pred_nn = modelo_nn.predict(X_test_scaled).flatten()

mae_nn = mean_absolute_error(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print(f"\n--- Resultados Red Neuronal ---")
print(f"MAE: {mae_nn:.2f} autos")
print(f"MSE: {mse_nn:.2f} autos²")
print(f"R2: {r2_nn:.4f}")