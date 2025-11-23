# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# --- Titre de l'application ---
st.title("Prédiction de consommation énergétique")
st.write("Entrez les paramètres pour prédire la consommation d'énergie (kWh)")

# --- Charger modèle et scaler ---
model = joblib.load("../model/knn_model.pkl")
scaler = joblib.load("../model/scaler.pkl")

# --- Saisie utilisateur ---
st.sidebar.header("Paramètres de saisie")
temperature = st.sidebar.slider("Température (°C)", -10, 50, 22)
humidity = st.sidebar.slider("Humidité (%)", 0, 100, 50)
weekday = st.sidebar.selectbox(
    "Jour de la semaine", list(range(7)),
    format_func=lambda x: ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"][x]
)
occupants = st.sidebar.slider("Nombre d'occupants", 1, 20, 4)

# --- Bouton de prédiction ---
if st.button("Prédire la consommation"):

    # Préparer les données pour le modèle
    input_data = np.array([[temperature, humidity, weekday, occupants]])
    input_scaled = scaler.transform(input_data)

    # Prédiction
    prediction = model.predict(input_scaled)[0]
    st.success(f"Consommation estimée : {prediction:.2f} kWh")

    # --- Gestion historique ---
    history_file = "../model/prediction_history.csv"
    os.makedirs(os.path.dirname(history_file), exist_ok=True)

    # Lire ou créer le fichier
    if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
        history = pd.read_csv(history_file)
    else:
        history = pd.DataFrame(columns=["temperature","humidity","weekday","occupants","prediction"])
        history.to_csv(history_file, index=False)

    # Ajouter la nouvelle prédiction
    new_row = {"temperature": temperature,
               "humidity": humidity,
               "weekday": weekday,
               "occupants": occupants,
               "prediction": prediction}

    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    history.to_csv(history_file, index=False)
    st.info("Prédiction ajoutée à l'historique !")

    # --- Graphique 1 : Prédictions successives ---
    plt.figure(figsize=(6,4))
    plt.scatter(range(len(history)), history["prediction"], color='blue', label="Prédiction")
    plt.plot(range(len(history)), history["prediction"], 'r--', alpha=0.7)
    plt.xlabel("Nombre de prédictions")
    plt.ylabel("Consommation (kWh)")
    plt.title("Évolution des prédictions successives")
    plt.legend()
    st.pyplot(plt)

# --- Graphique 2 : Valeur réelle vs valeur prédite sur dataset entier ---
st.subheader("Performance du modèle sur tout le dataset")

try:
    df = pd.read_csv("../dataset/energy_consumption_dataset.csv")
    X = df[["temperature","humidity","weekday","occupants"]]
    y = df["consumption_kwh"]

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    plt.figure(figsize=(6,6))
    plt.scatter(y, y_pred, color='blue', alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    plt.xlabel("Valeur réelle (kWh)")
    plt.ylabel("Valeur prédite (kWh)")
    plt.title("Valeur réelle vs Valeur prédite")
    st.pyplot(plt)
except FileNotFoundError:
    st.warning("Dataset complet introuvable pour le graphique réel vs prédit.")
