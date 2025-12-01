# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from database import create_db, authenticate_user, add_user
create_db()

# Partie Authentification et Inscription
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False

# ===============================
# Connexion et Inscription
# ===============================
if not st.session_state.authenticated:
    st.title("🔐 Connexion")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Se connecter"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("✅ Connexion réussie")
                st.stop()
            else:
                st.error("❌ Nom d'utilisateur ou mot de passe incorrect")
    with col2:
        if st.button("Créer un compte"):
            st.session_state.show_signup = True

    # --- Formulaire d'inscription visible uniquement si bouton cliqué ---
    if st.session_state.show_signup:
       st.write("### 📝 Inscription")
       st.text_input("Nom d'utilisateur", key="signup_username")
       st.text_input("Mot de passe", type="password", key="signup_password")
       st.text_input("Nom complet", key="signup_fullname")
       st.text_input("Email", key="signup_email")
       st.text_input("Téléphone", key="signup_phone")

       if st.button("Valider l'inscription"):
           username_val = st.session_state.signup_username
           password_val = st.session_state.signup_password
           fullname_val = st.session_state.signup_fullname
           email_val = st.session_state.signup_email
           phone_val = st.session_state.signup_phone

           if username_val and password_val and fullname_val and email_val and phone_val:
              if add_user(username_val, password_val, fullname_val, email_val, phone_val):
                  st.success("🎉 Compte créé ! Reconnecte-toi.")
                  st.session_state.show_signup = False
              else:
                  st.error("⚠ Nom d’utilisateur ou email déjà existant.")
           else:
              st.warning("✍ Remplis tous les champs pour t’inscrire.")
    st.stop()

# ===============================
# Sidebar Logout
# ===============================
st.sidebar.write(f"👤 Connecté : {st.session_state.username}")
if st.sidebar.button("Se déconnecter"):
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.stop() 

# --- Fonctions ajoutées ---
def conseils_conso(valeur_predite):
    if valeur_predite > 50:
        return [
            "Éteindre les appareils en veille",
            "Utiliser des ampoules LED",
            "Optimiser le chauffage/climatisation",
            "Débrancher les chargeurs non utilisés"
        ]
    elif valeur_predite > 30:
        return [
            "Utiliser les appareils en heures creuses",
            "Réduire les équipements énergivores"
        ]
    else:
        return [
            "✅ Bonne gestion d’énergie ! Continue comme ça."
        ]

def niveau_conso(valeur):
    if valeur > 50:
        return "<span style='color:red;font-size:20px;'>🔴 Consommation élevée</span>"
    elif valeur > 30:
        return "<span style='color:orange;font-size:20px;'>🟠 Consommation normale</span>"
    else:
        return "<span style='color:green;font-size:20px;'>🟢 Consommation faible</span>"

# --- Prix du kWh en DH ---
PRIX_KWH = 0.2  # adapter selon le tarif réel

# --- Navigation entre pages ---
page = st.sidebar.radio("Navigation", ["Prédiction", "Analyse & Historique", "Prévisions sur plusieurs jours"])


# --- Charger modèle et scaler ---
model = joblib.load("../model/knn_model.pkl")
scaler = joblib.load("../model/scaler.pkl")

# ========================
# 📌 PAGE 1 : PRÉDICTION
# ========================
if page == "Prédiction":
    st.title("⚡ Prédiction de consommation énergétique")
    st.write("Sélectionne les paramètres dans la barre latérale, puis clique sur *Prédire la consommation*.")

    # --- Saisie utilisateur ---
    st.sidebar.header("Paramètres de saisie")
    temperature = st.sidebar.slider("Température (°C)", -10, 50, 22)
    humidity = st.sidebar.slider("Humidité (%)", 0, 100, 50)
    weekday = st.sidebar.selectbox(
        "Jour de la semaine", list(range(7)),
        format_func=lambda x: ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"][x]
    )
    occupants = st.sidebar.slider("Nombre d'occupants", 1,50, 4)

    # --- Bouton de prédiction ---
    if st.button("Prédire la consommation"):
        input_data = np.array([[temperature, humidity, weekday, occupants]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        # Consommation
        st.success(f"⚡ Consommation estimée : {prediction:.2f} kWh / jour")

        # Coût
        cout_pred = prediction * PRIX_KWH
        st.info(f"💰 Coût estimé : {cout_pred:.2f} DH / jour")

        # Niveau coloré
        st.markdown(niveau_conso(prediction), unsafe_allow_html=True)

        # Conseils d’économie
        st.write("### 💡 Conseils pour réduire la consommation :")
        for c in conseils_conso(prediction):
            st.write("• " + c)

        # --- Gestion historique CSV ---
        history_file = "../model/prediction_history.csv"
        os.makedirs(os.path.dirname(history_file), exist_ok=True)

        if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
            history = pd.read_csv(history_file)
        else:
            history = pd.DataFrame(columns=["temperature","humidity","weekday","occupants","prediction","cout"])
            history.to_csv(history_file, index=False)

        new_row = {
            "temperature": temperature,
            "humidity": humidity,
            "weekday": weekday,
            "occupants": occupants,
            "prediction": prediction,
            "cout": cout_pred
        }

        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        history.to_csv(history_file, index=False)
        st.info("✅ Prédiction ajoutée à l’historique")

# ===============================
# 📊 PAGE 2 : ANALYSE & HISTORIQUE
# ===============================
if page == "Analyse & Historique":
    st.title("📊 Analyse & Historique des prédictions")

    # Charger historique
    history_file = "../model/prediction_history.csv"
    if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
        history = pd.read_csv(history_file)
        st.dataframe(history)
    else:
        st.warning("Aucun historique disponible.")
        history = None

    # --- Analyse performance modèle sur dataset global ---
    st.subheader("🎯 Performance du modèle sur le dataset complet")

    try:
        df = pd.read_csv("../dataset/energy_consumption_dataset.csv")
        X = df[["temperature","humidity","weekday","occupants"]]
        y = df["consumption_kwh"]
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        # Calcul des métriques
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        st.write(f"**MAE :** {mae:.2f} kWh")
        st.write(f"**RMSE :** {rmse:.2f} kWh")
        st.write(f"**R² Score :** {r2:.2f}")

        # Scatter réel vs prédit
        st.write("### Valeurs réelles vs prédites")
        plt.figure()
        plt.scatter(y, y_pred, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], "--")
        plt.xlabel("Valeur réelle (kWh)")
        plt.ylabel("Valeur prédite (kWh)")
        plt.title("Réelle vs Prédite")
        st.pyplot(plt)

        # Distribution globale
        st.write("### Distribution des prédictions (dataset global)")
        plt.figure()
        plt.hist(y_pred)
        plt.title("Distribution des valeurs prédites")
        plt.xlabel("kWh")
        plt.ylabel("Fréquence")
        st.pyplot(plt)

    except FileNotFoundError:
        st.warning("Dataset global introuvable pour l'analyse.")

    # --- Analyse historique si disponible ---
    if history is not None:
        st.subheader("📈 Évolution des prédictions enregistrées")

        plt.figure()
        plt.scatter(range(len(history)), history["prediction"])
        plt.plot(range(len(history)), history["prediction"], "--", alpha=0.7)
        plt.xlabel("Nombre de prédictions")
        plt.ylabel("kWh")
        plt.title("Évolution des prédictions successives")
        st.pyplot(plt)

        plt.figure()
        plt.hist(history["prediction"])
        plt.xlabel("kWh")
        plt.ylabel("Fréquence")
        plt.title("Distribution des prédictions enregistrées")
        st.pyplot(plt)

        # Histogramme du coût
        plt.figure()
        plt.hist(history["cout"])
        plt.xlabel("DH")
        plt.ylabel("Fréquence")
        plt.title("Distribution des coûts estimés")
        st.pyplot(plt)

    # --- Télécharger historique ---
    if history is not None:
        csv_data = history.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Télécharger l’historique CSV", data=csv_data, file_name="prediction_history.csv")
# ===============================
# 📅 PAGE 3 : PRÉVISIONS SUR PLUSIEURS JOURS
# ===============================
if page == "Prévisions sur plusieurs jours":
    st.title("📅 Prévisions de consommation sur plusieurs jours")
    st.write("Simulez la consommation et le coût sur une période donnée.")

    # Nombre de jours à simuler
    n_days = st.slider("Nombre de jours à simuler", 1, 30, 7)

    # Paramètres de base (utilisateur peut ajuster)
    base_temp = st.slider("Température moyenne (°C)", -10, 50, 22)
    base_humidity = st.slider("Humidité moyenne (%)", 0, 100, 50)
    occupants = st.slider("Nombre d'occupants", 1, 20, 4)

    # Génération des jours et variations aléatoires
    np.random.seed(42)  # pour reproductibilité
    temperatures = base_temp + np.random.normal(0, 3, n_days)  # petite variation
    humidities = base_humidity + np.random.normal(0, 5, n_days)
    weekdays = np.arange(n_days) % 7  # 0 = lundi, ..., 6 = dimanche

    # Prédiction consommation et coût pour chaque jour
    predictions = []
    couts = []
    for t, h, w in zip(temperatures, humidities, weekdays):
        input_data = np.array([[t, h, w, occupants]])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        predictions.append(pred)
        couts.append(pred * PRIX_KWH)

    # Affichage tableau
    df_simulation = pd.DataFrame({
        "Jour": np.arange(1, n_days+1),
        "Température": temperatures,
        "Humidité": humidities,
        "Consommation (kWh)": predictions,
        "Coût (DH)": couts
    })
    st.subheader("Tableau des prévisions")
    st.dataframe(df_simulation)

    # Graphique consommation
    st.subheader("Graphique consommation")
    plt.figure(figsize=(8,4))
    plt.plot(df_simulation["Jour"], df_simulation["Consommation (kWh)"], marker='o', color='blue')
    plt.xlabel("Jour")
    plt.ylabel("Consommation (kWh)")
    plt.title("Prévision de consommation sur plusieurs jours")
    plt.grid(True)
    st.pyplot(plt)

    # Graphique coût
    st.subheader("Graphique coût")
    plt.figure(figsize=(8,4))
    plt.plot(df_simulation["Jour"], df_simulation["Coût (DH)"], marker='o', color='green')
    plt.xlabel("Jour")
    plt.ylabel("Coût (DH)")
    plt.title("Prévision du coût sur plusieurs jours")
    plt.grid(True)
    st.pyplot(plt)
