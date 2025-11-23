import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.title("Analyse interactive des prédictions")

history_file = "../model/prediction_history.csv"

if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
    history = pd.read_csv(history_file)
else:
    history = pd.DataFrame(columns=["temperature","humidity","weekday","occupants","prediction"])

if history.empty:
    st.info("Aucune prédiction enregistrée pour le moment. Faites d'abord une prédiction sur la page principale.")
else:
    st.subheader("Tableau des prédictions")
    st.dataframe(history)

    # Graphique interactif avec Plotly
    x_axis = st.selectbox("Variable X", ["temperature", "humidity", "weekday", "occupants"])
    y_axis = "prediction"
    color_var = st.selectbox("Couleur selon", ["weekday", "occupants", "temperature", "humidity"])

    fig = px.scatter(history, x=x_axis, y=y_axis, color=color_var,
                     size_max=15, title=f"{y_axis} vs {x_axis} coloré par {color_var}")
    st.plotly_chart(fig, use_container_width=True)

    # Télécharger historique
    st.download_button(
        label="Télécharger l'historique au format CSV",
        data=history.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv"
    )
