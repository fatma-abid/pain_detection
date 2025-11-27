# app.py
import streamlit as st
from pain_detector import PainPredictor
import tempfile
from PIL import Image

st.set_page_config(page_title="Détection de Douleur", layout="wide")
st.title("🔮 Détection de Douleur")

@st.cache_resource
def load_model():
    return PainPredictor('best_pain_model.pkl')

detector = load_model()

option = st.radio("Choisissez une option :", ("Tester une image", "Tester un dossier", "Tester sur le test set"))

# --- Tester une image ---
if option == "Tester une image":
    uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg","jpeg","png"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        result = detector.predict_single_image(temp_path)
        if result:
            st.subheader("Résultat de la prédiction")
            st.write(f"**Classe prédite :** {result['pain_type']}")
            if result['probabilities'] is not None:
                st.subheader("Probabilités")
                for i, prob in enumerate(result['probabilities']):
                    st.write(f"{detector.pain_labels[i]} : {prob*100:.2f}%")
            st.subheader("Image avec landmarks")
            st.image(result['visualization'], use_column_width=True)

# --- Tester un dossier ---
elif option == "Tester un dossier":
    folder_path = st.text_input("Chemin vers le dossier contenant les images")
    pattern = st.text_input("Pattern (par défaut *.jpg)", "*.jpg")
    if st.button("Lancer la prédiction sur le dossier"):
        if folder_path:
            df_results = detector.predict_batch(folder_path, pattern)
            if df_results is not None:
                st.dataframe(df_results)
                df_results.to_csv("batch_predictions.csv", index=False)
                st.success("Résultats sauvegardés dans batch_predictions.csv")

# --- Tester sur le test set ---
elif option == "Tester sur le test set":
    n_samples = st.number_input("Nombre d'échantillons à tester", min_value=1, value=5)
    if st.button("Lancer le test"):
        detector.test_on_test_set(n_samples=n_samples)
