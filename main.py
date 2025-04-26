import streamlit as st
import json
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ---- Configuration initiale ----
st.set_page_config(page_title="AgriBot Pro", page_icon="🌱")

# Création des dossiers si inexistants
os.makedirs("conversation_history", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# ---- CSS Personnalisé ----
def load_css():
    try:
        with open("assets/custom.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        st.warning("CSS non chargé")

load_css()

# ---- Initialisation Session ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lang" not in st.session_state:
    st.session_state.lang = "fr"
if "theme" not in st.session_state:
    st.session_state.theme = "Tous"

# ---- Chargement des Données ----
@st.cache_data
def load_data():
    with open('connaissances.json', 'r', encoding='utf-8') as f:
        return json.load(f)

data = load_data()

# ---- Fonctions Utiles ----
def get_response(prompt):
    """Génère une réponse à partir de la question"""
    questions = []
    reponses = []
    
    categories = data.values() if st.session_state.theme == "Tous" else [data[st.session_state.theme]]
    
    for categorie in categories:
        for item in categorie:
            if st.session_state.lang in item:
                questions.append(item[st.session_state.lang]['question'])
                reponses.append(item[st.session_state.lang]['reponse'])
    
    if questions:
        vectorizer = TfidfVectorizer().fit(questions + [prompt])
        vectors = vectorizer.transform(questions + [prompt])
        sim = cosine_similarity(vectors[-1], vectors[:-1])[0]
        best_idx = sim.argmax()
        return reponses[best_idx] if sim[best_idx] > 0.3 else "Je ne sais pas répondre à cette question."
    return "Aucune donnée disponible pour ce thème."

def save_conversation():
    """Sauvegarde la conversation dans un CSV"""
    df = pd.DataFrame({
        "date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "langue": [st.session_state.lang],
        "theme": [st.session_state.theme],
        "messages": [str(st.session_state.messages)]
    })
    
    save_path = "conversation_history/historiques.csv"
    df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)
    return save_path

# ---- Sidebar ----
with st.sidebar:
    st.title("⚙️ Paramètres")
    
    # Sélection de langue
    st.subheader("🌍 Langue")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🇫🇷 Français", key="fr_btn", help="Passer en français"):
            st.session_state.lang = "fr"
    with col2:
        if st.button("🇲🇱 Bambara", key="br_btn", help="Passer en bambara"):
            st.session_state.lang = "br"
    
    # Sélection de thème
    st.subheader("📚 Thèmes")
    theme = st.radio(
        "Choisir un thème:",
        ["Tous"] + list(data.keys()),
        key="theme_selector",
        label_visibility="collapsed"
    )
    st.session_state.theme = theme
    
    # Questions suggérées
    st.subheader("💡 Questions rapides")
    if st.session_state.theme != "Tous":
        for i, item in enumerate(data[st.session_state.theme][:5]):
            q = item[st.session_state.lang]['question']
            if st.button(f"• {q}", key=f"suggest_{i}"):
                st.session_state.messages.append({"role": "user", "content": q})
                response = get_response(q)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    # Gestion historique
    st.subheader("💾 Sauvegarde")
    if st.button("Sauvegarder cette conversation", help="Enregistre dans conversation_history/"):
        save_path = save_conversation()
        st.success(f"Conversation sauvegardée dans : {save_path}")
    
    if st.checkbox("Afficher l'historique"):
        try:
            hist = pd.read_csv("conversation_history/historiques.csv")
            st.dataframe(hist)
        except:
            st.warning("Aucun historique existant")

# ---- Interface Principale ----
st.title("🌱 AgriBot")
st.caption(f"Langue: {st.session_state.lang.upper()} | Thème: {st.session_state.theme}")

# Affichage des messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Gestion nouvelle question
if prompt := st.chat_input("Écrivez votre question ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    save_conversation()  # Sauvegarde automatique
    st.rerun()