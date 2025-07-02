import streamlit as st
import pandas as pd
import pickle

# Load cleaned data
df = pd.read_csv("restaurant_reviews.csv")

# Load pretrained model
with open('svd_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🍽️ Recommandation de Restaurants en Mauritanie")

user_id = st.text_input("Entrez votre ID utilisateur")

if st.button("Recommander") and user_id in df['reviewerId'].values:
    all_items = df['restaurant_name'].unique()
    rated = df[df['reviewerId'] == user_id]['restaurant_name']
    not_rated = [i for i in all_items if i not in rated.values]
    preds = [(iid, model.predict(user_id, iid).est) for iid in not_rated]
    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:5]

    st.subheader("🍽️ Vos recommandations :")
    for name, score in top_preds:
        st.markdown(f"- **{name}** (score : {score:.2f})")
elif user_id and user_id not in df['reviewerId'].values:
    st.warning("Utilisateur non trouvé.")
