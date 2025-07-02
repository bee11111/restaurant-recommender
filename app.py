import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader

# Load cleaned data
df = pd.read_csv("restaurant_reviews.csv")

# Train SVD model
reader = Reader(rating_scale=(0, 1))  # If your stars are normalized
data = Dataset.load_from_df(df[['reviewerId', 'restaurant_name', 'stars']], reader)
trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

# App interface
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
