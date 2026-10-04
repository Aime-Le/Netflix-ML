import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Netflix Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Netflix Content Recommendation System")
st.write("Choose a Netflix title and test recommendations based on TF-IDF and cosine similarity.")

uploaded = st.sidebar.file_uploader("Upload your Dataset.csv", type=["csv"])
if uploaded is None:
    st.info("Upload Dataset.csv using the left sidebar to start.")
    st.stop()

df = pd.read_csv(uploaded)
required = ["title", "type", "director", "country", "rating", "listed_in"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

df = df.dropna(subset=["title"]).copy()
df["title"] = df["title"].astype(str)
df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
for c in required:
    if c != "title":
        df[c] = df[c].fillna("").astype(str)

features = ["type", "director", "country", "rating", "listed_in"]
df["content_features"] = df[features].agg(" ".join, axis=1)
tfidf = TfidfVectorizer(stop_words="english")
matrix = tfidf.fit_transform(df["content_features"])
similarity = cosine_similarity(matrix, matrix)

st.sidebar.header("Test the system")
selected_title = st.sidebar.selectbox("Select a Netflix title", sorted(df["title"].tolist()))
n = st.sidebar.slider("Number of recommendations", 3, 10, 5)
selected_index = df.index[df["title"] == selected_title][0]
selected = df.iloc[selected_index]

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Selected title")
    st.write("**Title:**", selected["title"])
    st.write("**Type:**", selected["type"])
    st.write("**Release year:**", selected["release_year"] if "release_year" in df.columns else "Not available")
    st.write("**Rating:**", selected["rating"] or "Not available")
    st.write("**Categories:**", selected["listed_in"] or "Not available")
    st.write("**Country:**", selected["country"] or "Not available")

indices = similarity[selected_index].argsort()[::-1]
indices = [i for i in indices if i != selected_index][:n]
results = df.iloc[indices].copy()
results["Similarity score"] = [round(float(similarity[selected_index, i]), 3) for i in indices]

with col2:
    st.subheader("Recommended titles")
    display_cols = ["title", "type", "release_year", "rating", "listed_in", "Similarity score"]
    display_cols = [c for c in display_cols if c in results.columns]
    st.dataframe(results[display_cols], use_container_width=True, hide_index=True)
    st.caption("Similarity scores measure text-feature similarity; they do not guarantee a viewer will like a title.")

st.divider()
st.subheader("🧪 Quick test: category overlap")
st.write("Checks whether each recommendation shares at least one category with the selected title. This is a simple diagnostic, not a formal accuracy score.")

def categories(value):
    return {x.strip().lower() for x in str(value).split(",") if x.strip()}

selected_categories = categories(selected["listed_in"])
if selected_categories and len(results) > 0:
    overlaps = [
        bool(selected_categories.intersection(categories(value)))
        for value in results["listed_in"]
    ]
    rate = sum(overlaps) / len(overlaps)
    a, b, c = st.columns(3)
    a.metric("Recommendations tested", len(overlaps))
    b.metric("Shared a category", sum(overlaps))
    c.metric("Category overlap rate", f"{rate:.0%}")
    if any(overlaps):
        st.success("At least one recommendation shares a category with the selected title.")
    else:
        st.warning("No recommendations shared a category in this test. Try another title.")
else:
    st.info("Category overlap cannot be calculated because category data is missing.")

with st.expander("How the system works"):
    st.markdown("1. Combines type, director, country, rating, and categories.\n2. Converts text into numerical features using TF-IDF.\n3. Uses cosine similarity to compare titles.\n4. Displays the closest titles, excluding the selected title.\n5. Checks category overlap as a simple diagnostic.")

st.caption(f"Dataset loaded: {len(df):,} titles | Features: {', '.join(features)}")
