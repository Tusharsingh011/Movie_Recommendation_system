
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom styling
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #111827 55%, #1e1b4b 100%);
        color: #f8fafc;
    }

    .hero {
        padding: 35px 40px;
        border-radius: 24px;
        margin-bottom: 28px;
        background: linear-gradient(135deg, rgba(99,102,241,.28), rgba(236,72,153,.20));
        border: 1px solid rgba(255,255,255,.12);
        box-shadow: 0 15px 45px rgba(0,0,0,.25);
    }

    .hero h1 {
        font-size: 48px;
        margin-bottom: 5px;
        font-weight: 800;
    }

    .hero p {
        font-size: 18px;
        color: #cbd5e1;
        margin-top: 0;
    }

    .movie-card {
        padding: 22px;
        border-radius: 18px;
        min-height: 155px;
        background: rgba(255,255,255,.07);
        border: 1px solid rgba(255,255,255,.10);
        box-shadow: 0 10px 30px rgba(0,0,0,.18);
        margin-bottom: 18px;
    }

    .movie-title {
        font-size: 20px;
        font-weight: 750;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .score {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(99,102,241,.22);
        color: #c7d2fe;
        font-size: 13px;
        font-weight: 700;
    }

    .section-title {
        font-size: 28px;
        font-weight: 800;
        margin: 25px 0 15px 0;
    }

    [data-testid="stSidebar"] {
        background: rgba(7, 10, 20, .94);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.06);
        padding: 15px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,.08);
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 50px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load + cache model
# -----------------------------
@st.cache_data
def load_movies():
    df = pd.read_excel("movies_20000.xlsx")

    required = {"title", "description"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing))}"
        )

    df = df[["title", "description"]].copy()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    df = df[(df["title"] != "") & (df["description"] != "")]
    df = df.drop_duplicates(subset=["title", "description"]).reset_index(drop=True)

    return df


@st.cache_resource
def build_model(descriptions):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform(descriptions)
    return vectorizer, matrix


movies = load_movies()
_, tfidf_matrix = build_model(tuple(movies["description"]))

# -----------------------------
# Recommendation function
# -----------------------------
def recommend(movie_title, n):
    matches = movies.index[
        movies["title"].str.lower().eq(movie_title.lower())
    ].tolist()

    if not matches:
        return None

    idx = matches[0]

    scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    ranked = np.argsort(scores)[::-1]

    results = []
    for i in ranked:
        if i == idx:
            continue

        results.append({
            "title": movies.iloc[i]["title"],
            "description": movies.iloc[i]["description"],
            "score": float(scores[i])
        })

        if len(results) == n:
            break

    return results


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🎬 CineMatch AI")
    st.caption("Content-Based Movie Recommendation System")

    st.markdown("---")

    st.markdown("### ⚙️ Recommendation Settings")
    number = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=12,
        value=6
    )

    st.markdown("---")
    st.markdown("### 🧠 Model")
    st.info(
        "TF-IDF + Cosine Similarity\n\n"
        "Recommendations are generated from movie descriptions."
    )

    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.metric("Movies", f"{len(movies):,}")

# -----------------------------
# Hero section
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch AI</h1>
    <p>Discover your next favorite movie using intelligent content-based recommendations.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Dataset metrics
# -----------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("🎞️ Movies Available", f"{len(movies):,}")

with c2:
    st.metric("🧠 Recommendation Model", "TF-IDF")

with c3:
    st.metric("📐 Similarity Method", "Cosine")

# -----------------------------
# Movie selector
# -----------------------------
st.markdown('<div class="section-title">🍿 Choose a Movie</div>', unsafe_allow_html=True)

movie_titles = movies["title"].sort_values().tolist()

selected_movie = st.selectbox(
    "Search or select a movie",
    movie_titles,
    index=0
)

col1, col2 = st.columns([3, 1])

with col1:
    search_text = st.text_input(
        "Or type a movie title",
        placeholder="Example: Interstellar"
    )

with col2:
    st.write("")
    st.write("")
    recommend_button = st.button(
        "✨ Recommend Movies",
        use_container_width=True,
        type="primary"
    )

if search_text.strip():
    typed_matches = [
        title for title in movie_titles
        if search_text.lower() in title.lower()
    ]

    if typed_matches:
        selected_movie = typed_matches[0]
        st.caption(f"Using: **{selected_movie}**")
    else:
        st.warning("No matching movie title found. Try another title.")

# -----------------------------
# Recommendations
# -----------------------------
if recommend_button:
    results = recommend(selected_movie, number)

    if results is None:
        st.error("Movie not found in the dataset.")
    else:
        st.markdown(
            f'<div class="section-title">🎯 Movies Similar to {selected_movie}</div>',
            unsafe_allow_html=True
        )

        # Selected movie description
        selected_row = movies[
            movies["title"].str.lower() == selected_movie.lower()
        ].iloc[0]

        with st.expander("📖 View selected movie description"):
            st.write(selected_row["description"])

        # Cards
        for start in range(0, len(results), 2):
            cols = st.columns(2)

            for col, item in zip(cols, results[start:start + 2]):
                with col:
                    score_pct = item["score"] * 100

                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">🎥 {item["title"]}</div>
                        <span class="score">Similarity: {score_pct:.1f}%</span>
                        <p style="color:#cbd5e1; margin-top:14px; line-height:1.55;">
                            {item["description"][:260]}{"..." if len(item["description"]) > 260 else ""}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
    Built with ❤️ using Python, Pandas, Scikit-learn and Streamlit
    <br>
    <small>CineMatch AI • Content-Based Recommendation Engine</small>
</div>
""", unsafe_allow_html=True)
