import streamlit as st
import movie_recommender as rec

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background: radial-gradient(circle at top,#0f172a,#020617);
}

.title {
    font-size: 48px;
    font-weight: 700;
    color: #f1f5f9;
}

.subtitle {
    color: #94a3b8;
    margin-bottom: 30px;
}

.movie-card {
    background: #111827;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    text-align: center;
    transition: 0.3s;
}
.movie-card:hover {
    transform: scale(1.05);
    border-color: #6366f1;
}

button[kind="primary"] {
    background: linear-gradient(90deg,#6366f1,#22c55e);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover movies similar to your favorites</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2,1])

movie_titles = rec.data['title'].values

with col1:
    selected_movie = st.selectbox("Choose a Movie", movie_titles)

with col2:
    num = st.slider("Recommendations", 1, 12, 6)

if st.button("✨ Recommend Movies"):

    idx = rec.data[rec.data['title']==selected_movie].index[0]
    top_idx = rec.simi_matrix[idx].argsort()[::-1][1:num+1]

    st.markdown("## 🍿 Recommended For You")

    cols = st.columns(3)

    for i, movie_index in enumerate(top_idx):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="movie-card">
                    <h4>{rec.data['title'].iloc[movie_index]}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
