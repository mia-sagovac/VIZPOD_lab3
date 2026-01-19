import streamlit as st
import pandas as pd
import plotly.express as px

# Naslov i opis
st.set_page_config(
    page_title="Spotify track_genre EDA",
    layout="wide"
)

st.title("Spotify track_genre Exploratory Analysis")
st.write(
    "Interaktivna vizualna eksploratorna analiza audio značajki Spotify pjesama."
)

# Učitavanje podataka iz .csv-a
@st.cache_data
def load_data():
    return pd.read_csv("data/spotify_tracks.csv")

df = load_data()

st.write("Broj zapisa:", df.shape[0])



# Postavljanje filtera
st.sidebar.header("Filteri")

track_genres = st.sidebar.multiselect(
    "Odaberi žanrove",
    options=sorted(df["track_genre"].unique()),
    default=sorted(df["track_genre"].unique())[:5]
)

popularity_range = st.sidebar.slider(
    "Popularnost",
    int(df["popularity"].min()),
    int(df["popularity"].max()),
    (20, 80)
)

df_filtered = df[
    (df["track_genre"].isin(track_genres)) &
    (df["popularity"].between(*popularity_range))
]

st.write("Filtrirani zapisi:", df_filtered.shape[0])


# Prikaz filtriranih zapisa
if st.checkbox("Prikaži filtrirane podatke (tablica)"):
    st.dataframe(
        df_filtered,
        use_container_width=True,
        height=500
    )



# Prva vizualizacija - Danceability vs Energy
fig = px.scatter(
    df_filtered,
    x="danceability",
    y="energy",
    color="track_genre",
    size="popularity",
    hover_data=["track_name", "artists"],
    title="Danceability vs Energy by Genre"
)

st.plotly_chart(fig, use_container_width=True)


# Box-plot distribucija za Danceability vs Energy
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribucija Danceability")
    fig1 = px.box(df_filtered, x="track_genre", y="danceability")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Distribucija Energy")
    fig2 = px.box(df_filtered, x="track_genre", y="energy")
    st.plotly_chart(fig2, use_container_width=True)

# heat map korelacije
st.subheader("Korelacija audio značajki (Heatmap)")

audio_features = [
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "liveness", "speechiness",
    "tempo", "loudness", "duration_ms", "popularity"
]

audio_features = [c for c in audio_features if c in df_filtered.columns]

if len(audio_features) >= 2:
    corr_method = st.selectbox(
        "Metoda korelacije",
        ["pearson", "spearman", "kendall"],
        index=0
    )

    corr_matrix = df_filtered[audio_features].corr(method=corr_method)

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title=f"Korelacijska matrica ({corr_method})"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("pogreska")

# pca

st.subheader("PCA (2D projekcija audio značajki)")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pca_features = [
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "liveness", "speechiness",
    "tempo", "loudness"
]
pca_features = [c for c in pca_features if c in df_filtered.columns]

if len(pca_features) >= 2 and df_filtered.shape[0] >= 5:
    pca_df = df_filtered.dropna(subset=pca_features).copy()

    X = pca_df[pca_features].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X_scaled)

    pca_df["PC1"] = comps[:, 0]
    pca_df["PC2"] = comps[:, 1]

    explained = pca.explained_variance_ratio_ * 100

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="track_genre" if "track_genre" in pca_df.columns else None,
        size="popularity" if "popularity" in pca_df.columns else None,
        hover_data=["track_name", "artists"] if all(c in pca_df.columns for c in ["track_name", "artists"]) else None,
        title=f"PCA: PC1 ({explained[0]:.1f}%) vs PC2 ({explained[1]:.1f}%)"
    )
    st.plotly_chart(fig_pca, use_container_width=True)
else:
    st.info("pogreska")

#agregacije po žanru
st.subheader("Agregacije po žanru (prosječne vrijednosti)")

agg_features = [
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "liveness", "speechiness",
    "tempo", "loudness", "popularity"
]
agg_features = [c for c in agg_features if c in df_filtered.columns]

if "track_genre" in df_filtered.columns and len(agg_features) > 0:
    df_genre_agg = (
        df_filtered
        .groupby("track_genre")[agg_features]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("popularity", ascending=False) if "popularity" in agg_features else
        df_filtered.groupby("track_genre")[agg_features].mean(numeric_only=True).reset_index()
    )

    st.dataframe(df_genre_agg, use_container_width=True)

    st.markdown("### Bar chart: odaberi feature")
    selected_feature = st.selectbox(
        "Feature za usporedbu po žanru",
        options=agg_features,
        index=agg_features.index("popularity") if "popularity" in agg_features else 0
    )

    fig_bar = px.bar(
        df_genre_agg.sort_values(selected_feature, ascending=False),
        x="track_genre",
        y=selected_feature,
        title=f"Prosječni {selected_feature} po žanru"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("pogreska")