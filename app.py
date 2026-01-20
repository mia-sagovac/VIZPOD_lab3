import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Naslov i opis
st.set_page_config(
    page_title="Spotify track_genre EDA",
    layout="wide"
)

st.title("Eksploratorna Analiza audio značajki Spotify pjesama")
st.markdown(
    """
    Ova aplikacija prikazuje **interaktivnu vizualnu eksploratornu analizu**
    audio značajki Spotify pjesama, s fokusom na razlike između glazbenih žanrova.
    """
)

# Učitavanje podataka iz .csv-a
@st.cache_data
def load_data():
    return pd.read_csv("data/spotify_tracks.csv")

df = load_data()

st.write("**Ukupan broj zapisa:**", df.shape[0])

# Filteri
st.sidebar.header("Filteri")

track_genres = st.sidebar.multiselect(
    "Odaberi žanrove",
    options=sorted(df["track_genre"].unique()),
    default=sorted(df["track_genre"].unique())[:5]
)

popularity_range = st.sidebar.slider(
    "Raspon popularnosti",
    int(df["popularity"].min()),
    int(df["popularity"].max()),
    (20, 80)
)

df_filtered = df[
    (df["track_genre"].isin(track_genres)) &
    (df["popularity"].between(*popularity_range))
]

st.write("**Broj filtriranih zapisa:**", df_filtered.shape[0])

# Prikaz filtriranih zapisa
with st.expander("Prikaži filtrirane podatke (tablica)"):
    st.dataframe(
        df_filtered,
        use_container_width=True,
        height=500
    )

# Tabovi
tab1, tab2, tab3, tab4 = st.tabs(
    ["Pregled", "Distribucije", "Korelacije", "Agregacije"]
)

# Tab 1 - pregled
with tab1:
    st.subheader("Danceability vs Energy")

    fig_scatter = px.scatter(
        df_filtered,
        x="danceability",
        y="energy",
        color="track_genre",
        size="popularity",
        hover_data=["track_name", "artists"],
        title="Danceability vs Energy po žanrovima"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.info(
        "Scatter plot prikazuje odnos između plesnosti (danceability) i energije pjesama. "
        "Vidljivo je da se pojedini žanrovi grupiraju u prostoru, što sugerira slične audio karakteristike."
    )

# Tab 2 - distribucije
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribucija Danceability")
        fig_box_d = px.box(
            df_filtered,
            x="track_genre",
            y="danceability"
        )
        st.plotly_chart(fig_box_d, use_container_width=True)

    with col2:
        st.subheader("Distribucija Energy")
        fig_box_e = px.box(
            df_filtered,
            x="track_genre",
            y="energy"
        )
        st.plotly_chart(fig_box_e, use_container_width=True)

    st.info(
        "Box-plotovi prikazuju raspodjelu audio značajki po žanrovima. "
        "Neki žanrovi imaju višu medijanu energije ili plesnosti, dok su drugi raspršeniji."
    )

# Tab 3 - korelacije
with tab3:
    st.subheader("Korelacija audio značajki")

    audio_features = [
        "danceability", "energy", "valence", "acousticness",
        "instrumentalness", "liveness", "speechiness",
        "tempo", "loudness", "duration_ms", "popularity"
    ]
    audio_features = [c for c in audio_features if c in df_filtered.columns]

    corr_method = st.selectbox(
        "Metoda korelacije",
        ["pearson", "spearman", "kendall"],
        index=0
    )

    corr_matrix = df_filtered[audio_features].corr(method=corr_method)

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        title=f"Korelacijska matrica ({corr_method})"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    st.info(
        "Heatmap korelacije pokazuje koje su audio značajke međusobno povezane. "
        "Primjerice, energy i loudness često imaju jaku pozitivnu korelaciju, "
        "dok acousticness ima negativnu povezanost s energy."
    )

# Tab 4 - Agregacije
with tab4:
    # Agregacije
    st.subheader("Agregacije po žanru")

    agg_features = [
        "danceability", "energy", "valence", "acousticness",
        "instrumentalness", "liveness", "speechiness",
        "tempo", "loudness", "popularity"
    ]
    agg_features = [c for c in agg_features if c in df_filtered.columns]

    df_genre_agg = (
        df_filtered
        .groupby("track_genre")[agg_features]
        .mean(numeric_only=True)
        .reset_index()
    )

    selected_feature = st.selectbox(
        "Odaberi značajku za usporedbu po žanru",
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

    st.info(
        "Agregirani prikaz omogućuje usporedbu prosječnih vrijednosti audio značajki "
        "između žanrova. Ovo olakšava uočavanje dominantnih karakteristika svakog žanra."
    )
