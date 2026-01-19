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
