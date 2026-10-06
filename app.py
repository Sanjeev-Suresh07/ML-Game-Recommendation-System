from __future__ import annotations

import ast
import re
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from src.recommender import GameRecommender, demo_games


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "dataset" / "games.csv"
DATA_COLUMNS = [
    "appid", "name", "release_date", "price", "recommendations", "positive",
    "negative", "header_image", "categories", "genres", "tags", "short_description",
]

st.set_page_config(page_title="Sidequest — Find your next game", page_icon="🎮", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');
:root { --paper:#101613; --ink:#f2f4eb; --forest:#19241e; --soft:#a6b0a7; --rule:#303b34; --orange:#d9f36a; --card:#18211c; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(ellipse at 52% -25%,#304337 0%,#101613 55%); color:var(--ink); }
header[data-testid="stHeader"] { background:rgba(16,22,19,.94); }
.block-container { max-width:1260px; padding-top:3.2rem; padding-bottom:4rem; }
#MainMenu, footer { visibility:hidden; }
.masthead { display:flex; justify-content:space-between; align-items:center; padding:5px 0 18px; border-bottom:1px solid var(--rule); }
.wordmark { color:#f2f4eb!important; opacity:1!important; font-size:1.04rem; font-weight:700; letter-spacing:.1em; }
.wordmark span { color:var(--orange); }
.mast-note { color:#b8c4ba!important; opacity:1!important; font-size:.82rem; }
.hero { display:flex; justify-content:space-between; align-items:flex-end; gap:30px; background:linear-gradient(120deg,#26382d,#17221c 70%); color:#f2f4eb; border:1px solid #3d5143; border-radius:10px; padding:42px 48px; margin:22px 0 29px; box-shadow:0 20px 60px #0003; }
.hero-kicker { color:var(--orange); font-size:.71rem; font-weight:700; text-transform:uppercase; letter-spacing:.15em; }
.hero h1 { font:400 clamp(2.35rem,5vw,4rem)/1.03 'DM Serif Display',Georgia,serif; letter-spacing:-.025em; margin:.62rem 0 .85rem; max-width:670px; }
.hero p { color:#c0cbc1; font-size:.98rem; line-height:1.65; max-width:570px; margin:0; }
.hero-stat { flex:0 0 180px; border-left:1px solid rgba(255,255,255,.22); padding-left:24px; margin-bottom:4px; }
.hero-stat strong { display:block; font:400 2.1rem 'DM Serif Display',Georgia,serif; color:var(--orange); }
.hero-stat span { color:#c0cbc1; font-size:.78rem; line-height:1.5; }
.eyebrow { color:var(--orange); text-transform:uppercase; letter-spacing:.14em; font-size:.68rem; font-weight:700; }
.section-title { font:400 1.8rem 'DM Serif Display',Georgia,serif; margin:.15rem 0 .2rem; color:var(--ink); }
.subtle { color:var(--soft); font-size:.88rem; line-height:1.55; }
.result-head { display:flex; justify-content:space-between; align-items:end; margin:24px 0 12px; }
.game-title { font:600 1.02rem 'DM Sans',sans-serif; line-height:1.3; color:var(--ink); margin:.7rem 0 .35rem; min-height:2.6em; }
.game-description { font-size:.82rem; line-height:1.55; color:#b3beb5; min-height:3.85em; margin:.2rem 0 .55rem; }
.game-meta { color:#a3afa5; font-size:.74rem; letter-spacing:.015em; padding:.1rem 0 .5rem; }
.game-tag { color:#d9f36a; background:#2a382e; display:inline-block; font-size:.69rem; border-radius:4px; padding:4px 9px; margin:0 4px 4px 0; }
.game-card-rule { border-top:1px solid var(--rule); margin:.25rem 0 .35rem; }
.match-note { font-size:.76rem; color:#d9f36a; margin:.1rem 0 .55rem; }
div[data-testid="stImage"] img { border-radius:7px; object-fit:cover; aspect-ratio:460/215; outline:1px solid #354238; }
.stButton>button { background:#1b2720; color:var(--ink); border:1px solid #46584a; border-radius:5px; font-weight:600; font-size:.82rem; padding:.5rem .7rem; }
.stButton>button:hover { background:var(--orange); border-color:var(--orange); color:#11180f; }
.stTextInput input,.stSelectbox [data-baseweb="select"]>div,.stSelectbox [role="combobox"] { background:#19221c!important; border-color:#3a483e!important; border-radius:5px!important; color:var(--ink)!important; }
div[data-testid="stAlert"] { border-radius:6px; }
.empty-cover { display:flex; align-items:center; justify-content:center; aspect-ratio:460/215; background:linear-gradient(135deg,#26382d,#19231d); border-radius:5px; color:var(--orange); font:400 1.4rem 'DM Serif Display',Georgia,serif; }
.footer { border-top:1px solid var(--rule); color:#89968c; margin-top:46px; padding-top:17px; font-size:.78rem; }
.login-shell { margin:7vh auto 0; max-width:1030px; border:1px solid #3a493e; background:#141c17; border-radius:14px; overflow:hidden; box-shadow:0 32px 100px #0008; }
.login-art { min-height:550px; height:100%; padding:56px 46px; display:flex; flex-direction:column; justify-content:space-between; background:radial-gradient(ellipse at 75% 36%,#60744755,transparent 37%),linear-gradient(150deg,#293c30,#152019 72%); border-right:1px solid #3a493e; }
.login-art h1 { color:#f1f4e9; font:400 clamp(2.7rem,5vw,4.6rem)/.98 'DM Serif Display',Georgia,serif; margin:18px 0; }
.login-art p { color:#bfccbf; line-height:1.7; max-width:370px; }
.login-ornament { font-size:5rem; line-height:1; filter:drop-shadow(0 10px 25px #d9f36a33); }
.login-panel { padding:52px 42px; }
.login-panel h2 { font:400 2.1rem 'DM Serif Display',Georgia,serif; margin:.4rem 0; }
.login-footnote { color:#92a095; font-size:.76rem; line-height:1.6; }
.login-panel input { background:#1b261e!important; color:#f2f4eb!important; }
.stVerticalBlockBorderWrapper { background:#141c17!important; border-color:#3a493e!important; border-radius:12px!important; padding:24px!important; }
label, [data-testid="stMarkdownContainer"] p { color:inherit; }
@media(max-width:750px) { .hero { padding:30px 25px; display:block; } .hero-stat { border:0; padding:17px 0 0; } .mast-note { display:none; } }
</style>
""", unsafe_allow_html=True)

if not st.session_state.get("sidequest_signed_in", False):
    st.markdown('<div class="masthead"><div class="wordmark">SIDE<span>QUEST</span></div><div class="mast-note">Your next favorite is out there.</div></div>', unsafe_allow_html=True)
    art, panel = st.columns([1.1, .9], gap="small")
    with art:
        st.markdown('''<div class="login-shell"><div class="login-art">
        <div><div class="eyebrow">Your next quest starts here</div><h1>Less scrolling.<br>More playing.</h1>
        <p>Find a world worth getting lost in. Pick a mood, discover your next game, and get back to the good part.</p></div>
        <div class="login-ornament">✳</div></div></div>''', unsafe_allow_html=True)
    with panel:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Welcome, player</div><div class="section-title">Sign in to Sidequest</div><p class="subtle">Jump back into your game discovery.</p>', unsafe_allow_html=True)
            with st.form("sidequest_login"):
                email = st.text_input("Email address", placeholder="you@example.com")
                password = st.text_input("Password", placeholder="Your password", type="password")
                submitted = st.form_submit_button("Sign in  →", use_container_width=True)
        if submitted:
            if email.strip() and password:
                st.session_state["sidequest_signed_in"] = True
                st.session_state["sidequest_email"] = email.strip()
                st.rerun()
            st.error("Enter an email address and password to continue.")
        if st.button("Browse as guest", use_container_width=True):
            st.session_state["sidequest_signed_in"] = True
            st.session_state["sidequest_email"] = "Guest"
            st.rerun()
        st.markdown('<div class="login-footnote">Preview sign-in only. Accounts and passwords are not saved or checked yet.</div>', unsafe_allow_html=True)
    st.stop()


def _list_value(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    if text[:1] == "[":
        values = re.findall(r"['\"]([^'\"]+)['\"]", text)
        if values:
            return values
    return [part.strip() for part in re.split(r"[,;]", text) if part.strip()]


@st.cache_resource(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return demo_games()
    available = set(pd.read_csv(DATA_PATH, nrows=0).columns)
    usecols = [column for column in DATA_COLUMNS if column in available]
    return pd.read_csv(DATA_PATH, usecols=usecols, low_memory=False)


@st.cache_resource(show_spinner=False)
def build_model() -> GameRecommender:
    source = load_catalog()
    if len(source) > 25000:
        popularity = pd.to_numeric(source["recommendations"], errors="coerce").fillna(0)
        head = source.loc[popularity.nlargest(12000).index]
        rest = source.drop(index=head.index)
        tail = rest.sample(n=min(13000, len(rest)), random_state=21)
        training = pd.concat([head, tail]).drop_duplicates("name")
    else:
        training = source
    return GameRecommender(training, catalog=source)


with st.spinner("Getting the Steam shelves ready… first visit can take a little while."):
    catalog = load_catalog()
    model = build_model()
is_demo = not DATA_PATH.exists()

for column in ("name", "genres", "short_description", "header_image", "price", "recommendations", "positive", "negative", "release_date"):
    if column not in catalog.columns:
        catalog[column] = "" if column not in {"price", "recommendations", "positive", "negative"} else 0

catalog = catalog.copy()
catalog["_genre_values"] = catalog["genres"].map(_list_value)
catalog["_popularity"] = pd.to_numeric(catalog["recommendations"], errors="coerce").fillna(0)
catalog["_positive"] = pd.to_numeric(catalog["positive"], errors="coerce").fillna(0)
catalog["_negative"] = pd.to_numeric(catalog["negative"], errors="coerce").fillna(0)
catalog["_rating"] = catalog["_positive"] / (catalog["_positive"] + catalog["_negative"] + 1)

mast_col, user_col = st.columns([6, 2], vertical_alignment="center")
with mast_col:
    st.markdown('<div class="masthead"><div class="wordmark">SIDE<span>QUEST</span></div><div class="mast-note">A thoughtful place to find your next game</div></div>', unsafe_allow_html=True)
with user_col:
    account_col, logout_col = st.columns([3, 2], vertical_alignment="center")
    with account_col:
        st.caption(st.session_state.get("sidequest_email", "Player"))
    with logout_col:
        if st.button("Sign out", key="sidequest_signout"):
            st.session_state.pop("sidequest_signed_in", None)
            st.session_state.pop("sidequest_email", None)
            st.rerun()
st.markdown(f'''
<div class="hero">
  <div><div class="hero-kicker">A good game changes the evening</div>
  <h1>Find something<br>you’ll want to play.</h1>
  <p>Pick a genre, wander through the shelves, and let the recommendation engine find a few games with the same spark.</p></div>
  <div class="hero-stat"><strong>{len(catalog):,}</strong><span>games on the shelf<br>from the Steam catalog</span></div>
</div>
''', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Browse the shelves</div><div class="section-title">What are you in the mood for?</div>', unsafe_allow_html=True)
all_genres = sorted({genre for genres in catalog["_genre_values"] for genre in genres}, key=str.casefold)
genre_order = ["Action", "Adventure", "RPG", "Indie", "Strategy", "Simulation", "Casual", "Racing", "Sports", "Horror", "Puzzle", "Free To Play"]
genres = [genre for genre in genre_order if genre in all_genres]
genres.extend(genre for genre in all_genres if genre not in genres)
filter_a, filter_b, filter_c = st.columns([1.2, 1.5, 1])
with filter_a:
    selected_genre = st.selectbox("Genre", ["All genres", *genres], label_visibility="collapsed")
with filter_b:
    search = st.text_input("Search", placeholder="Search for a game…", label_visibility="collapsed")
with filter_c:
    sort_by = st.selectbox("Sort", ["For you", "Most popular", "Best reviewed", "Recently released"], label_visibility="collapsed")

category_picks = selected_genre != "All genres" and not search.strip() and sort_by == "For you"
visible = catalog
if selected_genre != "All genres":
    visible = visible[visible["_genre_values"].map(lambda values: selected_genre in values)]
if search.strip():
    visible = visible[visible["name"].astype(str).str.contains(re.escape(search.strip()), case=False, na=False)]
if category_picks:
    visible = model.recommend_by_genre(selected_genre, limit=12)
elif sort_by == "Best reviewed":
    visible = visible.sort_values(["_rating", "_positive"], ascending=False)
elif sort_by == "Recently released":
    visible = visible.assign(_release=pd.to_datetime(visible["release_date"], errors="coerce")).sort_values("_release", ascending=False)
else:
    visible = visible.sort_values("_popularity", ascending=False)

result_count = len(visible)
section_name = f"Our {selected_genre} picks" if category_picks else (selected_genre if selected_genre != "All genres" else "Popular right now")
section_note = "Matched by the recommendation model from game descriptions, genres and player tags." if category_picks else f"{result_count:,} games to browse · selected from the full Steam catalog"
st.markdown(f'<div class="result-head"><div><div class="section-title">{escape(section_name)}</div><div class="subtle">{section_note}</div></div><div class="subtle">Showing up to 12</div></div>', unsafe_allow_html=True)


def game_card(game: pd.Series, key: str, match: int | None = None) -> None:
    title = str(game.get("name", game.get("_title", "Untitled game")))
    description = str(game.get("short_description", "")).strip()
    if not description or description.lower() == "nan":
        description = "A game from the Steam catalog. Open its Steam page to learn more."
    description = description[:210].rsplit(" ", 1)[0] + ("…" if len(description) > 210 else "")
    image_url = str(game.get("header_image", "")).strip()
    genre_labels = _list_value(game.get("genres", ""))[:3]
    release = str(game.get("release_date", "")).strip()
    price_value = pd.to_numeric(pd.Series([game.get("price", 0)]), errors="coerce").fillna(0).iloc[0]
    price = "Free to play" if price_value == 0 else f"${price_value:,.2f}"
    with st.container():
        if image_url.startswith("https://"):
            st.image(image_url, caption=None, use_container_width=True)
        else:
            initials = "".join(word[0] for word in title.split()[:2]).upper()
            st.markdown(f'<div class="empty-cover">{escape(initials)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="game-title">{escape(title)}</div><div class="game-meta">{escape(release[:4] if release else "Steam")}&nbsp;&nbsp;·&nbsp;&nbsp;{escape(price)}</div><div class="game-description">{escape(description)}</div>{"".join(f"<span class=\"game-tag\">{escape(label)}</span>" for label in genre_labels)}<div class="game-card-rule"></div>', unsafe_allow_html=True)
        if match is not None:
            st.markdown(f'<div class="match-note">A {match}% profile match</div>', unsafe_allow_html=True)
        if st.button("Find games like this  ↗", key=f"similar-{key}", use_container_width=True):
            st.session_state["sidequest_pick"] = title
            st.rerun()


if visible.empty:
    st.info("No games found for that search. Try another title or genre.")
else:
    cards = st.columns(3, gap="large")
    for index, (row_index, game) in enumerate(visible.head(12).iterrows()):
        appid = str(game.get("appid", row_index))
        with cards[index % 3]:
            game_card(game, appid)

picked = st.session_state.get("sidequest_pick")
if picked:
    recommendations = model.recommend(picked, limit=6)
    st.markdown(f'<div class="eyebrow" style="margin-top:2rem">A few in the same spirit</div><div class="section-title">If you like {escape(picked)}, try these</div><div class="subtle">Picked by matching game descriptions, genres and player tags across the catalog.</div>', unsafe_allow_html=True)
    if st.button("Clear these picks", key="clear-sidequest-picks"):
        st.session_state.pop("sidequest_pick", None)
        st.rerun()
    if recommendations.empty:
        st.info("That title isn’t in the current recommendation index yet. Try another game from the shelves above.")
    else:
        rec_cards = st.columns(3, gap="large")
        for index, (_, game) in enumerate(recommendations.iterrows()):
            appid = str(game.get("appid", game.name if hasattr(game, "name") else index))
            with rec_cards[index % 3]:
                game_card(game, f"recommendation-{appid}", int(game["_match"]))

if is_demo:
    st.info("The Steam CSV isn’t beside this app, so this page is showing a small preview. Add `dataset/games.csv` to browse the full catalog.")

st.markdown('<div class="footer">SIDEQUEST &nbsp;·&nbsp; Recommendations are based on similarities in game descriptions, genres and community tags.</div>', unsafe_allow_html=True)
