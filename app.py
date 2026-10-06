from __future__ import annotations

import ast
import re
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.recommender import GameRecommender, demo_games


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "dataset" / "games.csv.gz"
if not DATA_PATH.exists():
    DATA_PATH = ROOT / "dataset" / "games.csv"
DATA_COLUMNS = [
    "appid", "name", "release_date", "price", "recommendations", "positive",
    "negative", "header_image", "categories", "genres", "tags", "short_description",
]

st.set_page_config(page_title="Sidequest — Find your next game", page_icon="🎮", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@500;700;800&display=swap');
:root { --paper:#000000; --ink:#f0fff7; --forest:#022c22; --soft:#91aa9c; --rule:#164332; --orange:#00ff9d; --card:#08100c; }
html, body, [class*="css"] { font-family:'Plus Jakarta Sans',sans-serif; }
.stApp { background:radial-gradient(ellipse at 76% 0%,rgba(16,185,129,.12),transparent 38%),linear-gradient(rgba(16,185,129,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(16,185,129,.035) 1px,transparent 1px),#000; background-size:auto,60px 60px,60px 60px,auto; color:var(--ink); }
header[data-testid="stHeader"] { background:rgba(0,0,0,.92); }
.block-container { max-width:1260px; padding-top:3.2rem; padding-bottom:4rem; }
#MainMenu, footer { visibility:hidden; }
.masthead { display:flex; justify-content:space-between; align-items:center; padding:12px 20px; border:1px solid rgba(0,255,157,.25); border-radius:999px; background:rgba(8,16,12,.85); backdrop-filter:blur(16px); box-shadow:0 8px 38px #0008; }
.wordmark { color:#f2f4eb!important; opacity:1!important; font-size:1.04rem; font-weight:700; letter-spacing:.1em; }
.wordmark span { color:var(--orange); }
.mast-note { color:#b8c4ba!important; opacity:1!important; font-size:.82rem; }
.hero { display:flex; justify-content:space-between; align-items:flex-end; gap:30px; background:linear-gradient(120deg,#022c22,#08100c 70%); color:#f0fff7; border:1px solid #0b6b4a; border-radius:10px; padding:42px 48px; margin:22px 0 29px; box-shadow:0 20px 60px #0003; }
.hero-kicker { color:var(--orange); font-size:.71rem; font-weight:700; text-transform:uppercase; letter-spacing:.15em; }
.hero h1 { font:800 clamp(2.35rem,5vw,4rem)/1.03 'Syne',sans-serif; letter-spacing:-.035em; margin:.62rem 0 .85rem; max-width:670px; }
.hero p { color:#c0cbc1; font-size:.98rem; line-height:1.65; max-width:570px; margin:0; }
.hero-stat { flex:0 0 180px; border-left:1px solid rgba(255,255,255,.22); padding-left:24px; margin-bottom:4px; }
.hero-stat strong { display:block; font:800 2.1rem 'Syne',sans-serif; color:var(--orange); }
.hero-stat span { color:#c0cbc1; font-size:.78rem; line-height:1.5; }
.eyebrow { color:var(--orange); text-transform:uppercase; letter-spacing:.14em; font-size:.68rem; font-weight:700; }
.section-title { font:700 1.8rem 'Syne',sans-serif; margin:.15rem 0 .2rem; color:var(--ink); }
.subtle { color:var(--soft); font-size:.88rem; line-height:1.55; }
.result-head { display:flex; justify-content:space-between; align-items:end; margin:24px 0 12px; }
.game-title { font:700 1.02rem 'Plus Jakarta Sans',sans-serif; line-height:1.3; color:var(--ink); margin:.7rem 0 .35rem; min-height:2.6em; }
.game-description { font-size:.82rem; line-height:1.55; color:#b3beb5; min-height:3.85em; margin:.2rem 0 .55rem; }
.game-meta { color:#a3afa5; font-size:.74rem; letter-spacing:.015em; padding:.1rem 0 .5rem; }
.game-tag { color:#00ff9d; background:#022c22; border:1px solid #0e5b3f; display:inline-block; font-size:.69rem; border-radius:4px; padding:4px 9px; margin:0 4px 4px 0; }
.game-card-rule { border-top:1px solid var(--rule); margin:.25rem 0 .35rem; }
.match-note { font-size:.76rem; color:#00ff9d; margin:.1rem 0 .55rem; }
div[data-testid="stImage"] img { border-radius:7px; object-fit:cover; aspect-ratio:460/215; outline:1px solid #354238; }
.stButton>button { background:rgba(8,16,12,.88); color:var(--ink); border:1px solid rgba(0,255,157,.35); border-radius:999px; font-weight:700; font-size:.82rem; padding:.5rem .9rem; }
.stButton>button:hover { background:var(--orange); border-color:var(--orange); color:#00180e; box-shadow:0 0 20px #00ff9d44; }
.stTextInput input,.stSelectbox [data-baseweb="select"]>div,.stSelectbox [role="combobox"] { background:#08100c!important; border-color:#164332!important; border-radius:8px!important; color:var(--ink)!important; }
div[data-testid="stAlert"] { border-radius:6px; }
.empty-cover { display:flex; align-items:center; justify-content:center; aspect-ratio:460/215; background:linear-gradient(135deg,#022c22,#08100c); border-radius:5px; color:var(--orange); font:700 1.4rem 'Syne',sans-serif; }
.footer { border-top:1px solid var(--rule); color:#89968c; margin-top:46px; padding-top:17px; font-size:.78rem; }
.login-shell { margin:7vh auto 0; max-width:1030px; border:1px solid #164332; background:rgba(8,16,12,.88); border-radius:14px; overflow:hidden; box-shadow:0 32px 100px #0008; }
.login-art { min-height:550px; height:100%; padding:56px 46px; display:flex; flex-direction:column; justify-content:space-between; background:radial-gradient(ellipse at 75% 36%,#00ff9d25,transparent 37%),linear-gradient(150deg,#022c22,#050b08 72%); border-right:1px solid #164332; }
.login-art h1 { color:#f0fff7; font:800 clamp(2.7rem,5vw,4.6rem)/.98 'Syne',sans-serif; margin:18px 0; }
.login-art p { color:#bfccbf; line-height:1.7; max-width:370px; }
.login-ornament { font-size:5rem; line-height:1; filter:drop-shadow(0 10px 25px #00ff9d55); color:#00ff9d; }
.login-panel { padding:52px 42px; }
.login-panel h2 { font:700 2.1rem 'Syne',sans-serif; margin:.4rem 0; }
.login-footnote { color:#92a095; font-size:.76rem; line-height:1.6; }
.login-panel input { background:#08100c!important; color:#f0fff7!important; }
.stVerticalBlockBorderWrapper { background:rgba(8,16,12,.88)!important; border-color:#164332!important; border-radius:12px!important; padding:24px!important; }
.hero-actions { display:flex; gap:12px; flex-wrap:wrap; margin:12px 0 28px; }
.hero-cta { display:inline-flex; align-items:center; gap:18px; padding:9px 10px 9px 18px; background:rgba(8,16,12,.85); border:1px solid rgba(0,255,157,.25); border-radius:999px; color:#effff7!important; font-size:.82rem; font-weight:700; text-decoration:none!important; backdrop-filter:blur(12px); }
.hero-cta span { display:grid; place-items:center; width:30px; height:30px; border-radius:50%; color:#00180e; background:#f0fff7; transition:transform .2s,background .2s; }
.hero-cta:hover { border-color:#00ff9d; box-shadow:0 0 22px #00ff9d22; }
.hero-cta:hover span { transform:translateX(2px); background:#00ff9d; }
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
    if len(source) > 8000:
        popularity = pd.to_numeric(source["recommendations"], errors="coerce").fillna(0)
        head = source.loc[popularity.nlargest(4000).index]
        rest = source.drop(index=head.index)
        tail = rest.sample(n=min(4000, len(rest)), random_state=21)
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
hero_candidates = catalog.assign(
    _cover=catalog["header_image"].astype(str).str.strip()
).loc[lambda frame: frame["_cover"].str.startswith("https://")].sort_values("_popularity", ascending=False)
hero_games = hero_candidates.drop_duplicates("_cover").head(2)
hero_rows = list(hero_games.iterrows())
hero_base_url = escape(str(hero_rows[0][1]["_cover"]), quote=True) if hero_rows else ""
hero_reveal_url = escape(str(hero_rows[1][1]["_cover"]), quote=True) if len(hero_rows) > 1 else hero_base_url
hero_base_title = escape(str(hero_rows[0][1]["name"])) if hero_rows else "The Steam catalog"
hero_reveal_title = escape(str(hero_rows[1][1]["name"])) if len(hero_rows) > 1 else hero_base_title

hero_html = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@500;700;800&display=swap');
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#000;font-family:'Plus Jakarta Sans',sans-serif;color:#f0fff7}
.stage{--spot-x:76%;--spot-y:52%;position:relative;width:100%;height:465px;overflow:hidden;border:1px solid rgba(0,255,157,.32);border-radius:16px;background:#000;isolation:isolate;box-shadow:0 24px 72px #0009,0 0 45px #00ff9d14}
.base,.reveal{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center;pointer-events:none}
.base{filter:brightness(.62) saturate(.85);transform:scale(1.015)}
.reveal-wrap{position:absolute;inset:0;overflow:hidden;mask-image:radial-gradient(circle 175px at var(--spot-x) var(--spot-y),#000 0%,rgba(0,0,0,.95) 42%,rgba(0,0,0,.55) 66%,transparent 100%);-webkit-mask-image:radial-gradient(circle 175px at var(--spot-x) var(--spot-y),#000 0%,rgba(0,0,0,.95) 42%,rgba(0,0,0,.55) 66%,transparent 100%);will-change:mask-image,-webkit-mask-image}
.reveal{filter:brightness(.8) saturate(1.3) hue-rotate(16deg)}
.vignette{position:absolute;inset:0;background:linear-gradient(90deg,rgba(0,0,0,.96) 0%,rgba(0,0,0,.82) 33%,rgba(0,0,0,.35) 72%,rgba(0,0,0,.1) 100%),linear-gradient(0deg,rgba(0,0,0,.55),transparent 40%);pointer-events:none}
.cyber-grid{position:absolute;inset:0;opacity:.44;background-image:linear-gradient(rgba(16,185,129,.075) 1px,transparent 1px),linear-gradient(90deg,rgba(16,185,129,.075) 1px,transparent 1px);background-size:48px 48px;mask-image:linear-gradient(90deg,#000 0%,transparent 86%);pointer-events:none}
canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;opacity:.76}
.halo{position:absolute;left:76%;top:52%;width:350px;height:350px;transform:translate(-50%,-50%);border:1px solid rgba(0,255,157,.45);border-radius:50%;box-shadow:0 0 22px rgba(0,255,157,.12),inset 0 0 22px rgba(0,255,157,.08);pointer-events:none;will-change:left,top}
.halo:before,.halo:after{content:'';position:absolute;inset:20px;border:1px solid rgba(0,255,157,.2);border-radius:50%}.halo:after{inset:-13px;border-color:rgba(0,255,157,.14)}
.content{position:absolute;inset:0;z-index:3;padding:42px 52px;display:flex;flex-direction:column;align-items:flex-start;justify-content:center;pointer-events:none}
.pill{display:inline-flex;align-items:center;gap:9px;border:1px solid rgba(0,255,157,.38);border-radius:999px;background:rgba(0,10,6,.78);backdrop-filter:blur(16px);padding:8px 14px;color:#00ff9d;text-transform:uppercase;font:700 10px 'Plus Jakarta Sans',sans-serif;letter-spacing:.16em}
.dot{width:7px;height:7px;border-radius:50%;background:#00ff9d;box-shadow:0 0 12px #00ff9d;animation:pulse 1.8s infinite}
h1{max-width:690px;margin:22px 0 14px;font:800 clamp(38px,6.2vw,76px)/.99 'Syne',sans-serif;letter-spacing:-.055em;text-shadow:0 4px 22px #000}
h1 em{color:#00ff9d;font-style:normal;text-shadow:0 0 28px #00ff9d35}
.sub{max-width:530px;margin:0;color:#d4e6db;font-size:clamp(13px,1.5vw,17px);line-height:1.7;font-weight:400}
.stats{display:flex;align-items:center;gap:12px;margin-top:28px;padding:11px 15px;border:1px solid rgba(0,255,157,.2);border-radius:10px;background:rgba(0,0,0,.58);backdrop-filter:blur(12px)}
.stats strong{font:800 23px 'Syne',sans-serif;color:#00ff9d}.stats span{font-size:10px;line-height:1.4;color:#b7cec0;letter-spacing:.1em;text-transform:uppercase}
.featured{position:absolute;z-index:3;right:25px;bottom:24px;padding:10px 14px;border:1px solid rgba(0,255,157,.25);border-radius:8px;background:rgba(0,0,0,.62);backdrop-filter:blur(14px);font-size:10px;color:#b7cec0;letter-spacing:.04em;max-width:40%;text-align:right}
.featured b{display:block;margin-top:3px;color:#f0fff7;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.scan{position:absolute;z-index:3;left:20px;bottom:22px;color:rgba(0,255,157,.78);font:600 9px ui-monospace,monospace;letter-spacing:.15em;text-transform:uppercase}
@keyframes pulse{50%{opacity:.4;transform:scale(.8)}}
@media(max-width:700px){.stage{height:490px;border-radius:12px}.content{padding:30px 24px;justify-content:center}.featured{right:12px;bottom:14px;max-width:46%;font-size:9px}.featured b{font-size:10px}.scan{left:14px;bottom:16px}.halo{width:270px;height:270px}}
@media(prefers-reduced-motion:reduce){.dot{animation:none}}
</style></head><body>
<section class="stage" id="stage" aria-label="Sidequest game discovery hero">
 <img class="base" id="base" src="__BASE_URL__" alt="" onerror="this.style.display='none'">
 <div class="reveal-wrap" id="revealWrap"><img class="reveal" src="__REVEAL_URL__" alt="" onerror="this.style.display='none'"></div>
 <div class="vignette"></div><div class="cyber-grid"></div><canvas id="particles"></canvas><div class="halo" id="halo"></div>
 <div class="content"><div class="pill"><i class="dot"></i> SIDEQUEST · DISCOVERY SYSTEM ONLINE</div>
 <h1>FIND YOUR<br>NEXT <em>WORLD.</em></h1>
 <p class="sub">A galaxy of games, one great match away. Follow the signal through the Steam shelves and see what pulls you in.</p>
 <div class="stats"><strong>__GAME_COUNT__</strong><span>games in the Steam<br>discovery system</span></div></div>
 <div class="scan">◉ LIVE SCAN · MOVE TO REVEAL</div>
 <div class="featured">SIGNAL LOCKED<b>__FEATURED_TITLE__</b></div>
</section>
<script>
(()=>{const stage=document.getElementById('stage'), wrap=document.getElementById('revealWrap'), halo=document.getElementById('halo'), canvas=document.getElementById('particles'), ctx=canvas.getContext('2d');
let rect=stage.getBoundingClientRect(), targetX=rect.width*.76,targetY=rect.height*.52,x=targetX,y=targetY,hovered=false,w=0,h=0,dpr=1;
const resize=()=>{rect=stage.getBoundingClientRect();dpr=Math.min(window.devicePixelRatio||1,2);w=rect.width;h=rect.height;canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0)};
resize();window.addEventListener('resize',resize,{passive:true});
const point=e=>{rect=stage.getBoundingClientRect();targetX=e.clientX-rect.left;targetY=e.clientY-rect.top;hovered=true};
stage.addEventListener('pointermove',point,{passive:true});stage.addEventListener('pointerleave',()=>{hovered=false},{passive:true});
stage.addEventListener('touchmove',e=>{if(e.touches.length){const t=e.touches[0];rect=stage.getBoundingClientRect();targetX=t.clientX-rect.left;targetY=t.clientY-rect.top;hovered=true}},{passive:true});
const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;let particles=Array.from({length:42},()=>({x:Math.random()*w,y:Math.random()*h,r:Math.random()*1.5+.35,vx:(Math.random()-.5)*.24,vy:-Math.random()*.22-.04,a:Math.random()*.5+.15}));
function frame(now){if(!hovered&&!reduce){targetX=w*.76+Math.cos(now*.00045)*w*.07;targetY=h*.52+Math.sin(now*.0006)*h*.08}const ease=hovered ? 0.17 : 0.045;x+=(targetX-x)*ease;y+=(targetY-y)*ease;stage.style.setProperty('--spot-x',x+'px');stage.style.setProperty('--spot-y',y+'px');halo.style.left=x+'px';halo.style.top=y+'px';
ctx.clearRect(0,0,w,h);if(!reduce){for(const p of particles){p.x+=p.vx;p.y+=p.vy;if(p.y<-4){p.y=h+4;p.x=Math.random()*w}if(p.x<0||p.x>w)p.vx*=-1;ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle='rgba(0,255,157,'+p.a+')';ctx.shadowColor='#00ff9d';ctx.shadowBlur=9;ctx.fill()}ctx.shadowBlur=0}requestAnimationFrame(frame)}requestAnimationFrame(frame);
})();
</script></body></html>'''
hero_html = (
    hero_html.replace("__BASE_URL__", hero_base_url)
    .replace("__REVEAL_URL__", hero_reveal_url)
    .replace("__GAME_COUNT__", f"{len(catalog):,}")
    .replace("__FEATURED_TITLE__", hero_reveal_title)
)
components.html(hero_html, height=485, scrolling=False)
st.markdown('''<div class="hero-actions"><a class="hero-cta" href="#browse-games">Explore the shelves <span>↗</span></a><a class="hero-cta" href="#game-picks">See today’s picks <span>⌄</span></a></div>''', unsafe_allow_html=True)

st.markdown('<div id="browse-games" class="eyebrow">Browse the shelves</div><div class="section-title">What are you in the mood for?</div>', unsafe_allow_html=True)
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
st.markdown(f'<div id="game-picks" class="result-head"><div><div class="section-title">{escape(section_name)}</div><div class="subtle">{section_note}</div></div><div class="subtle">Showing up to 12</div></div>', unsafe_allow_html=True)


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
