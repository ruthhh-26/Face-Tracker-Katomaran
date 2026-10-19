"""Dashboard for the face tracker: live view, run control, unique visitors, entry/exit events, images, events.log.

Run:  .venv\\Scripts\\streamlit run dashboard/app.py
Reads the same PostgreSQL DB and logs/ folder that main.py writes to, so it works while tracking is running.
Styling: Streamlit's theme (.streamlit/config.toml) + a small CSS block below. Charts use Altair (ships with Streamlit).
"""
import io
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import psycopg2
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
from src.config import load_config  # noqa: E402  (needs ROOT on sys.path)
RUN_CFG = load_config(str(ROOT / "config.json"))      # config.json merged with the defaults, for the info pages

ACCENT, ACCENT2, GOOD, WARN, BAD, MUTED = "#7C5CFF", "#22D3EE", "#34D399", "#FBBF24", "#F87171", "#8B95AD"

st.set_page_config(page_title="Face Tracker", page_icon="🎯", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#0B0E17;--panel:#121829;--panel2:#171E33;--line:rgba(255,255,255,.08);--text:#E8ECF4;--muted:#8B95AD;
      --accent:#7C5CFF;--accent2:#22D3EE;--good:#34D399;--warn:#FBBF24;--bad:#F87171;}
html, body, .stApp, [class*="css"]{font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;}
.stApp{background:
  radial-gradient(1100px 560px at 8% -12%, rgba(124,92,255,.20), transparent 60%),
  radial-gradient(900px 480px at 100% 0%, rgba(34,211,238,.12), transparent 55%), var(--bg);}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1400px;}
#MainMenu, footer{visibility:hidden;}

/* sidebar */
section[data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--line);}
.brand{display:flex;align-items:center;gap:10px;font-weight:800;font-size:1.15rem;margin:.2rem 0 .4rem;}
.brand .logo{width:34px;height:34px;border-radius:10px;display:grid;place-items:center;font-size:1.1rem;
  background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 6px 20px rgba(124,92,255,.45);}
.side-h{font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:1.1rem 0 .3rem;font-weight:600;}

/* hero */
.hero{display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;flex-wrap:wrap;margin:.2rem 0 1.2rem;}
.eyebrow{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--accent2);font-weight:700;}
.hero h1{margin:.15rem 0 .25rem;font-size:2.5rem;font-weight:800;letter-spacing:-.02em;line-height:1.05;padding:0;
  background:linear-gradient(90deg,#fff 0%,#C9BEFF 55%,var(--accent2) 100%);-webkit-background-clip:text;
  background-clip:text;-webkit-text-fill-color:transparent;}
.hero p{margin:0;color:var(--muted);font-size:.95rem;max-width:640px;}

/* pills + badges */
.pill{display:inline-flex;align-items:center;gap:8px;padding:6px 13px;border-radius:999px;font-size:.8rem;font-weight:600;
  border:1px solid var(--line);background:rgba(255,255,255,.04);color:var(--text);white-space:nowrap;margin:0 6px 6px 0;}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;}
.dot.pulse{animation:pulse 1.6s ease-out infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.7)}100%{box-shadow:0 0 0 10px rgba(52,211,153,0)}}
.chip{display:inline-block;padding:3px 10px;border-radius:8px;background:rgba(124,92,255,.14);color:#CFC6FF;font-size:.74rem;
  font-weight:600;margin:0 6px 6px 0;border:1px solid rgba(124,92,255,.25);}
.badge{display:inline-block;padding:2px 9px;border-radius:7px;font-size:.72rem;font-weight:700;margin-right:6px;}
.badge.f{background:rgba(124,92,255,.2);color:#CFC6FF;}
.badge.p{background:rgba(251,191,36,.16);color:#FCD36B;}
.badge.entry{background:rgba(52,211,153,.16);color:#6EE7B7;}
.badge.exit{background:rgba(248,113,113,.16);color:#FCA5A5;}

/* KPI cards */
.kpi{position:relative;overflow:hidden;background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);
  border-radius:16px;padding:16px 18px 14px;height:100%;box-sizing:border-box;}
.kpi::before{content:"";position:absolute;left:0;top:0;right:0;height:3px;background:var(--a,var(--accent));opacity:.9;}
.kpi .lbl{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:600;}
.kpi .val{font-size:2.15rem;font-weight:800;line-height:1.15;margin:.2rem 0 .1rem;letter-spacing:-.02em;white-space:nowrap;}
.kpi .val.sm{font-size:1.2rem;line-height:1.85;margin:.4rem 0 .2rem;}
.kpi .sub{font-size:.76rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}

/* section headings, empty state, feed */
.sec{font-size:1.02rem;font-weight:700;margin:1.2rem 0 .5rem;display:flex;align-items:center;gap:8px;}
.sec small{color:var(--muted);font-weight:500;font-size:.8rem;}
.empty{border:1px dashed rgba(255,255,255,.14);border-radius:16px;padding:2.2rem 1rem;text-align:center;color:var(--muted);
  background:rgba(255,255,255,.02);}
.feed-item{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:9px 12px;border-radius:12px;
  background:rgba(255,255,255,.035);border:1px solid var(--line);margin-bottom:8px;font-size:.84rem;}
.feed-item .t{color:var(--muted);font-variant-numeric:tabular-nums;font-size:.78rem;}
.cap{font-size:.78rem;color:var(--muted);margin-top:.3rem;}

/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:6px;background:var(--panel);padding:6px;border-radius:14px;border:1px solid var(--line);width:fit-content;}
.stTabs [data-baseweb="tab"]{border-radius:10px;padding:8px 18px;height:auto;color:var(--muted);font-weight:600;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,var(--accent),#5B8DEF);color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none;}

/* cards made from st.container(border=True) */
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px;border:1px solid var(--line);background:rgba(255,255,255,.025);}

/* buttons, inputs, media */
.stButton > button{border-radius:12px;font-weight:700;padding:.6rem 1.2rem;border:1px solid var(--line);transition:all .15s ease;}
.stButton > button:hover{transform:translateY(-1px);border-color:rgba(124,92,255,.6);}
.stButton > button[kind="primary"]{background:linear-gradient(135deg,var(--accent),#5B8DEF);border:none;
  box-shadow:0 8px 22px rgba(124,92,255,.35);}
.stDownloadButton > button{border-radius:12px;font-weight:600;}
div[data-testid="stImage"] img{border-radius:14px;border:1px solid var(--line);box-shadow:0 14px 40px rgba(0,0,0,.45);}
.st-key-gallery div[data-testid="stImage"] img{aspect-ratio:3/4;object-fit:cover;object-position:top;width:100%;box-shadow:none;}
div[data-testid="stFileUploader"] section{border-radius:14px;border:1px dashed rgba(124,92,255,.5);background:rgba(124,92,255,.05);}
.stProgress > div > div > div{background:linear-gradient(90deg,var(--accent),var(--accent2));}
div[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;border:1px solid var(--line);}
code, pre{font-size:.78rem !important;}

/* "How it works" page */
.flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:.4rem 0 1rem;}
.step{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:16px;padding:14px 16px;}
.step .n{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;font-weight:800;font-size:.8rem;color:#fff;
  background:linear-gradient(135deg,var(--accent),var(--accent2));margin-bottom:8px;}
.step h4{margin:0 0 4px;font-size:.95rem;font-weight:700;padding:0;}
.step p{margin:0;color:var(--muted);font-size:.8rem;line-height:1.45;}
.rule{display:flex;gap:12px;align-items:flex-start;padding:12px 14px;border-radius:14px;background:rgba(255,255,255,.03);
  border:1px solid var(--line);margin-bottom:10px;}
.rule .k{min-width:86px;text-align:center;font-weight:800;font-size:.95rem;padding:6px 8px;border-radius:10px;
  background:rgba(124,92,255,.16);color:#CFC6FF;}
.rule b{display:block;font-size:.88rem;margin-bottom:2px;}
.rule span.d{color:var(--muted);font-size:.8rem;line-height:1.45;}
.stack{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;}
.tech{padding:12px 14px;border-radius:14px;border:1px solid var(--line);background:rgba(255,255,255,.03);}
.tech b{font-size:.9rem;}
.tech span{display:block;color:var(--muted);font-size:.78rem;margin-top:2px;}

/* narrow screens */
@media (max-width: 900px){
  .block-container{padding:1rem .8rem 2rem;}
  .hero h1{font-size:1.9rem;}
  .kpi .val{font-size:1.6rem;}
  .kpi .val.sm{font-size:1.05rem;}
  .stTabs [data-baseweb="tab-list"]{width:100%;overflow-x:auto;}
  div[class*="st-key-kpis"] div[data-testid="stHorizontalBlock"]{flex-wrap:wrap;gap:.7rem !important;row-gap:.7rem !important;}
  div[class*="st-key-kpis"] div[data-testid="stColumn"], div[class*="st-key-kpis"] div[data-testid="column"]{
    flex:1 1 calc(50% - .7rem) !important;min-width:calc(50% - .7rem) !important;width:auto !important;}
  .st-key-gallery div[data-testid="stHorizontalBlock"]{flex-wrap:wrap;gap:.7rem !important;row-gap:.7rem !important;}
  .st-key-gallery div[data-testid="stColumn"], .st-key-gallery div[data-testid="column"]{
    flex:1 1 calc(33% - .6rem) !important;min-width:calc(33% - .6rem) !important;width:auto !important;}
  .rule{flex-direction:column;gap:8px;}
  .rule .k{align-self:flex-start;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---- small UI helpers --------------------------------------------------------
def pill(text: str, color: str = MUTED, pulse: bool = False) -> str:
    return f'<span class="pill"><span class="dot{" pulse" if pulse else ""}" style="background:{color}"></span>{text}</span>'


def kpi_row(items, key="kpis"):
    """items: (label, value, sub-text, accent colour). The container key lets the CSS wrap the cards 2-per-row on phones."""
    with st.container(key=key):
        for col, (lbl, val, sub, acc) in zip(st.columns(len(items)), items):
            size = " sm" if len(str(val)) > 6 else ""      # long values (e.g. "405 / 1502") get a smaller font
            col.markdown(f'<div class="kpi" style="--a:{acc}"><div class="lbl">{lbl}</div><div class="val{size}">{val}</div>'
                         f'<div class="sub" title="{sub}">{sub}</div></div>', unsafe_allow_html=True)


def section(title: str, hint: str = ""):
    st.markdown(f'<div class="sec">{title} <small>{hint}</small></div>', unsafe_allow_html=True)


def empty(msg: str):
    st.markdown(f'<div class="empty">{msg}</div>', unsafe_allow_html=True)


def style_chart(ch):
    return (ch.configure_view(strokeWidth=0)
              .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="rgba(255,255,255,0.06)",
                              domainColor="rgba(255,255,255,0.12)", tickColor="rgba(255,255,255,0.12)")
              .configure_legend(labelColor="#C7CEDF", titleColor=MUTED)
              .properties(background="transparent"))


@st.cache_resource
def _conn_params():
    d = CFG["db"]
    return dict(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"])


def query(sql: str) -> pd.DataFrame:
    """Fresh short-lived connection per query: survives DB restarts and needs no cache invalidation."""
    with psycopg2.connect(**_conn_params()) as conn:
        return pd.read_sql_query(sql, conn)


def db_ok() -> bool:
    try:
        query("SELECT 1")
        return True
    except Exception:  # noqa: BLE001
        return False


def label(row_kind, vid):
    return f"{'F' if row_kind == 'face' else 'P'}{int(vid):04d}"


def badge(visitor: str) -> str:
    return f'<span class="badge {"f" if visitor.startswith("F") else "p"}">{visitor}</span>'


def _visitor_table(events: pd.DataFrame, faces: pd.DataFrame) -> pd.DataFrame:
    """One row per visitor: entries, exits, first/last event, last event type, kind (face/body)."""
    if events.empty:
        return pd.DataFrame()
    agg = events.groupby("visitor").agg(
        entries=("event_type", lambda s: int((s == "entry").sum())),
        exits=("event_type", lambda s: int((s == "exit").sum())),
        first_event=("ts", "min"), last_event=("ts", "max"), last_type=("event_type", "last"))
    return agg.join(faces.set_index("visitor")[["kind"]])


def _summary(faces: pd.DataFrame, events: pd.DataFrame) -> str:
    """Plain-text run summary for the report."""
    s = _read_status()
    ts = pd.to_datetime(events.ts)
    lines = ["FACE TRACKER REPORT", f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", "",
             f"Source: {s.get('source', 'n/a')}",
             f"Frames processed: {s.get('frame', 'n/a')} / {s.get('total', 'n/a')}",
             f"Average processing FPS: {s.get('fps', 'n/a')}",
             f"Device: {RUN_CFG['device']}   |   detection skip frames: {RUN_CFG['detection_skip_frames']}", "",
             f"Unique visitors: {len(faces)}",
             f"  identified by face (F####): {int((faces.kind == 'face').sum())}",
             f"  body-only (P####): {int((faces.kind == 'body').sum())}",
             f"Entries logged: {int((events.event_type == 'entry').sum())}",
             f"Exits logged: {int((events.event_type == 'exit').sum())}",
             f"First event: {ts.min()}", f"Last event: {ts.max()}"]
    return "\n".join(lines)


def _report_zip(summary: str, faces: pd.DataFrame, events: pd.DataFrame, with_images: bool) -> bytes:
    """summary.txt + visitors.csv + events.csv (+ the entry/exit crops) in one zip, built in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary.txt", summary)
        z.writestr("visitors.csv", _visitor_table(events, faces).to_csv())
        z.writestr("events.csv", events.to_csv(index=False))
        if with_images:
            for p in events.image_path.dropna().unique():
                f = ROOT / p
                if p and f.is_file():
                    z.write(f, arcname=str(p))
    return buf.getvalue()


# ---- run control state (before the sidebar so the header can show it) -------------
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@st.cache_resource
def _proc_holder() -> dict:
    return {"proc": None}      # survives Streamlit reruns / browser refreshes


def _running() -> bool:
    p = _proc_holder()["proc"]
    return p is not None and p.poll() is None


# ---- sidebar -----------------------------------------------------------------
st.sidebar.markdown('<div class="brand"><div class="logo">🎯</div>Face Tracker</div>', unsafe_allow_html=True)
st.sidebar.caption("Unique visitor counting with YOLO, ArcFace and body re-ID")
st.sidebar.markdown('<div class="side-h">Display</div>', unsafe_allow_html=True)
refresh = st.sidebar.slider("Auto-refresh (seconds, 0 = off)", 0, 30, 5)
gallery_n = st.sidebar.slider("Images shown", 6, 48, 18, step=6)
st.sidebar.markdown('<div class="side-h">Filters</div>', unsafe_allow_html=True)
kind_filter = st.sidebar.multiselect("Visitor type", ["face", "body"], default=["face", "body"])
type_filter = st.sidebar.multiselect("Event type", ["entry", "exit"], default=["entry", "exit"])
st.sidebar.markdown('<div class="side-h">Storage</div>', unsafe_allow_html=True)
log_dir = ROOT / st.sidebar.text_input("Log folder", CFG.get("log_dir", "logs"))
st.sidebar.markdown(
    '<div class="side-h">Legend</div>'
    '<span class="badge f">F0001</span> recognised by face<br><br>'
    '<span class="badge p">P0001</span> tracked without a visible face', unsafe_allow_html=True)

# ---- header ------------------------------------------------------------------
_state_pill = pill("Tracker running", GOOD, pulse=True) if _running() else pill("Tracker idle", MUTED)
_db_pill = pill("Database connected", GOOD) if db_ok() else pill("Database offline", BAD)
st.markdown(
    '<div class="hero"><div><div class="eyebrow">Real-time visitor intelligence</div><h1>Face Tracker</h1>'
    '<p>Detects and tracks people, recognises faces, registers every new visitor once and logs each entry and exit.</p>'
    f'</div><div>{_state_pill}{_db_pill}</div></div>', unsafe_allow_html=True)


# ---- results ----------------------------------------------------------------------
@st.fragment(run_every=refresh if refresh else None)
def dashboard():
    try:
        faces = query("SELECT face_id, kind, first_seen FROM faces ORDER BY face_id")
        events = query("SELECT id, face_id, event_type, ts, image_path, track_id FROM events ORDER BY id")
    except Exception as e:  # noqa: BLE001
        st.error(f"Cannot reach PostgreSQL ({e}). Start it with `docker compose up -d`.")
        return

    kinds = dict(zip(faces.face_id, faces.kind))
    events["visitor"] = [label(kinds.get(v, "face"), v) for v in events.face_id]
    events["kind"] = [kinds.get(v, "face") for v in events.face_id]
    faces["visitor"] = [label(k, v) for k, v in zip(faces.kind, faces.face_id)]

    last = events.sort_values("id").groupby("face_id").event_type.last() if len(events) else pd.Series(dtype=str)
    inside = int((last == "entry").sum())
    n_entry, n_exit = int((events.event_type == "entry").sum()), int((events.event_type == "exit").sum())
    kpi_row([
        ("Unique visitors", len(faces), "counted once each", ACCENT),
        ("By face", int((faces.kind == "face").sum()), "F#### IDs", "#8B7CFF"),
        ("Body-only", int((faces.kind == "body").sum()), "P#### IDs", WARN),
        ("Entries / exits", f"{n_entry} / {n_exit}", "logged events", ACCENT2),
        ("In frame now", inside, "last event was an entry", GOOD),
    ], key="kpis_results")

    ev = events[events.kind.isin(kind_filter) & events.event_type.isin(type_filter)]
    st.write("")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([":material/show_chart: Timeline", ":material/groups: Visitors",
                                            ":material/receipt_long: Events", ":material/photo_library: Images & log",
                                            ":material/summarize: Report"])

    with tab1:
        if ev.empty:
            empty("No events yet. Start a run in the <b>Run</b> tab, or run <code>python main.py --source &lt;video|rtsp&gt;</code>.")
        else:
            t = ev.assign(ts=pd.to_datetime(ev.ts))
            span = t.ts.max() - t.ts.min()
            freq, unit = ("10s", "10-second") if span < pd.Timedelta(minutes=10) else ("min", "per-minute")
            t["bucket"] = t.ts.dt.floor(freq)
            counts = t.groupby(["bucket", "event_type"]).size().reset_index(name="count")
            c1, c2 = st.columns([3, 2])
            with c1:
                section("Entries and exits", f"{unit} buckets")
                bars = alt.Chart(counts).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("bucket:T", title=None), y=alt.Y("count:Q", title=None),
                    color=alt.Color("event_type:N", title=None,
                                    scale=alt.Scale(domain=["entry", "exit"], range=[GOOD, BAD])),
                    tooltip=["bucket:T", "event_type:N", "count:Q"]).properties(height=280)
                st.altair_chart(style_chart(bars), width="stretch")
            with c2:
                section("Visitor mix", "face vs body-only")
                mix = pd.DataFrame({"type": ["Face", "Body-only"],
                                    "n": [int((faces.kind == "face").sum()), int((faces.kind == "body").sum())]})
                donut = alt.Chart(mix).mark_arc(innerRadius=62, cornerRadius=5).encode(
                    theta="n:Q", color=alt.Color("type:N", title=None,
                                                 scale=alt.Scale(domain=["Face", "Body-only"], range=[ACCENT, WARN])),
                    tooltip=["type:N", "n:Q"]).properties(height=280)
                st.altair_chart(style_chart(donut), width="stretch")
            section("New unique visitors over time", "cumulative")
            f = faces.assign(first_seen=pd.to_datetime(faces.first_seen)).sort_values("first_seen")
            f["cumulative"] = range(1, len(f) + 1)
            area = alt.Chart(f).mark_area(
                line={"color": ACCENT2, "strokeWidth": 2.5}, interpolate="step-after",
                color=alt.Gradient(gradient="linear", stops=[alt.GradientStop(color="rgba(34,211,238,0.35)", offset=0),
                                                              alt.GradientStop(color="rgba(34,211,238,0.02)", offset=1)],
                                   x1=1, x2=1, y1=1, y2=0)).encode(
                x=alt.X("first_seen:T", title=None), y=alt.Y("cumulative:Q", title=None),
                tooltip=["visitor:N", "first_seen:T", "cumulative:Q"]).properties(height=240)
            st.altair_chart(style_chart(area), width="stretch")

    with tab2:
        if len(events):
            agg = _visitor_table(events, faces)
            section("All visitors", f"{len(agg)} unique")
            st.dataframe(agg[agg.kind.isin(kind_filter)], width="stretch")
        else:
            empty("No visitors yet.")

    with tab3:
        section("Event log", "newest first")
        if ev.empty:
            empty("No events match the current filters.")
        else:
            st.dataframe(ev[["id", "ts", "event_type", "visitor", "kind", "track_id", "image_path"]].iloc[::-1],
                         width="stretch", hide_index=True)
            st.download_button("Download events CSV", ev.to_csv(index=False), "events.csv", "text/csv",
                               icon=":material/download:")

    with tab4:
        section("Latest captures", "cropped entry and exit images")
        recent = ev[ev.image_path.notna() & (ev.image_path != "")].iloc[::-1].head(gallery_n)
        shown = 0
        with st.container(key="gallery"):               # key -> CSS class used to crop thumbnails evenly
            cols = st.columns(6)
            for r in recent.itertuples():
                p = ROOT / r.image_path
                if p.exists():
                    with cols[shown % 6]:
                        st.image(str(p), width="stretch")
                        st.markdown(f'{badge(r.visitor)}<span class="badge {r.event_type}">{r.event_type}</span>'
                                    f'<div class="cap">{pd.to_datetime(r.ts):%H:%M:%S}</div>', unsafe_allow_html=True)
                    shown += 1
        if not shown:
            empty("No images yet.")
        section("events.log", "last 60 lines")
        lf = log_dir / "events.log"
        if lf.exists():
            st.code("\n".join(lf.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]), language="text")
        else:
            st.caption(f"{lf} not found")

    with tab5:
        section("Run report", "summary and downloads")
        if events.empty:
            empty("Nothing to report yet. Run the tracker first.")
        else:
            summary = _summary(faces, events)
            st.code(summary, language="text")
            with_img = st.toggle("Include the entry/exit images in the zip", value=False)
            d1, d2, d3 = st.columns(3)
            d1.download_button("Visitors CSV", _visitor_table(events, faces).to_csv(), "visitors.csv", "text/csv",
                               icon=":material/groups:", width="stretch")
            d2.download_button("Events CSV", events.to_csv(index=False), "events.csv", "text/csv",
                               icon=":material/receipt_long:", width="stretch")
            d3.download_button("Full report (.zip)", _report_zip(summary, faces, events, with_img),
                               "face_tracker_report.zip", "application/zip", icon=":material/folder_zip:", width="stretch")
            st.caption("Reports always include every visitor and event, whatever the sidebar filters are set to.")


# ---- run control + live view ---------------------------------------------------
def _read_status() -> dict:
    try:
        return json.loads((log_dir / "status.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _list_videos() -> dict:
    found = {}
    for folder in (ROOT / "uploads", ROOT / "Sample"):
        if folder.exists():
            for f in sorted(folder.rglob("*")):
                if f.suffix.lower() in VIDEO_EXT:
                    found[str(f.relative_to(ROOT))] = f
    return found


def run_panel():
    running = _running()
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        with st.container(border=True):
            section("1 · Choose a source", "video file or live camera")
            up = st.file_uploader("Upload a sample video", type=[e.strip(".") for e in VIDEO_EXT], disabled=running)
            if up is not None and not running:
                dest = ROOT / "uploads" / up.name
                dest.parent.mkdir(exist_ok=True)
                if not dest.exists() or dest.stat().st_size != up.size:
                    dest.write_bytes(up.getbuffer())
                    st.success(f"Saved to uploads/{up.name}")

            mode = st.radio("Source", ["Video file", "RTSP / live camera"], horizontal=True, disabled=running)
            source = None
            if mode == "Video file":
                vids = _list_videos()
                if vids:
                    names = list(vids)
                    default = next((i for i, n in enumerate(names) if up is not None and n.endswith(up.name)), 0)
                    choice = st.selectbox("Video", names, index=default, disabled=running)
                    source = str(vids[choice])
                else:
                    st.info("Upload a video above (or put videos in the Sample/ folder).")
            else:
                source = st.text_input("RTSP URL", "rtsp://", disabled=running,
                                       help="e.g. rtsp://user:pass@192.168.1.50:554/stream1")

    with right:
        with st.container(border=True):
            section("2 · Run", "start and stop tracking")
            st.markdown("".join(f'<span class="chip">{k}: {v}</span>' for k, v in (
                ("device", CFG.get("device", "cpu")), ("skip frames", CFG.get("detection_skip_frames")),
                ("detect size", CFG.get("detection_imgsz")), ("face threshold", CFG.get("recognition_threshold")))),
                unsafe_allow_html=True)
            reset = st.checkbox("Reset database and logs before this run (start counting from zero)", value=True,
                                disabled=running)

            if st.button("Start tracking", type="primary", icon=":material/play_arrow:", width="stretch",
                         disabled=running or not source or source == "rtsp://"):
                if reset:
                    with psycopg2.connect(**_conn_params()) as conn, conn.cursor() as cur:
                        from src.db import SCHEMA            # a fresh database has no tables until the tracker first runs
                        cur.execute(SCHEMA)
                        cur.execute("TRUNCATE events, faces RESTART IDENTITY CASCADE")
                    for sub in ("entries", "exits"):
                        shutil.rmtree(log_dir / sub, ignore_errors=True)
                    (log_dir / "events.log").unlink(missing_ok=True)
                log_dir.mkdir(exist_ok=True)
                for stale in ("stop.flag", "status.json", "live.jpg"):    # do not show the previous run's numbers
                    (log_dir / stale).unlink(missing_ok=True)
                out = open(log_dir / "run.out", "w", encoding="utf-8")
                _proc_holder()["proc"] = subprocess.Popen(
                    [sys.executable, "main.py", "--source", source], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                st.success("Started. Open the Live tab.")
                time.sleep(1)
                st.rerun()
            if st.button("Stop (finish cleanly)", icon=":material/stop:", width="stretch", disabled=not running):
                (log_dir / "stop.flag").write_text("stop", encoding="utf-8")   # main.py closes open tracks, then exits
                st.info("Stop requested. Open exits are being logged...")
            st.markdown(pill("Tracker running", GOOD, pulse=True) if running else pill("Tracker idle", MUTED),
                        unsafe_allow_html=True)

    ro = log_dir / "run.out"
    if ro.exists() and not running and ro.stat().st_size:
        txt = ro.read_text(encoding="utf-8", errors="replace")
        if "Traceback" in txt:
            st.error("The last run crashed:")
            st.code(txt[-1500:], language="text")


@st.fragment(run_every=1)
def live_panel():
    status = _read_status()
    state = "running" if _running() else status.get("state", "idle")
    img = log_dir / "live.jpg"
    total = status.get("total", 0)
    frame_txt = f"{status.get('frame', 0)} / {total}" if total > 0 else status.get("frame", 0)
    st_color = {"running": GOOD, "finished": ACCENT2}.get(state, MUTED)
    kpi_row([
        ("State", state.capitalize(), status.get("source", "no source yet").split("\\")[-1].split("/")[-1], st_color),
        ("Unique visitors", status.get("visitors", 0), "counted so far", ACCENT),
        ("People in frame", status.get("active", 0), "currently tracked", GOOD),
        ("Processing FPS", status.get("fps", 0), "average this run", ACCENT2),
        ("Frame", frame_txt, "progress", WARN),
    ], key="kpis_live")
    if total > 0:
        st.progress(min(1.0, status.get("frame", 0) / total))
    st.write("")
    vid_col, feed_col = st.columns([3, 1.1], gap="large")
    with vid_col:
        if img.exists():
            try:
                st.image(img.read_bytes(), width="stretch")
            except Exception:  # noqa: BLE001
                pass
        else:
            empty("No live frame yet. Open the <b>Run</b> tab and start tracking.")
    with feed_col:
        section("Latest events", "live")
        try:
            rec = query("SELECT e.event_type, e.ts, e.face_id, f.kind FROM events e JOIN faces f USING (face_id) "
                        "ORDER BY e.id DESC LIMIT 8")
        except Exception:  # noqa: BLE001
            rec = pd.DataFrame()
        if rec.empty:
            empty("Waiting for the first entry...")
        else:
            for r in rec.itertuples():
                st.markdown(f'<div class="feed-item"><span>{badge(label(r.kind, r.face_id))}'
                            f'<span class="badge {r.event_type}">{r.event_type}</span></span>'
                            f'<span class="t">{pd.to_datetime(r.ts):%H:%M:%S}</span></div>', unsafe_allow_html=True)


def how_it_works():
    """Static explainer page. Numbers come from config.json (merged with the defaults), so it never goes out of date."""
    c = RUN_CFG
    n = c["detection_skip_frames"] + 1
    section("The pipeline", "from camera to unique visitor count")
    steps = [
        ("Capture", "Reads a video file or an RTSP stream. A reader thread keeps only the newest frame, so a live stream never lags."),
        ("Detect", f"YOLOv8 finds people and faces once every {n} frames (skip = {c['detection_skip_frames']}), which saves compute."),
        ("Track", "ByteTrack gives each person a stable ID from frame to frame, even when their face is not visible."),
        ("Recognise", "ArcFace turns a face into 512 numbers. OSNet does the same for a whole body, for people seen from behind."),
        ("Match", "The new fingerprints are compared with every stored visitor, then safety rules decide: same person or new."),
        ("Log", "Every entry and exit saves a cropped photo, a timestamp, an event type and an ID to a folder, events.log and PostgreSQL."),
        ("Count", "Unique visitors = the number of rows in the faces table. A returning person never adds to it."),
    ]
    st.markdown('<div class="flow">' + "".join(
        f'<div class="step"><div class="n">{i}</div><h4>{t}</h4><p>{d}</p></div>' for i, (t, d) in enumerate(steps, 1)) + '</div>',
        unsafe_allow_html=True)

    section("How it decides who is who", "thresholds are read from config.json")
    rules = [
        (f"&ge; {c['recognition_threshold']}", "Strong face match", "ArcFace similarity at or above this means the same person."),
        (f"&ge; {c['body_threshold']}", "Strong body match", "Body similarity alone can match someone whose face is not visible."),
        (f"{c['face_low_threshold']} + {c['body_low_threshold']}", "Face and body together",
         "A weaker face match and a weaker body match that agree are accepted as the same person."),
        (f"&lt; {c['face_veto']}", "Face veto", "If the faces clearly differ, no body similarity can override that."),
        ("Never together", "Two people on screen at once are different people",
         f"Applies to live matching and to the end-of-run cleanup (overlap of {c['overlap_frames']} frames or more)."),
        ("Continuity", "Where someone vanished matters",
         "A new track that starts near where, and soon after, a person disappeared can take over that identity, if the body does not contradict it."),
        (f"{c['exit_grace_frames']} frames", "Exit rule", "A person unseen for this long has left, and exactly one exit is logged."),
    ]
    st.markdown("".join(f'<div class="rule"><div class="k">{k}</div><div><b>{t}</b><span class="d">{d}</span></div></div>'
                        for k, t, d in rules), unsafe_allow_html=True)

    section("Visitor IDs")
    st.markdown(f'{badge("F0001")} identified by a face &nbsp;&nbsp; {badge("P0001")} tracked without a usable face (body-only). '
                'A body-only visitor is promoted to F when a face is seen later.', unsafe_allow_html=True)

    section("Tech stack", f"running on {c['device']}")
    tech = [("Python 3.11", "Backend and processing"), ("YOLOv8n (Ultralytics)", "Face and person detection"),
            ("ByteTrack", "Person tracking across frames"), ("InsightFace ArcFace", "Face recognition, 512-number fingerprints"),
            ("OSNet (torchreid)", "Body re-identification"), ("PyTorch", "Runs YOLO and OSNet on GPU or CPU"),
            ("OpenCV", "Video reading, cropping and drawing"), ("PostgreSQL 16 (Docker)", "Visitors, events and fingerprint galleries"),
            ("Streamlit", "This dashboard"), ("config.json", "All settings, including detection_skip_frames")]
    st.markdown('<div class="stack">' + "".join(f'<div class="tech"><b>{a}</b><span>{b}</span></div>' for a, b in tech) + '</div>',
                unsafe_allow_html=True)


t_live, t_run, t_data, t_how = st.tabs([":material/sensors: Live", ":material/play_circle: Run",
                                        ":material/insights: Results", ":material/help: How it works"])
with t_live:
    live_panel()
with t_run:
    run_panel()
with t_data:
    dashboard()
with t_how:
    how_it_works()
