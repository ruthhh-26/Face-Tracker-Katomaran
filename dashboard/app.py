"""Live dashboard for the face tracker: unique visitors, entry/exit events, images, events.log.

Run:  .venv\\Scripts\\streamlit run dashboard/app.py
Reads the same PostgreSQL DB and logs/ folder that main.py writes to, so it works while tracking is running.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

st.set_page_config(page_title="Face Tracker Dashboard", page_icon="👤", layout="wide")


@st.cache_resource
def _conn_params():
    d = CFG["db"]
    return dict(host=d["host"], port=d["port"], dbname=d["name"], user=d["user"], password=d["password"])


def query(sql: str) -> pd.DataFrame:
    """Fresh short-lived connection per query: survives DB restarts and needs no cache invalidation."""
    with psycopg2.connect(**_conn_params()) as conn:
        return pd.read_sql_query(sql, conn)


def label(row_kind, vid):
    return f"{'F' if row_kind == 'face' else 'P'}{int(vid):04d}"


# ---- sidebar -----------------------------------------------------------------
st.sidebar.title("Face Tracker")
refresh = st.sidebar.slider("Auto-refresh (seconds, 0 = off)", 0, 30, 5)
log_dir = ROOT / st.sidebar.text_input("Log folder", CFG.get("log_dir", "logs"))
kind_filter = st.sidebar.multiselect("Visitor type", ["face", "body"], default=["face", "body"])
type_filter = st.sidebar.multiselect("Event type", ["entry", "exit"], default=["entry", "exit"])
gallery_n = st.sidebar.slider("Images shown", 6, 48, 18, step=6)
st.sidebar.caption("F#### = recognised by face, P#### = person tracked without a visible face")


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

    # KPIs
    last = events.sort_values("id").groupby("face_id").event_type.last() if len(events) else pd.Series(dtype=str)
    inside = int((last == "entry").sum())
    c = st.columns(5)
    c[0].metric("Unique visitors", len(faces))
    c[1].metric("By face", int((faces.kind == "face").sum()))
    c[2].metric("Body-only", int((faces.kind == "body").sum()))
    c[3].metric("Entries / exits", f"{int((events.event_type == 'entry').sum())} / {int((events.event_type == 'exit').sum())}")
    c[4].metric("Currently in frame", inside)

    ev = events[events.kind.isin(kind_filter) & events.event_type.isin(type_filter)]

    tab1, tab2, tab3, tab4 = st.tabs(["Timeline", "Visitors", "Events", "Images & log"])

    with tab1:
        if ev.empty:
            st.info("No events yet. Run `python main.py --source <video|rtsp>`.")
        else:
            t = ev.assign(minute=pd.to_datetime(ev.ts).dt.floor("min"))
            st.subheader("Events per minute")
            st.bar_chart(t.pivot_table(index="minute", columns="event_type", values="id", aggfunc="count").fillna(0))
            st.subheader("New unique visitors over time")
            f = faces.assign(first_seen=pd.to_datetime(faces.first_seen)).sort_values("first_seen")
            f["cumulative"] = range(1, len(f) + 1)
            st.line_chart(f.set_index("first_seen")["cumulative"])

    with tab2:
        if len(events):
            agg = events.groupby("visitor").agg(
                entries=("event_type", lambda s: int((s == "entry").sum())),
                exits=("event_type", lambda s: int((s == "exit").sum())),
                first_event=("ts", "min"), last_event=("ts", "max"), last_type=("event_type", "last"))
            agg = agg.join(faces.set_index("visitor")[["kind"]])
            st.dataframe(agg[agg.kind.isin(kind_filter)], use_container_width=True)
        else:
            st.info("No visitors yet.")

    with tab3:
        st.dataframe(ev[["id", "ts", "event_type", "visitor", "kind", "track_id", "image_path"]].iloc[::-1],
                     use_container_width=True, hide_index=True)
        st.download_button("Download events CSV", ev.to_csv(index=False), "events.csv", "text/csv")

    with tab4:
        st.subheader("Latest captures")
        recent = ev[ev.image_path.notna()].iloc[::-1].head(gallery_n)
        cols = st.columns(6)
        for i, r in enumerate(recent.itertuples()):
            p = ROOT / r.image_path
            if p.exists():
                cols[i % 6].image(str(p), caption=f"{r.visitor} · {r.event_type} · {pd.to_datetime(r.ts):%H:%M:%S}",
                                  use_container_width=True)
        st.subheader("events.log (last 60 lines)")
        lf = log_dir / "events.log"
        if lf.exists():
            st.code("\n".join(lf.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]), language="text")
        else:
            st.caption(f"{lf} not found")


# ---- run control + live view ---------------------------------------------------
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@st.cache_resource
def _proc_holder() -> dict:
    return {"proc": None}      # survives Streamlit reruns / browser refreshes


def _running() -> bool:
    p = _proc_holder()["proc"]
    return p is not None and p.poll() is None


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
    st.subheader("Choose a source")
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

    reset = st.checkbox("Reset database and logs before this run (start counting from zero)", value=True,
                        disabled=running)

    c1, c2 = st.columns(2)
    if c1.button("▶ Start tracking", type="primary", disabled=running or not source or source == "rtsp://"):
        if reset:
            with psycopg2.connect(**_conn_params()) as conn, conn.cursor() as cur:
                cur.execute("TRUNCATE events, faces RESTART IDENTITY CASCADE")
            for sub in ("entries", "exits"):
                shutil.rmtree(log_dir / sub, ignore_errors=True)
            (log_dir / "events.log").unlink(missing_ok=True)
        log_dir.mkdir(exist_ok=True)
        (log_dir / "stop.flag").unlink(missing_ok=True)
        out = open(log_dir / "run.out", "w", encoding="utf-8")
        _proc_holder()["proc"] = subprocess.Popen(
            [sys.executable, "main.py", "--source", source], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        st.success("Started. Open the Live tab.")
        time.sleep(1)
        st.rerun()
    if c2.button("■ Stop (finish cleanly)", disabled=not running):
        (log_dir / "stop.flag").write_text("stop", encoding="utf-8")   # main.py closes open tracks, then exits
        st.info("Stop requested. Open exits are being logged...")
    st.caption("Status: " + ("🟢 tracker running" if running else "⚪ idle"))
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
    c = st.columns(5)
    c[0].metric("State", state)
    c[1].metric("Unique visitors", status.get("visitors", 0))
    c[2].metric("People in frame", status.get("active", 0))
    c[3].metric("Processing FPS", status.get("fps", 0))
    total = status.get("total", 0)
    c[4].metric("Frame", f"{status.get('frame', 0)}/{total}" if total > 0 else status.get("frame", 0))
    if total > 0:
        st.progress(min(1.0, status.get("frame", 0) / total))
    if img.exists():
        try:
            st.image(img.read_bytes(), use_container_width=True, caption=f"Source: {status.get('source', '')}")
        except Exception:  # noqa: BLE001
            pass
    else:
        st.info("No live frame yet. Go to the Run tab and start tracking.")


st.title("👤 Intelligent Face Tracker")
t_live, t_run, t_data = st.tabs(["🔴 Live", "▶ Run", "📊 Results"])
with t_live:
    live_panel()
with t_run:
    run_panel()
with t_data:
    dashboard()
