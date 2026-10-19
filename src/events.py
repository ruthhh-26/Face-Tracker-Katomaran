"""Entry/exit state machine at PERSON-track level, with face + body re-identification.

A person track is the unit that enters/exits. Identity is resolved in this order:
  1. Face (ArcFace) match            -> F-ID   (strongest cue)
  2. Body (OSNet) match              -> links back-view and front-view tracks of one person
  3. Neither matches                 -> new visitor (F-ID if a face was seen, else body-only P-ID)
A visitor currently visible on another track is never matched (two people on screen are two people).
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

log = logging.getLogger("facetracker")


@dataclass
class TrackState:
    track_id: int
    first_frame: int
    last_seen: int
    hits: int = 0
    embs: list = field(default_factory=list)        # face embeddings
    body_embs: list = field(default_factory=list)   # body embeddings
    body_hs: list = field(default_factory=list)     # person-box height each body embedding was taken at
    face_id: int = None
    entered: bool = False
    face_crop: np.ndarray = None   # largest face crop seen
    face_area: int = 0
    person_crop: np.ndarray = None  # largest person crop seen
    person_area: int = 0
    last_face: np.ndarray = None
    last_person: np.ndarray = None
    first_box: tuple = None
    last_box: tuple = None
    trail: list = field(default_factory=list)   # (frame, centre_x, centre_y) of the last few detections


class EventManager:
    def __init__(self, cfg, recognizer, db, images, embedder=None, gallery=None):
        self.cfg, self.rec, self.db, self.images = cfg, recognizer, db, images
        self.embedder = embedder
        self.fg = recognizer.gallery            # face gallery (ArcFace)
        self.bg = gallery                       # body gallery (OSNet), None when re-ID is disabled
        self.tracks: dict = {}
        self.recent: list = []                  # tracks that just ended: (visitor id, last_box, last_seen, velocity)
        self.frame_idx = 0
        self.total_visitors = db.count_unique()               # live counter shown in the overlay
        self.kinds = {i: k for i, k, _ in db.load_bodies()}   # visitor id -> 'face' | 'body'
        for vid, kind, vec in db.load_embeddings():           # known visitors survive restarts / new videos
            if kind == "face":
                self.fg.add(vid, vec, force=True)
            elif self.bg is not None:
                self.bg.add(vid, vec, force=True)

    def label(self, t: TrackState) -> str:
        return f"{'F' if self.kinds.get(t.face_id) == 'face' else 'P'}{t.face_id:04d}"

    @staticmethod
    def _crop(frame, box, pad=0.1):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        return frame[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)].copy()

    @staticmethod
    def _assign_faces(persons, faces):
        """Map person track_id -> its face. A face belongs to the person whose box it fits best: centred
        horizontally, near the TOP of the box, and a plausible size (6-40% of the person's height).
        (Choosing the tightest containing box wrongly gave a nearby adult's face to a small child in front.)"""
        out = {}
        for f in faces:
            fx1, fy1, fx2, fy2 = f.box
            cx, cy, fh = (fx1 + fx2) / 2, (fy1 + fy2) / 2, fy2 - fy1
            fa = (fx2 - fx1) * fh
            best, best_cost = None, None
            for p in persons:
                x1, y1, x2, y2 = p.box
                pw, ph = x2 - x1, y2 - y1
                if not (x1 <= cx <= x2 and y1 <= cy <= y1 + 0.45 * ph) or not (0.06 <= fh / ph <= 0.40):
                    continue
                cost = abs(cx - (x1 + x2) / 2) / pw + abs(cy - (y1 + 0.12 * ph)) / ph
                if best_cost is None or cost < best_cost:
                    best, best_cost = p, cost
            if best is not None and (best.track_id not in out or fa > out[best.track_id][1]):
                out[best.track_id] = (f, fa)
        return {k: v[0] for k, v in out.items()}

    @staticmethod
    def _drop_union_boxes(persons, contain=0.8, min_inside=2):
        """Drop a person box that swallows two or more other person boxes (one box over a whole group).
        Such boxes mix several people's appearance and wreck identity; the individual boxes are kept."""
        out = []
        for a in persons:
            ax1, ay1, ax2, ay2 = a.box
            area_a = (ax2 - ax1) * (ay2 - ay1)
            inside = 0
            for b in persons:
                if b is a:
                    continue
                bx1, by1, bx2, by2 = b.box
                area_b = (bx2 - bx1) * (by2 - by1)
                inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
                if area_b > 0 and inter / area_b >= contain and area_b < 0.85 * area_a:
                    inside += 1
            if inside >= min_inside:
                log.info("UNION_BOX_DROPPED track=%d box=%s covers %d other people", a.track_id,
                         tuple(int(v) for v in a.box), inside)
                continue
            out.append(a)
        return out

    # ---- per detection cycle -----------------------------------------------
    def update(self, frame, persons, faces, frame_idx):
        self.frame_idx = frame_idx
        if self.cfg.get("drop_union_boxes", True):
            persons = self._drop_union_boxes(persons)
        face_of = self._assign_faces(persons, faces)
        for p in persons:
            t = self.tracks.get(p.track_id)
            if t is None:
                t = self.tracks[p.track_id] = TrackState(p.track_id, frame_idx, frame_idx)
                log.info("TRACK_START track=%d frame=%d box=%s", p.track_id, frame_idx, tuple(int(v) for v in p.box))
            t.last_seen = frame_idx
            t.hits += 1
            t.first_box = t.first_box or p.box
            t.last_box = p.box
            t.trail = (t.trail + [(frame_idx, (p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2)])[-8:]
            pc = self._crop(frame, p.box)
            if pc.size:
                t.last_person = pc
                area = (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])
                if area > t.person_area:
                    t.person_crop, t.person_area = pc, area
                self._sample_body(t, pc, p.box)

            f = face_of.get(p.track_id)
            if f is not None:
                self._use_face(frame, t, f)

            if t.face_id is None:                      # identity decision
                if (len(t.embs) >= self.cfg["embedding_samples"] and t.hits >= self.cfg["min_confirm_frames"]) \
                        or t.hits >= self.cfg["body_confirm_frames"]:
                    self._identify(t)
            elif self.kinds.get(t.face_id) == "body" and len(t.embs) >= self.cfg["embedding_samples"]:
                self._upgrade_late_face(t)             # body-only visitor whose face finally showed up
            if t.face_id is not None and not t.entered:
                self._emit(t, "entry", self._best_crop(t))

        self._expire(frame_idx)

    def _sample_body(self, t, crop, box):
        """OSNet embeddings per track (CPU-friendly): a few early ones, plus one whenever the person becomes
        noticeably bigger (closer) than anything sampled so far - near crops are the most reliable."""
        h = box[3] - box[1]
        if self.embedder is None or h < self.cfg["min_body_height"]:
            return
        n, cap = len(t.body_embs), self.cfg["body_samples"]
        first = n < cap and t.hits % self.cfg["body_sample_interval"] == 1
        bigger = n < 3 * cap and h > 1.25 * max(t.body_hs, default=0)
        if first or bigger:
            t.body_embs.append(self.embedder.embed(crop, key=(self.frame_idx, t.track_id)))
            t.body_hs.append(h)

    def _body_vec(self, t: TrackState):
        """Mean of the track's body embeddings taken at >= 70% of its largest size (small far crops are noisy)."""
        if not t.body_embs or self.bg is None:
            return None
        top = max(t.body_hs)
        return self._mean([e for e, h in zip(t.body_embs, t.body_hs) if h >= 0.7 * top])

    def _use_face(self, frame, t: TrackState, f):
        x1, y1, x2, y2 = f.box
        size = min(x2 - x1, y2 - y1)
        fc = self._crop(frame, f.box, 0.15)
        if fc.size == 0:
            return
        t.last_face = fc
        if size >= self.cfg["min_face_size"]:
            if (x2 - x1) * (y2 - y1) > t.face_area:
                t.face_crop, t.face_area = fc, (x2 - x1) * (y2 - y1)
            if (t.face_id is None or self.kinds.get(t.face_id) == "body") \
                    and len(t.embs) < self.cfg["embedding_samples"]:
                t.embs.append(self.rec.embed(frame, f.box, f.kps))
                log.info("EMBEDDING track=%d sample=%d/%d", t.track_id, len(t.embs), self.cfg["embedding_samples"])

    def _best_crop(self, t: TrackState):
        if self.kinds.get(t.face_id) == "face" and t.face_crop is not None:
            return t.face_crop
        return t.person_crop if t.person_crop is not None else (t.last_person if t.last_person is not None else t.last_face)

    def _exit_crop(self, t: TrackState):
        if self.kinds.get(t.face_id) == "face" and t.last_face is not None:
            return t.last_face
        return t.last_person if t.last_person is not None else t.last_face

    # ---- identity ------------------------------------------------------------
    @staticmethod
    def _mean(vecs):
        m = np.mean(vecs, axis=0)
        return m / (np.linalg.norm(m) + 1e-9)

    def _active_ids(self, t):
        """Visitors visible right now on another track. A track unseen for > lost_after_frames is 'lost' (no visual)
        and is therefore a prediction candidate, not an exclusion."""
        return {o.face_id for o in self.tracks.values() if o is not t and o.face_id is not None
                and self.frame_idx - o.last_seen <= self.cfg["lost_after_frames"]}

    @staticmethod
    def _velocity(trail):
        """Pixels per frame (vx, vy) over the recent trail; (0, 0) if not enough history."""
        if len(trail) < 2 or trail[-1][0] == trail[0][0]:
            return 0.0, 0.0
        dt = trail[-1][0] - trail[0][0]
        return (trail[-1][1] - trail[0][1]) / dt, (trail[-1][2] - trail[0][2]) / dt

    def _continuity_ids(self, t):
        """Where could each person who is currently out of sight have gone? For every lost visitor (closed track or
        open-but-unseen track) build a prediction zone that GROWS WITH TIME IN ALL DIRECTIONS around where they
        vanished, plus a second zone around where their last velocity would have carried them. A new track that
        starts inside a zone, at a plausible size, is a candidate for being that same person.
        Returns {visitor id: normalised distance (<= 1 means inside the zone)}."""
        if t.first_box is None:
            return {}
        c = self.cfg
        self.recent = [r for r in self.recent if self.frame_idx - r[2] <= 4 * c["continuity_frames"]]
        lost = [(vid, box, seen, vel) for vid, box, seen, vel in self.recent]
        for o in self.tracks.values():          # open tracks that have not been seen for a while
            if o is not t and o.face_id is not None and o.last_box is not None                     and self.frame_idx - o.last_seen > c["lost_after_frames"]:
                lost.append((o.face_id, o.last_box, o.last_seen, self._velocity(o.trail)))
        x1, y1, x2, y2 = t.first_box
        ncx, ncy, nh = (x1 + x2) / 2, (y1 + y2) / 2, y2 - y1
        out = {}
        for vid, (lx1, ly1, lx2, ly2), last_seen, (vx, vy) in lost:
            gap = t.first_frame - last_seen
            if not (-c["lost_after_frames"] <= gap <= c["continuity_frames"]):
                continue
            gap = max(gap, 0)
            lh = ly2 - ly1
            h = max(nh, lh)
            if not 0.5 <= nh / max(lh, 1) <= 2.0:             # perspective: size cannot jump
                continue
            lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
            r_all = h * (c["zone_base"] + c["zone_growth"] * gap)           # all directions, grows with time
            d_all = ((ncx - lcx) ** 2 + (ncy - lcy) ** 2) ** 0.5
            reach = min(gap, 60)                                              # do not extrapolate a line for ever
            px, py = lcx + vx * reach, lcy + vy * reach
            r_dir = h * (c["zone_base"] + 0.7 * c["zone_growth"] * gap)
            d_dir = ((ncx - px) ** 2 + (ncy - py) ** 2) ** 0.5
            d = min(d_all / r_all, d_dir / r_dir)
            if d <= 1.0 and d < out.get(vid, 9.0):
                out[vid] = d
        return out

    def _best_match(self, femb, bemb, exclude, cont=None):
        """Fuse face + body evidence per known visitor. Accept when ANY of:
             face >= recognition_threshold                       (strong face)
             body >= body_threshold                              (strong body)
             face >= face_low_threshold AND body >= body_low_threshold   (two weak cues agree)
        Face-accepted candidates win over body-only ones; ties broken by normalised score."""
        c = self.cfg
        cont = cont or {}
        fs = self.fg.scores(femb) if femb is not None else {}
        bs = self.bg.scores(bemb) if (bemb is not None and self.bg is not None) else {}
        best, best_key = None, None
        for vid in (set(fs) | set(bs) | set(cont)) - exclude:
            f, b = fs.get(vid), bs.get(vid)
            if f is not None and f < c["face_veto"]:    # faces clearly differ: body similarity cannot override that
                continue
            nf = f / c["recognition_threshold"] if f is not None else 0.0
            nb = b / c["body_threshold"] if b is not None else 0.0
            if f is not None and f >= c["recognition_threshold"]:
                rule = "face"
            elif b is not None and b >= c["body_threshold"]:
                rule = "body"
            elif f is not None and b is not None and f >= c["face_low_threshold"] and b >= c["body_low_threshold"]:
                rule = "face+body"
            elif vid in cont and ((b is not None and b >= c["continuity_body_threshold"])
                                  or (f is not None and f >= c["face_low_threshold"])):
                rule = "continuity"      # same place + moments later + not contradicted by body/face
            elif c["motion_only"] and vid in cont and (f is None or f >= c["face_low_threshold"])                     and (b is None or b >= c["motion_body_floor"])                     and all(cont[vid] <= 0.6 * d for v, d in cont.items() if v != vid):
                rule = "motion"          # the only person who could have walked here from where they vanished
            else:
                continue
            key = (rule == "face", max(nf, nb, (nf + nb) / 2))
            if best_key is None or key > best_key:
                best, best_key = (vid, f, b, rule), key
        top_f = max(fs.values()) if fs else None
        top_b = max(bs.values()) if bs else None
        return best, top_f, top_b

    def _remember(self, vid, femb, bemb):
        """Add the new appearance to the galleries (only if it is novel) and persist it."""
        if femb is not None and self.fg.add(vid, femb):
            self.db.add_embedding(vid, "face", femb)
        if bemb is not None and self.bg is not None and self.bg.add(vid, bemb):
            self.db.add_embedding(vid, "body", bemb)

    def _identify(self, t: TrackState):
        """Match against all known visitors (face + body fusion), else register a new one."""
        femb = self._mean(t.embs) if t.embs else None
        bemb = self._body_vec(t)
        cont = self._continuity_ids(t)
        if cont:
            bs_dbg = self.bg.scores(bemb) if bemb is not None else {}
            log.info("PREDICTION_ZONE track=%d candidates(zone_dist)=%s body_sims=%s", t.track_id,
                     {v: round(d, 2) for v, d in sorted(cont.items())},
                     {v: round(bs_dbg.get(v, float("nan")), 3) for v in sorted(cont)})
        best, top_f, top_b = self._best_match(femb, bemb, self._active_ids(t), cont)
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        if best is not None:
            vid, f, b, rule = best
            t.face_id = vid
            log.info("MATCH id=%d track=%d rule=%s face_sim=%s body_sim=%s", vid, t.track_id, rule, fmt(f), fmt(b))
            if femb is not None and self.kinds.get(vid) == "body":
                self._promote(vid, femb)                # earlier back-view visitor now has a face
        else:
            kind = "face" if femb is not None else "body"
            t.face_id = self.db.add_face(femb, datetime.now().astimezone(), None, kind, bemb)
            self.kinds[t.face_id] = kind
            self.total_visitors += 1
            log.info("REGISTER id=%s%04d track=%d NEW visitor (best face_sim=%s, best body_sim=%s, no match)",
                     "F" if kind == "face" else "P", t.face_id, t.track_id, fmt(top_f), fmt(top_b))
        self._remember(t.face_id, femb, bemb)

    def _promote(self, vid, femb):
        """Body-only visitor gets a face identity (and joins the face gallery)."""
        self.db.upgrade_to_face(vid, femb)
        self.kinds[vid] = "face"
        self.fg.add(vid, femb, force=True)
        self.db.add_embedding(vid, "face", femb)
        log.info("PROMOTE id=P%04d -> F%04d (face seen after back view)", vid, vid)

    def _merge_into(self, t: TrackState, vid: int):
        """The body-only visitor of this track is the same person as known face visitor `vid`: fold them together."""
        old = t.face_id
        self.db.merge_visitors(vid, old)
        self.fg.remap(old, vid)
        if self.bg is not None:
            self.bg.remap(old, vid)
        self.kinds.pop(old, None)
        self.total_visitors -= 1
        for other in self.tracks.values():
            if other.face_id == old:
                other.face_id = vid
        self.recent = [(vid if v == old else v, box, seen, vel) for v, box, seen, vel in self.recent]
        log.info("MERGE_HEAD_BODY track=%d: body-only visitor %d is the same person as F%04d "
                 "(head and body seen together)", t.track_id, old, vid)

    def _upgrade_late_face(self, t: TrackState):
        """A track first identified by body now shows a face. Head and body in the same track are one person:
        attach the face; if it matches a known visitor, merge the body-only record into that visitor."""
        femb = self._mean(t.embs)
        scores = {v: s for v, s in self.fg.scores(femb).items() if v != t.face_id}
        busy = self._active_ids(t)
        best = max(((s, v) for v, s in scores.items() if v not in busy), default=(0.0, None))
        if best[1] is not None and best[0] >= self.cfg["recognition_threshold"]:
            self._merge_into(t, best[1])
            self._remember(t.face_id, femb, self._body_vec(t))
        else:
            self._promote(t.face_id, femb)
            if best[1] is None and scores and max(scores.values()) >= self.cfg["recognition_threshold"]:
                log.info("LATE_FACE_MATCH track=%d matches a visitor on screen now; kept separate", t.track_id)
        t.embs = []

    # ---- events ---------------------------------------------------------------
    def _emit(self, t: TrackState, kind: str, crop):
        now = datetime.now().astimezone()      # timezone-aware, so PostgreSQL (timestamptz) stores the exact moment
        lab = self.label(t)
        path = self.images.save("entries" if kind == "entry" else "exits", lab, now, crop)
        self.db.add_event(t.face_id, kind, now, path, t.track_id)   # image first, then DB row
        if kind == "entry":
            t.entered = True
        log.info("%s id=%s track=%d image=%s", kind.upper(), lab, t.track_id, path)

    def _close(self, t: TrackState, at_end: bool = False):
        """Finish a track: guarantee entry (if it was a real person) then exactly one exit.
        at_end=True (stream ended): someone who only just walked into frame must still be counted, so the
        minimum-hits bar is lowered - but they still need a usable body or face crop (guards against noise)."""
        if at_end and t.face_id is None and t.hits >= self.cfg["min_hits_at_end"] and (t.embs or t.body_embs):
            self._identify(t)
            log.info("LATE_ARRIVAL track=%d counted at end of stream (hits=%d)", t.track_id, t.hits)
        if t.face_id is None and ((t.embs and t.hits >= self.cfg["min_confirm_frames"])
                                  or t.hits >= self.cfg["min_body_hits"]):
            self._identify(t)
        if t.face_id is None:
            log.info("TRACK_DISCARDED track=%d (too short, hits=%d)", t.track_id, t.hits)
            return
        if not t.entered:
            self._emit(t, "entry", self._best_crop(t))
        self._emit(t, "exit", self._exit_crop(t))
        if t.body_embs and self.bg is not None:          # remember the best (closest) body view of this person
            b = self._body_vec(t)
            if self.bg.add(t.face_id, b):
                self.db.add_embedding(t.face_id, "body", b)
        if t.last_box is not None:
            self.recent.append((t.face_id, t.last_box, t.last_seen, self._velocity(t.trail)))
        self.db.add_span(t.face_id, self.cfg.get("run_id", "run"), t.first_frame, t.last_seen)   # for consolidation
        log.info("TRACK_END track=%d id=%s frames=%d-%d last_box=%s", t.track_id, self.label(t), t.first_frame, t.last_seen,
                 tuple(int(v) for v in t.last_box) if t.last_box is not None else None)

    def _expire(self, frame_idx):
        for tid in [k for k, t in self.tracks.items()
                    if frame_idx - t.last_seen > self.cfg["exit_grace_frames"]]:
            self._close(self.tracks.pop(tid))

    def flush(self):
        """Stream ended / interrupted: close every open track so no exit is lost."""
        for tid in list(self.tracks):
            self._close(self.tracks.pop(tid), at_end=True)
