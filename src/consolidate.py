"""End-of-run consolidation: merge duplicate visitors using everything we know, with a hard safety rule.

NEVER-TOGETHER RULE: two visitors that were on screen at the same time (in the same run) are different
people, however similar they look. Everything else is a merge candidate, so the thresholds can be looser than
the live matching ones (which have no such protection). Greedy: best-scoring pair first, then recompute.
"""
import logging

import numpy as np

log = logging.getLogger("facetracker")


def _best(a, b):
    """Best cosine similarity between two sets of embeddings (None if either set is empty)."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return None
    return float((a @ b.T).max())


def _overlap(spans_a, spans_b, min_frames):
    for ra, a0, a1 in spans_a:
        for rb, b0, b1 in spans_b:
            if ra == rb and min(a1, b1) - max(a0, b0) >= min_frames:
                return True
    return False


def consolidate(db, cfg) -> list:
    """Merge duplicate visitors in the database. Returns [(kept_id, dropped_id, rule, face_sim, body_sim)]."""
    kinds = {i: k for i, k, _ in db.load_bodies()}
    clusters = {i: {"ids": [i], "face": [], "body": [], "spans": []} for i in kinds}
    for vid, kind, vec in db.load_embeddings():
        if vid in clusters:
            clusters[vid][kind].append(vec)
    for vid, run, a, b in db.load_spans():
        if vid in clusters:
            clusters[vid]["spans"].append((run, a, b))
    for c in clusters.values():
        for k in ("face", "body"):
            c[k] = np.stack(c[k]) if c[k] else None

    merges = []
    while True:
        best, best_key = None, None
        ids = sorted(clusters)
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = clusters[ids[x]], clusters[ids[y]]
                if _overlap(a["spans"], b["spans"], cfg["overlap_frames"]):
                    continue                                   # seen together => different people
                f, bd = _best(a["face"], b["face"]), _best(a["body"], b["body"])
                if f is not None and f < cfg["face_veto"]:
                    continue                                   # faces clearly differ
                # Same video: the never-together rule protects us, so looser thresholds are safe.
                # Different videos: nothing proves they were never together -> use the strict live thresholds.
                if {r for r, _, _ in a["spans"]} & {r for r, _, _ in b["spans"]}:
                    t_face, t_body = cfg["consolidate_face"], cfg["consolidate_body"]
                    t_pf, t_pb = cfg["consolidate_pair_face"], cfg["consolidate_pair_body"]
                else:
                    t_face, t_body = cfg["recognition_threshold"], cfg["body_threshold"]
                    t_pf, t_pb = cfg["face_low_threshold"], cfg["body_low_threshold"]
                if f is not None and f >= t_face:
                    rule = "face"
                elif bd is not None and bd >= t_body:
                    rule = "body"
                elif f is not None and bd is not None and f >= t_pf and bd >= t_pb:
                    rule = "face+body"
                else:
                    continue
                score = max((f or 0) / t_face, (bd or 0) / t_body)
                key = (rule == "face", score)
                if best_key is None or key > best_key:
                    best, best_key = (ids[x], ids[y], rule, f, bd), key
        if best is None:
            break
        keep, drop, rule, f, bd = best
        k, d = clusters[keep], clusters.pop(drop)
        k["ids"] += d["ids"]
        k["spans"] += d["spans"]
        for key_ in ("face", "body"):
            parts = [x for x in (k[key_], d[key_]) if x is not None]
            k[key_] = np.vstack(parts) if parts else None
        merges.append(best)
    return merges


def apply_merges(db, merges):
    """Apply merges to the database. A merged-away id may itself have absorbed others, so map ids as we go."""
    alias = {}
    for keep, drop, rule, f, bd in merges:
        keep = alias.get(keep, keep)
        db.merge_visitors(keep, drop)
        alias[drop] = keep
        log.info("CONSOLIDATE merged visitor %d into %d (rule=%s face_sim=%s body_sim=%s)", drop, keep, rule,
                 "n/a" if f is None else f"{f:.3f}", "n/a" if bd is None else f"{bd:.3f}")
