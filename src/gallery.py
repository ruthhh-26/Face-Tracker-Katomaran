"""Multi-embedding gallery: several appearances per visitor, matched by best (max) cosine similarity.

One mean embedding per person blurs different poses together; keeping a handful of diverse embeddings
lets a new sighting match whichever stored pose it resembles.
"""
import numpy as np


class MultiGallery:
    def __init__(self, dim: int = 512, max_per_id: int = 8, novelty: float = 0.92):
        self.max_per_id, self.novelty = max_per_id, novelty
        self.matrix = np.zeros((0, dim), dtype=np.float32)
        self.owner: list = []          # owner[i] = visitor id of matrix row i

    def __contains__(self, vid) -> bool:
        return vid in self.owner

    def scores(self, emb: np.ndarray) -> dict:
        """visitor id -> best cosine similarity over that visitor's stored embeddings."""
        if not self.owner:
            return {}
        sims = self.matrix @ emb
        out: dict = {}
        for vid, s in zip(self.owner, sims):
            if s > out.get(vid, -1.0):
                out[vid] = float(s)
        return out

    def add(self, vid: int, emb: np.ndarray, force: bool = False) -> bool:
        """Store `emb` for `vid` if it adds something new. Returns True when stored (caller persists it)."""
        rows = [i for i, o in enumerate(self.owner) if o == vid]
        if rows and not force:
            sims = self.matrix[rows] @ emb
            if sims.max() >= self.novelty:             # near-duplicate of a stored appearance
                return False
            if len(rows) >= self.max_per_id:           # full: replace the most similar (keeps diversity)
                self.matrix[rows[int(np.argmax(sims))]] = emb
                return True
        self.matrix = np.vstack([self.matrix, emb[None, :]])
        self.owner.append(vid)
        return True

    def remap(self, old: int, new: int):
        """A visitor was merged into another: its stored embeddings now belong to `new`."""
        self.owner = [new if o == old else o for o in self.owner]

    def load(self, rows):
        for vid, emb in rows:
            self.add(vid, emb, force=True)
