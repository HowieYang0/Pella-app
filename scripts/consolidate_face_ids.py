#!/usr/bin/env python3
"""Consolidate near-duplicate face embeddings under data/face_ids/<person>/.

For each person folder, applies greedy farthest-first selection over the
stored .npy embeddings and drops anything that's too similar to an
already-kept template. Useful after a noisy enrollment burst where ten
near-identical captures of the same pose got saved.

Independent of the live recognition pipeline. Run at any time:

    # Dry-run on the laptop (default — prints what would change):
    python scripts/consolidate_face_ids.py

    # Actually move redundant files into per-person _dropped/ backups:
    python scripts/consolidate_face_ids.py --apply

    # Be more aggressive (lower threshold = drop more):
    python scripts/consolidate_face_ids.py --threshold 0.92 --apply

    # Cap each person at most N templates:
    python scripts/consolidate_face_ids.py --max-keep 5 --apply

    # Pointing at the robot's tree from the laptop (via NFS / sshfs / etc.):
    python scripts/consolidate_face_ids.py --face-ids-dir /mnt/dock/.../data/face_ids

To run directly on the dock instead, copy the script and invoke it there.
Dropped files go into <person>/_dropped/ so nothing is permanently lost;
delete the backups manually once you've confirmed the result.
"""

import argparse
import os
import shutil
import sys

import numpy as np


DEFAULT_FACE_IDS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "face_ids"))


def _load_person_entries(person_dir: str) -> dict:
    """Walk a single person's directory and return {stem: {jpg, npy, emb}}.

    Only entries with BOTH a readable .npy embedding and a paired image
    are considered. Stems without a usable embedding are reported and
    skipped — we never delete a .jpg whose .npy we can't read (might be
    corrupted; safer to leave for manual review).
    """
    entries: dict = {}
    for fname in sorted(os.listdir(person_dir)):
        path = os.path.join(person_dir, fname)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext == ".npy":
            try:
                emb = np.load(path)
            except Exception as e:
                print(f"    warning: failed to load {fname}: {e}")
                continue
            # L2-normalise defensively — files saved before normalisation was
            # standardised would otherwise distort cosine sims.
            norm = float(np.linalg.norm(emb))
            if norm < 1e-6:
                print(f"    warning: zero-norm embedding in {fname}, skipping")
                continue
            entries.setdefault(stem, {})["emb"] = emb / norm
            entries[stem]["npy"] = path
        elif ext in (".jpg", ".jpeg", ".png"):
            entries.setdefault(stem, {})["jpg"] = path

    # Keep only entries that have both an image and an embedding.
    return {stem: e for stem, e in entries.items()
            if "emb" in e and "jpg" in e}


def _greedy_farthest_first(entries: dict, threshold: float,
                           max_keep: int) -> tuple:
    """Return (kept_stems, dropped_stems).

    Greedy farthest-first: start from the first stem (alphabetically),
    then iteratively add the candidate with the MAX cosine distance to
    the already-kept set. Stop when:
      * we've kept max_keep entries, OR
      * the best remaining candidate's max similarity to a kept entry
        already exceeds `threshold` (everything else is even more
        similar — nothing new to add).
    """
    stems = sorted(entries.keys())
    if not stems:
        return [], []
    if len(stems) == 1:
        return stems, []

    kept = [stems[0]]
    while len(kept) < max_keep:
        best_stem = None
        best_max_cos = 1.0
        for stem in stems:
            if stem in kept:
                continue
            emb = entries[stem]["emb"]
            max_cos = max(float(np.dot(emb, entries[k]["emb"]))
                          for k in kept)
            # Lower max_cos = more different from kept set = preferred.
            if max_cos < best_max_cos:
                best_max_cos = max_cos
                best_stem = stem
        if best_stem is None:
            break
        if best_max_cos > threshold:
            # Even the most-different remaining candidate is too similar
            # to something already kept; stop.
            break
        kept.append(best_stem)

    dropped = [s for s in stems if s not in kept]
    return kept, dropped


def _move_to_backup(entries: dict, dropped_stems: list,
                    person_dir: str) -> int:
    """Move dropped jpg/npy pairs into <person_dir>/_dropped/. Returns
    the number of files actually moved."""
    if not dropped_stems:
        return 0
    backup_dir = os.path.join(person_dir, "_dropped")
    os.makedirs(backup_dir, exist_ok=True)
    moved = 0
    for stem in dropped_stems:
        for key in ("jpg", "npy"):
            src = entries[stem].get(key)
            if src and os.path.exists(src):
                dst = os.path.join(backup_dir, os.path.basename(src))
                shutil.move(src, dst)
                moved += 1
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--face-ids-dir", default=DEFAULT_FACE_IDS_DIR,
                        help=f"Path to face_ids tree (default: {DEFAULT_FACE_IDS_DIR})")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Drop a candidate whose max cosine similarity "
                             "to any kept exceeds this (default: 0.95)")
    parser.add_argument("--max-keep", type=int, default=10,
                        help="Hard cap on templates per person (default: 10)")
    parser.add_argument("--apply", action="store_true",
                        help="Move dropped files to <person>/_dropped/. "
                             "Without this flag the script only prints what "
                             "it would do (dry-run).")
    args = parser.parse_args()

    if not os.path.isdir(args.face_ids_dir):
        print(f"ERR: face_ids dir not found: {args.face_ids_dir}",
              file=sys.stderr)
        return 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] consolidate {args.face_ids_dir}")
    print(f"  threshold={args.threshold}  max_keep={args.max_keep}")
    print()

    grand_kept = 0
    grand_dropped = 0
    for name in sorted(os.listdir(args.face_ids_dir)):
        person_dir = os.path.join(args.face_ids_dir, name)
        if not os.path.isdir(person_dir) or name.startswith("_"):
            continue

        entries = _load_person_entries(person_dir)
        if not entries:
            print(f"  {name}: no usable embeddings — skipped")
            continue

        kept, dropped = _greedy_farthest_first(
            entries, args.threshold, args.max_keep)
        grand_kept += len(kept)
        grand_dropped += len(dropped)

        before = len(kept) + len(dropped)
        print(f"  {name:24s}: {before:>3} -> {len(kept):>3}"
              f"  (dropped {len(dropped)})")
        if dropped:
            for stem in dropped:
                jpg_name = os.path.basename(entries[stem].get("jpg", stem))
                print(f"      drop {jpg_name}")
        if args.apply:
            moved = _move_to_backup(entries, dropped, person_dir)
            if moved:
                print(f"      moved {moved} file(s) to "
                      f"{os.path.relpath(os.path.join(person_dir, '_dropped'), args.face_ids_dir)}/")

    print()
    print(f"Total: {grand_kept + grand_dropped} -> {grand_kept}  "
          f"(dropped {grand_dropped})")
    if not args.apply and grand_dropped:
        print()
        print("This was a DRY-RUN. To actually move dropped files into "
              "<person>/_dropped/, re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
