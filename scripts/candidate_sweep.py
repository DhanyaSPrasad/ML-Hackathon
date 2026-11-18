"""Training-only candidate coverage sweep; never reads student/evaluator test labels."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from pathsnap.core import read_trip, make_network, candidate_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Student folder"
OUT = ROOT / "pathsnap" / "results"
RADII_M = (20, 30, 50, 75, 100, 150)
K_VALUES = (4, 8, 12, 16)
POOL = 128


def main():
    base = DATA / "train"
    if not base.is_dir():
        base = DATA / "train" / "train"
    folders = [p for p in sorted(base.iterdir())
               if p.is_dir() and (p / f"{p.name}.track").is_file()]
    if not folders:
        raise RuntimeError("No training trips found")
    rows = []
    for radius in RADII_M:
        for k in K_VALUES:
            valid = hits = observations = 0
            for folder in folders:
                track, nodes, arcs, route = read_trip(folder, include_route=True)
                net = make_network(nodes, arcs)
                _, _, cids, _, target = candidate_rows(
                    track, net, route=route, k=k, radius_m=radius,
                    candidate_pool=max(POOL, 8 * k))
                kk = min(k, len(net["ids"]))
                cand = cids.reshape(len(track), kk)
                mask = target >= 0
                valid += int(mask.sum())
                hits += int(((cand == target[:, None]).any(axis=1) & mask).sum())
                observations += len(track)
            rows.append({
                "radius_m": radius, "k_candidates": k,
                "training_observations": observations,
                "labelled_observations": valid,
                "candidate_coverage": hits / max(valid, 1),
                "candidate_miss_rate": 1.0 - hits / max(valid, 1),
            })
            print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values(
        ["candidate_coverage", "k_candidates"], ascending=[False, True])
    result.to_csv(OUT / "candidate_sweep.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
