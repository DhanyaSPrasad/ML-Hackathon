"""PathSnap data loading, preprocessing, candidate features and route decoding.

Coordinates are converted to a local metric plane for spatial indexing and
point-to-segment calculations. The reference route is read only by offline
training/evaluation code; prediction callers should use read_trip(...,
include_route=False) and candidate_rows(..., route=None).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FEATURES = [
    "distance_m", "bearing_diff", "road_length_m", "road_bearing",
    "gps_speed_mps", "time_gap_s", "heading_change", "candidate_rank",
    "projection_fraction", "segment_alignment", "gps_displacement_m",
]
EARTH_RADIUS_M = 6_371_000.0


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.asarray, (lon1, lat1, lon2, lat2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def angular_difference(a, b):
    """Smallest absolute angular difference in degrees."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def preprocess_track(track):
    """Coerce GPS fields, remove invalid rows and restore chronological order.

    No interpolation is applied: fabricated positions would create artificial
    training observations. The returned frame retains original values except
    for numeric coercion, invalid-row removal and stable timestamp sorting.
    """
    clean = track.copy()
    for col in ("lon", "lat", "timestamp"):
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    valid = (
        np.isfinite(clean[["lon", "lat", "timestamp"]]).all(axis=1)
        & clean.lon.between(-180, 180)
        & clean.lat.between(-90, 90)
    )
    clean = clean.loc[valid].sort_values("timestamp", kind="stable").reset_index(drop=True)
    if clean.empty:
        raise ValueError("Trip has no valid GPS observations after preprocessing")
    return clean


def read_trip(folder, include_route=True):
    folder = Path(folder)
    tid = folder.name

    def read(name, cols, dtype=None):
        path = folder / f"{tid}.{name}"
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path, sep=r"\s+", header=None, names=cols, dtype=dtype, engine="python")

    track = preprocess_track(read("track", ["lon", "lat", "timestamp"]))
    nodes = read("nodes", ["lon", "lat"])
    arcs = read("arcs", ["from_node", "to_node"], dtype="int64")
    route = None
    if include_route:
        route_path = folder / f"{tid}.route"
        if route_path.exists():
            route = pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"])["arc_id"].astype(int).to_numpy()
        else:
            route = np.asarray([], dtype=int)
    return track, nodes, arcs, route


def _local_xy(lon, lat, lon0, lat0):
    """Equirectangular local projection suitable for a trip-sized region."""
    sx = 111_320.0 * np.cos(np.radians(lat0))
    sy = 110_540.0
    return np.column_stack(((np.asarray(lon) - lon0) * sx, (np.asarray(lat) - lat0) * sy))


def _point_segment_projection(points, starts, ends):
    """Return point-to-segment distances and clamped projection fractions."""
    vec = ends - starts
    denom = np.einsum("ij,ij->i", vec, vec)
    denom = np.maximum(denom, 1e-12)
    rel = points[:, None, :] - starts[None, :, :]
    frac = np.einsum("nmi,mi->nm", rel, vec) / denom[None, :]
    frac = np.clip(frac, 0.0, 1.0)
    projected = starts[None, :, :] + frac[:, :, None] * vec[None, :, :]
    distances = np.linalg.norm(points[:, None, :] - projected, axis=2)
    return distances, frac


def make_network(nodes, arcs):
    if len(nodes) == 0 or len(arcs) == 0:
        raise ValueError("Road network must contain at least one node and one arc")
    ij_all = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
    valid = (ij_all[:, 0] >= 0) & (ij_all[:, 1] >= 0) & (ij_all[:, 0] < len(nodes)) & (ij_all[:, 1] < len(nodes))
    arc_ids = np.flatnonzero(valid)
    ij = ij_all[valid]
    if len(ij) == 0:
        raise ValueError("Road network contains no valid directed arcs")
    a = nodes.iloc[ij[:, 0]][["lon", "lat"]].to_numpy(dtype=float)
    b = nodes.iloc[ij[:, 1]][["lon", "lat"]].to_numpy(dtype=float)
    lon0 = float(nodes.lon.mean()); lat0 = float(nodes.lat.mean())
    a_xy = _local_xy(a[:, 0], a[:, 1], lon0, lat0)
    b_xy = _local_xy(b[:, 0], b[:, 1], lon0, lat0)
    mid_xy = (a_xy + b_xy) / 2.0
    tree = cKDTree(mid_xy)
    lengths = haversine(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
    bearings = bearing(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
    return {"ij": ij, "ids": arc_ids, "a": a, "b": b, "a_xy": a_xy,
            "b_xy": b_xy, "mid_xy": mid_xy, "tree": tree,
            "lengths": lengths, "bearings": bearings,
            "lon0": lon0, "lat0": lat0}


def candidate_rows(track, net, route=None, k=8, radius_m=180):
    """Generate nearest directed-arc candidates and engineered features.

    `route` is optional and is used only to derive offline weak labels. When
    omitted, no reference labels are computed and y/target are None/-1.
    """
    n = len(track)
    coords = track[["lon", "lat"]].to_numpy(dtype=float)
    gps_xy = _local_xy(coords[:, 0], coords[:, 1], net["lon0"], net["lat0"])
    kk = min(int(k), len(net["ids"]))
    midpoint_dist, midpoint_idx = net["tree"].query(gps_xy, k=kk)
    if kk == 1:
        midpoint_dist = midpoint_dist[:, None]
        midpoint_idx = midpoint_idx[:, None]

    gaps = np.ones(n, dtype=float)
    speed = np.zeros(n, dtype=float)
    displacement = np.zeros(n, dtype=float)
    gps_heading = np.zeros(n, dtype=float)
    if n > 1:
        raw_gaps = np.diff(track.timestamp.to_numpy(dtype=float))
        gaps[1:] = np.maximum(raw_gaps, 0.01)
        displacement[1:] = haversine(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1])
        speed[1:] = displacement[1:] / gaps[1:]
        gps_heading[1:] = bearing(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1])
    heading_change = np.zeros(n, dtype=float)
    if n > 2:
        heading_change[2:] = angular_difference(gps_heading[2:], gps_heading[1:-1])

    # Calculate actual point-to-segment distance for each selected candidate.
    rows, candidate_ids, candidate_dists = [], [], []
    candidate_groups = []
    for i in range(n):
        local_order = np.argsort(midpoint_dist[i], kind="stable")
        selected = midpoint_idx[i, local_order]
        starts = net["a_xy"][selected]
        ends = net["b_xy"][selected]
        p = np.repeat(gps_xy[i:i+1], len(selected), axis=0)
        vec = ends - starts
        denom = np.maximum(np.einsum("ij,ij->i", vec, vec), 1e-12)
        frac = np.clip(np.einsum("ij,ij->i", p - starts, vec) / denom, 0.0, 1.0)
        projected = starts + frac[:, None] * vec
        cross_track = np.linalg.norm(p - projected, axis=1)
        for rank, j in enumerate(selected):
            road_bearing = float(net["bearings"][j])
            bearing_delta = float(angular_difference(road_bearing, gps_heading[i]))
            rows.append([
                float(cross_track[rank]), bearing_delta,
                float(net["lengths"][j]), road_bearing, float(speed[i]), float(gaps[i]),
                float(heading_change[i]), float(rank), float(frac[rank]),
                float(np.cos(np.radians(bearing_delta))), float(displacement[i]),
            ])
            candidate_ids.append(int(net["ids"][j]))
            candidate_dists.append(float(cross_track[rank]))
        candidate_groups.append([int(net["ids"][j]) for j in selected])

    X = pd.DataFrame(rows, columns=FEATURES)
    candidate_ids = np.asarray(candidate_ids, dtype=int)
    candidate_dists = np.asarray(candidate_dists, dtype=float)
    target = np.full(n, -1, dtype=int)
    y = None

    if route is not None and len(route):
        valid_arc_ids = set(net["ids"].tolist())
        route_ids = np.asarray([int(aid) for aid in route if int(aid) in valid_arc_ids], dtype=int)
        if len(route_ids):
            id_to_pos = {int(aid): pos for pos, aid in enumerate(net["ids"])}
            route_pos = np.asarray([id_to_pos[int(aid)] for aid in np.unique(route_ids)], dtype=int)
            # Use a route-only midpoint index to shortlist nearby reference arcs,
            # then compute exact point-to-segment distance on that small shortlist.
            route_mid = net["mid_xy"][route_pos]
            route_tree = cKDTree(route_mid)
            route_k = min(8, len(route_pos))
            _, route_neighbours = route_tree.query(gps_xy, k=route_k)
            if route_k == 1:
                route_neighbours = route_neighbours[:, None]
            target_pos = np.empty(n, dtype=int)
            for point_i in range(n):
                shortlist = route_pos[np.asarray(route_neighbours[point_i], dtype=int)]
                starts, ends = net["a_xy"][shortlist], net["b_xy"][shortlist]
                vec = ends - starts
                denom = np.maximum(np.einsum("ij,ij->i", vec, vec), 1e-12)
                frac = np.clip(np.einsum("ij,ij->i", gps_xy[point_i] - starts, vec) / denom, 0.0, 1.0)
                projected = starts + frac[:, None] * vec
                d = np.linalg.norm(gps_xy[point_i] - projected, axis=1)
                target_pos[point_i] = shortlist[int(np.argmin(d))]
            target = net["ids"][target_pos]
            y = np.asarray([int(aid == target[i]) for i, group in enumerate(candidate_groups) for aid in group], dtype=int)
    return X, y, candidate_ids, candidate_dists, target


def viterbi(candidate_ids, distances, scores, arcs, k=8, transition_penalty=3.0,
            time_gaps=None, gap_reset_s=5.0):
    """Viterbi decoder with a transition reset across long GPS time gaps."""
    candidate_ids = np.asarray(candidate_ids, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return np.asarray([], dtype=int)
    n = len(scores) // k
    if n == 0 or len(candidate_ids) != n * k:
        raise ValueError("Candidate arrays must contain exactly k states per observation")
    cand = candidate_ids.reshape(n, k)
    dist = np.asarray(distances, dtype=float).reshape(n, k)
    prob = np.asarray(scores, dtype=float).reshape(n, k)
    edge = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
    gaps = np.ones(n, dtype=float) if time_gaps is None else np.asarray(time_gaps, dtype=float)
    if len(gaps) == n - 1:
        gaps = np.r_[0.0, gaps]
    if len(gaps) != n:
        raise ValueError("time_gaps must have n or n-1 entries")

    dp = np.full((n, k), -1e12, dtype=float)
    back = np.zeros((n, k), dtype=int)
    dp[0] = np.log(np.maximum(prob[0], 1e-9)) - dist[0] / 80.0
    for t in range(1, n):
        reset = gaps[t] > gap_reset_s
        for j, aid in enumerate(cand[t]):
            if aid < 0 or aid >= len(edge):
                continue
            frm = edge[aid, 0]
            best_val = -1e12; best_prev = 0
            for q, prev in enumerate(cand[t - 1]):
                if prev < 0 or prev >= len(edge):
                    continue
                connected = edge[prev, 1] == frm
                trans = 0.0 if (connected or reset) else -transition_penalty
                val = dp[t - 1, q] + trans
                if val > best_val:
                    best_val, best_prev = val, q
            dp[t, j] = best_val + np.log(max(prob[t, j], 1e-9)) - dist[t, j] / 80.0
            back[t, j] = best_prev
        if np.all(dp[t] <= -1e11):
            dp[t] = np.log(np.maximum(prob[t], 1e-9)) - dist[t] / 80.0
    state = int(np.argmax(dp[-1])); out = np.zeros(n, dtype=int)
    for t in range(n - 1, -1, -1):
        out[t] = cand[t, state]
        if t:
            state = back[t, state]
    return out


def trip_features(folder, k=8, max_observations=None):
    track, nodes, arcs, route = read_trip(folder, include_route=True)
    if max_observations and len(track) > max_observations:
        keep = np.linspace(0, len(track) - 1, max_observations).astype(int)
        track = track.iloc[keep].reset_index(drop=True)
    net = make_network(nodes, arcs)
    X, y, cids, dists, target = candidate_rows(track, net, route, k=k)
    return track, nodes, arcs, route, net, X, y, cids, dists, target
