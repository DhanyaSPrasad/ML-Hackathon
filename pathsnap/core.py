"""PathSnap data loading, candidate generation, route alignment and decoding."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Keep this schema in sync with model_metadata.json. Angles are represented
# both as differences and sine/cosine pairs to avoid the 0/360 discontinuity.
FEATURES = [
    "distance_m", "bearing_diff", "road_length_m",
    "gps_speed_mps", "time_gap_s", "heading_change", "candidate_rank",
    "projection_fraction", "segment_alignment", "gps_displacement_m",
    "road_bearing_sin", "road_bearing_cos", "gps_heading_sin",
    "gps_heading_cos", "distance_to_start_m", "distance_to_end_m",
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
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def preprocess_track(track):
    clean = track.copy()
    for col in ("lon", "lat", "timestamp"):
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    valid = (np.isfinite(clean[["lon", "lat", "timestamp"]]).all(axis=1)
             & clean.lon.between(-180, 180) & clean.lat.between(-90, 90))
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
        route = (pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"])["arc_id"].astype(int).to_numpy()
                 if route_path.exists() else np.asarray([], dtype=int))
    return track, nodes, arcs, route


def _local_xy(lon, lat, lon0, lat0):
    sx = 111_320.0 * np.cos(np.radians(lat0))
    sy = 110_540.0
    return np.column_stack(((np.asarray(lon) - lon0) * sx, (np.asarray(lat) - lat0) * sy))


def _point_segment_projection(points, starts, ends):
    vec = ends - starts
    denom = np.maximum(np.einsum("ij,ij->i", vec, vec), 1e-12)
    rel = points[:, None, :] - starts[None, :, :]
    frac = np.clip(np.einsum("nmi,mi->nm", rel, vec) / denom[None, :], 0.0, 1.0)
    projected = starts[None, :, :] + frac[:, :, None] * vec[None, :, :]
    return np.linalg.norm(points[:, None, :] - projected, axis=2), frac


def make_network(nodes, arcs):
    if len(nodes) == 0 or len(arcs) == 0:
        raise ValueError("Road network must contain at least one node and one arc")
    ij_all = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
    valid = ((ij_all[:, 0] >= 0) & (ij_all[:, 1] >= 0)
             & (ij_all[:, 0] < len(nodes)) & (ij_all[:, 1] < len(nodes)))
    arc_ids, ij = np.flatnonzero(valid), ij_all[valid]
    if len(ij) == 0:
        raise ValueError("Road network contains no valid directed arcs")
    a = nodes.iloc[ij[:, 0]][["lon", "lat"]].to_numpy(dtype=float)
    b = nodes.iloc[ij[:, 1]][["lon", "lat"]].to_numpy(dtype=float)
    lon0, lat0 = float(nodes.lon.mean()), float(nodes.lat.mean())
    a_xy, b_xy = _local_xy(a[:, 0], a[:, 1], lon0, lat0), _local_xy(b[:, 0], b[:, 1], lon0, lat0)
    mid_xy = (a_xy + b_xy) / 2.0
    return {"ij": ij, "ids": arc_ids, "a": a, "b": b, "a_xy": a_xy,
            "b_xy": b_xy, "mid_xy": mid_xy, "tree": cKDTree(mid_xy),
            "lengths": haversine(a[:, 0], a[:, 1], b[:, 0], b[:, 1]),
            "bearings": bearing(a[:, 0], a[:, 1], b[:, 0], b[:, 1]),
            "lon0": lon0, "lat0": lat0}


def _sequence_route_targets(track, net, route, gps_xy, gps_heading):
    """Monotonic dynamic-programming alignment to the ordered reference route.

    Repeated arc IDs remain distinct route positions. The prefix-minimum
    recurrence permits route segments to be skipped between sparse GPS fixes,
    while preventing the target from moving backwards through the route.
    """
    valid_ids = set(net["ids"].tolist())
    route_ids = np.asarray([int(a) for a in route if int(a) in valid_ids], dtype=int)
    if not len(route_ids):
        return np.full(len(track), -1, dtype=int)
    id_to_pos = {int(aid): pos for pos, aid in enumerate(net["ids"])}
    route_pos = np.asarray([id_to_pos[int(aid)] for aid in route_ids], dtype=int)
    distances, fractions = _point_segment_projection(
        gps_xy, net["a_xy"][route_pos], net["b_xy"][route_pos])
    # Heading is weak evidence: stationary/near-stationary GPS fixes have no
    # reliable heading and therefore receive no directional cost.
    route_bearings = net["bearings"][route_pos]
    heading_cost = angular_difference(gps_heading[:, None], route_bearings[None, :])
    moving = np.asarray(track.timestamp.diff().fillna(0).to_numpy() > 0)
    moving &= np.asarray(haversine(
        track.lon.shift(1).fillna(track.lon.iloc[0]).to_numpy(),
        track.lat.shift(1).fillna(track.lat.iloc[0]).to_numpy(),
        track.lon.to_numpy(), track.lat.to_numpy()) > 2.0)
    emission = distances + np.where(moving[:, None], 0.12 * heading_cost, 0.0)
    n_obs, n_route = emission.shape
    if n_route == 1:
        return np.full(n_obs, route_ids[0], dtype=int)
    skip_cost = 0.25
    dp = emission[0].astype(float)
    back = np.zeros((n_obs, n_route), dtype=np.int32)
    indices = np.arange(n_route, dtype=float)
    for t in range(1, n_obs):
        # min_{p <= j} dp[p] + skip_cost * (j-p)
        adjusted = dp - skip_cost * indices
        best_prefix = np.minimum.accumulate(adjusted)
        best_index = np.zeros(n_route, dtype=np.int32)
        best_value = np.empty(n_route, dtype=float)
        running_value, running_index = np.inf, 0
        for j in range(n_route):
            value = adjusted[j]
            if value < running_value:
                running_value, running_index = value, j
            best_value[j] = running_value + skip_cost * j
            best_index[j] = running_index
        dp = best_value + emission[t]
        back[t] = best_index
    state = int(np.argmin(dp))
    positions = np.empty(n_obs, dtype=np.int32)
    for t in range(n_obs - 1, -1, -1):
        positions[t] = state
        if t:
            state = int(back[t, state])
    return route_ids[positions]


def candidate_rows(track, net, route=None, k=8, radius_m=180, candidate_pool=None):
    """Build exact point-to-segment-ranked candidates and optional weak labels.

    Midpoint indexing is only a broad-phase shortlist. Exact segment distances
    determine the final candidate order. A larger pool reduces the chance that
    a long segment is discarded because its midpoint is far from the GPS fix.
    """
    n = len(track)
    if k < 1:
        raise ValueError("k must be at least 1")
    coords = track[["lon", "lat"]].to_numpy(dtype=float)
    gps_xy = _local_xy(coords[:, 0], coords[:, 1], net["lon0"], net["lat0"])
    kk = min(int(k), len(net["ids"]))
    pool = min(len(net["ids"]), max(int(candidate_pool or max(kk * 8, 64)), kk))
    _, midpoint_idx = net["tree"].query(gps_xy, k=pool)
    if pool == 1:
        midpoint_idx = midpoint_idx[:, None]
    # Any segment within radius_m has its midpoint within radius_m plus
    # half the segment length. Use this bound for a complete in-radius shortlist.
    max_half_length = float(np.max(net["lengths"]) / 2.0) if len(net["lengths"]) else 0.0
    spatial_radius = max(0.0, float(radius_m)) + max_half_length

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

    rows, candidate_ids, candidate_dists, candidate_fracs = [], [], [], []
    for i in range(n):
        spatial = np.asarray(net["tree"].query_ball_point(gps_xy[i], r=spatial_radius), dtype=int)
        fallback = np.asarray(midpoint_idx[i], dtype=int).reshape(-1)
        selected = np.unique(np.concatenate((spatial, fallback)))
        distances, fractions = _point_segment_projection(
            gps_xy[i:i+1], net["a_xy"][selected], net["b_xy"][selected])
        d, f = distances[0], fractions[0]
        # Exact geometric ranking, with stable tie-breaking by original arc ID.
        order = np.lexsort((net["ids"][selected], d))[:kk]
        selected, d, f = selected[order], d[order], f[order]
        for rank, (j, dist, frac) in enumerate(zip(selected, d, f)):
            road_bearing = float(net["bearings"][j])
            delta = float(angular_difference(road_bearing, gps_heading[i]))
            rows.append([
                float(dist), delta, float(net["lengths"][j]),
                float(speed[i]), float(gaps[i]), float(heading_change[i]),
                float(rank), float(frac), float(np.cos(np.radians(delta))),
                float(displacement[i]), float(np.sin(np.radians(road_bearing))),
                float(np.cos(np.radians(road_bearing))),
                float(np.sin(np.radians(gps_heading[i]))),
                float(np.cos(np.radians(gps_heading[i]))),
                float(np.linalg.norm(gps_xy[i] - net["a_xy"][j])),
                float(np.linalg.norm(gps_xy[i] - net["b_xy"][j])),
            ])
            candidate_ids.append(int(net["ids"][j]))
            candidate_dists.append(float(dist))
            candidate_fracs.append(float(frac))

    X = pd.DataFrame(rows, columns=FEATURES)
    candidate_ids = np.asarray(candidate_ids, dtype=int)
    candidate_dists = np.asarray(candidate_dists, dtype=float)
    candidate_fracs = np.asarray(candidate_fracs, dtype=float)
    target = np.full(n, -1, dtype=int)
    y = None
    if route is not None and len(route):
        target = _sequence_route_targets(track, net, route, gps_xy, gps_heading)
        y = np.asarray([
            int(aid == target[i])
            for i in range(n)
            for aid in candidate_ids[i * kk:(i + 1) * kk]
        ], dtype=int)
    return X, y, candidate_ids, candidate_dists, target


def viterbi(candidate_ids, distances, scores, arcs, k=8, transition_penalty=3.0,
            time_gaps=None, gap_reset_s=5.0, projection_fractions=None,
            arc_lengths=None, gps_displacements=None, speed_weight=0.015,
            max_speed_mps=55.0, distance_scale_m=80.0):
    """Gap-aware Viterbi with directed connectivity and movement feasibility.

    Emissions combine classifier log-scores and geometric distance. Connected
    transitions receive a soft movement-consistency cost based on the distance
    implied by the two segment projections and the observed GPS displacement.
    """
    candidate_ids = np.asarray(candidate_ids, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return np.asarray([], dtype=int)
    if k < 1 or len(candidate_ids) != len(scores) or len(scores) % k:
        raise ValueError("Candidate arrays must contain exactly k states per observation")
    n = len(scores) // k
    cand, dist, prob = candidate_ids.reshape(n, k), np.asarray(distances, dtype=float).reshape(n, k), scores.reshape(n, k)
    edge = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
    gaps = np.ones(n, dtype=float) if time_gaps is None else np.asarray(time_gaps, dtype=float)
    if len(gaps) == n - 1:
        gaps = np.r_[0.0, gaps]
    if len(gaps) != n:
        raise ValueError("time_gaps must have n or n-1 entries")
    frac = np.full((n, k), 0.5) if projection_fractions is None else np.asarray(projection_fractions, dtype=float).reshape(n, k)
    lengths = np.zeros(len(arcs), dtype=float) if arc_lengths is None else np.asarray(arc_lengths, dtype=float)
    gps_move = np.zeros(n, dtype=float) if gps_displacements is None else np.asarray(gps_displacements, dtype=float)
    dp = np.log(np.maximum(prob[0], 1e-9)) - dist[0] / max(float(distance_scale_m), 1e-6)
    back = np.zeros((n, k), dtype=np.int32)
    for t in range(1, n):
        reset = gaps[t] > gap_reset_s or gaps[t] <= 0
        for j, aid in enumerate(cand[t]):
            if aid < 0 or aid >= len(edge):
                if j == 0:
                    next_row = np.full(k, -1e12, dtype=float)
                continue
            current_from = edge[aid, 0]
            best_value, best_prev = -np.inf, 0
            for q, prev in enumerate(cand[t-1]):
                if prev < 0 or prev >= len(edge):
                    continue
                same_arc = prev == aid
                connected = same_arc or edge[prev, 1] == current_from
                if reset:
                    transition = 0.0
                elif not connected:
                    transition = -transition_penalty
                else:
                    transition = 0.0
                    if len(lengths) > max(prev, aid):
                        if same_arc:
                            implied = max(0.0, lengths[aid] * (frac[t, j] - frac[t-1, q]))
                        else:
                            implied = max(0.0, lengths[prev] * (1.0 - frac[t-1, q])) + max(0.0, lengths[aid] * frac[t, j])
                        observed = gps_move[t]
                        allowance = max(12.0, max_speed_mps * max(gaps[t], 0.0))
                        excess = max(0.0, implied - observed - allowance)
                        transition -= speed_weight * abs(implied - observed)
                        transition -= 0.08 * excess
                value = dp[t-1, q] + transition
                if value > best_value:
                    best_value, best_prev = value, q
            dp_current = best_value + np.log(max(prob[t, j], 1e-9)) - dist[t, j] / max(float(distance_scale_m), 1e-6)
            if j == 0:
                next_row = np.full(k, -1e12, dtype=float)
            next_row[j] = dp_current
            back[t, j] = best_prev
        if np.all(next_row <= -1e11):
            next_row = np.log(np.maximum(prob[t], 1e-9)) - dist[t] / max(float(distance_scale_m), 1e-6)
        dp = next_row
    state = int(np.argmax(dp))
    out = np.zeros(n, dtype=int)
    for t in range(n - 1, -1, -1):
        out[t] = cand[t, state]
        if t:
            state = int(back[t, state])
    return out


def trip_features(folder, k=8, max_observations=None, candidate_pool=None):
    track, nodes, arcs, route = read_trip(folder, include_route=True)
    # Never downsample before deriving temporal features: this preserves the
    # exact same adjacent-fix semantics in training, validation and inference.
    if max_observations and len(track) > max_observations:
        raise ValueError("Observation subsampling is disabled; train and infer on the same full-resolution track")
    net = make_network(nodes, arcs)
    X, y, cids, dists, target = candidate_rows(
        track, net, route, k=k, candidate_pool=candidate_pool)
    return track, nodes, arcs, route, net, X, y, cids, dists, target
