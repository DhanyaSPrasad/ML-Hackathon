import unittest
import numpy as np
import pandas as pd
from pathsnap.core import make_network, candidate_rows, viterbi, FEATURES
from pathsnap.artifacts import validate_metadata
from pathsnap.modeling import make_models


class MapMatchingCoreTests(unittest.TestCase):
    def test_exact_segment_ranking_finds_long_segment(self):
        # The long horizontal segment has a distant midpoint but passes through
        # the GPS fix. Exact in-radius ranking should select it.
        nodes = pd.DataFrame({
            "lon": [-0.01, 0.01, -0.0091, -0.0089],
            "lat": [0.0, 0.0, 0.0001, 0.0001],
        })
        arcs = pd.DataFrame({"from_node": [0, 2], "to_node": [1, 3]})
        track = pd.DataFrame({"lon": [-0.009], "lat": [0.0], "timestamp": [1.0]})
        net = make_network(nodes, arcs)
        _, _, candidate_ids, _, _ = candidate_rows(
            track, net, route=None, k=1, candidate_pool=1
        )
        self.assertEqual(int(candidate_ids[0]), 0)

    def test_viterbi_allows_repeated_same_arc(self):
        arcs = pd.DataFrame({"from_node": [0, 1], "to_node": [1, 2]})
        predicted = viterbi(
            candidate_ids=np.array([0, 1, 0, 1]),
            distances=np.array([1.0, 3.0, 1.0, 3.0]),
            scores=np.array([0.9, 0.1, 0.9, 0.1]),
            arcs=arcs,
            k=2,
            time_gaps=np.array([0.0, 1.0]),
            transition_penalty=5.0,
        )
        self.assertEqual(predicted.tolist(), [0, 0])

    def test_current_artifact_metadata_is_accepted(self):
        metadata = {"features": list(FEATURES), "observations_per_trip": "full resolution",
                    "k_candidates": 8, "candidate_pool": 64, "candidate_radius_m": 50}
        self.assertIs(validate_metadata(metadata), metadata)

    def test_stale_feature_metadata_is_rejected(self):
        metadata = {"features": ["old_feature"], "observations_per_trip": 120,
                    "k_candidates": 8, "candidate_pool": 64, "candidate_radius_m": 50}
        with self.assertRaisesRegex(ValueError, "feature schema is stale"):
            validate_metadata(metadata)

    def test_feature_rows_match_declared_schema(self):
        nodes = pd.DataFrame({"lon": [0.0, 0.001], "lat": [0.0, 0.0]})
        arcs = pd.DataFrame({"from_node": [0], "to_node": [1]})
        track = pd.DataFrame({"lon": [0.0002, 0.0004], "lat": [0.0, 0.0],
                              "timestamp": [1.0, 2.0]})
        net = make_network(nodes, arcs)
        X, _, candidate_ids, _, _ = candidate_rows(track, net, route=None, k=1)
        self.assertEqual(X.shape, (2, len(X.columns)))
        self.assertEqual(len(candidate_ids), 2)
        self.assertTrue(np.isfinite(X.to_numpy()).all())

    def test_candidate_context_features_are_present_and_finite(self):
        nodes = pd.DataFrame({"lon": [0.0, 0.001, 0.002],
                              "lat": [0.0, 0.0, 0.0]})
        arcs = pd.DataFrame({"from_node": [0, 1], "to_node": [1, 2]})
        track = pd.DataFrame({"lon": [0.0002, 0.0008, 0.0014],
                              "lat": [0.00001, 0.00001, 0.00001],
                              "timestamp": [1.0, 2.0, 3.0]})
        net = make_network(nodes, arcs)
        X, _, candidate_ids, _, _ = candidate_rows(
            track, net, route=None, k=2, radius_m=50, candidate_pool=2)
        self.assertEqual(X.shape, (len(track) * 2, len(FEATURES)))
        for name in ("distance_minus_min_m", "candidate_count", "dist_prev_pt",
                     "dist_next_pt", "turn_rate_rad_s", "previous_candidate_m",
                     "next_candidate_m"):
            self.assertIn(name, X.columns)
        self.assertTrue(np.isfinite(X.to_numpy()).all())
        self.assertEqual(len(candidate_ids), len(track) * 2)

    def test_viterbi_uses_directed_network_transition_without_index_error(self):
        arcs = pd.DataFrame({"from_node": [0, 1, 5], "to_node": [1, 2, 6]})
        predicted = viterbi(
            candidate_ids=np.array([0, 2, 0, 2]),
            distances=np.array([1.0, 4.0, 4.0, 1.0]),
            scores=np.array([0.9, 0.1, 0.1, 0.9]),
            arcs=arcs, k=2,
            time_gaps=np.array([0.0, 1.0]),
            projection_fractions=np.array([0.2, 0.2, 0.8, 0.8]),
            arc_lengths=np.array([100.0, 100.0, 100.0]),
            gps_displacements=np.array([0.0, 100.0]),
            beta=100.0, uturn_penalty=10.0,
        )
        self.assertEqual(predicted.shape, (2,))
        self.assertTrue(set(predicted.tolist()).issubset({0, 2}))

    def test_model_comparison_replaces_knn_with_lightgbm(self):
        models = make_models()
        self.assertIn("LightGBM", models)
        self.assertNotIn("K-Nearest Neighbours", models)
        self.assertEqual(len(models), 5)


if __name__ == "__main__":
    unittest.main()
