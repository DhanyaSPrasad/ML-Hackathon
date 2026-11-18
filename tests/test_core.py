import unittest
import numpy as np
import pandas as pd
from pathsnap.core import make_network, candidate_rows, viterbi


class MapMatchingCoreTests(unittest.TestCase):
    def test_exact_segment_ranking_finds_long_segment(self):
        # The long horizontal segment has a distant midpoint but passes through
        # the GPS fix. Exact in-radius ranking should select it.
        nodes = pd.DataFrame({
            "lon": [-0.01, 0.01, 0.0001, 0.0002],
            "lat": [0.0, 0.0, 0.0001, 0.0001],
        })
        arcs = pd.DataFrame({"from_node": [0, 2], "to_node": [1, 3]})
        track = pd.DataFrame({"lon": [0.0], "lat": [0.0], "timestamp": [1.0]})
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


if __name__ == "__main__":
    unittest.main()
