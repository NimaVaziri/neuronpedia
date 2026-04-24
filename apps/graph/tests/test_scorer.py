import json
import math
import unittest
from pathlib import Path

from neuronpedia_graph.scorer import BatchGraphScorer, prune_graph_for_scoring


GOLDEN_SCORING_PATH = Path(__file__).parent / "fixtures" / "scoring_golden.json"


def load_golden_scoring_fixture():
    with GOLDEN_SCORING_PATH.open() as file:
        return json.load(file)


def node(node_id, feature_type, **overrides):
    base = {
        "node_id": node_id,
        "feature_type": feature_type,
        "layer": "0",
        "ctx_idx": 0,
        "feature": 0,
        "token_prob": 0.0,
        "influence": 0.0,
        "clerp": "",
    }
    base.update(overrides)
    return base


def scoring_graph(object_endpoints=False):
    nodes = [
        node("embed_0", "embedding", layer="E", clerp="Dallas"),
        node("0_10", "cross layer transcoder", feature=10, influence=0.8),
        node("0_11", "cross layer transcoder", feature=11, influence=0.4),
        node("err_0", "mlp reconstruction error"),
        node("logit_target", "logit", token_prob=0.7, clerp=" Austin"),
        node("logit_other", "logit", token_prob=0.0, clerp=" Houston"),
    ]
    links = [
        {"source": "embed_0", "target": "0_10", "weight": 0.5},
        {"source": "embed_0", "target": "0_11", "weight": 0.2},
        {"source": "0_10", "target": "logit_target", "weight": 0.8},
        {"source": "0_11", "target": "logit_target", "weight": 0.3},
        {"source": "0_11", "target": "err_0", "weight": 0.4},
        {"source": "err_0", "target": "logit_other", "weight": 0.1},
    ]
    if object_endpoints:
        links = [
            {
                **link,
                "source": {"node_id": link["source"]},
                "target": {"node_id": link["target"]},
            }
            for link in links
        ]
    return nodes, links


class ScorerTests(unittest.TestCase):
    def assert_scores_close(self, left, right):
        self.assertTrue(math.isfinite(left["replacementScore"]))
        self.assertTrue(math.isfinite(left["completenessScore"]))
        self.assertAlmostEqual(left["replacementScore"], right["replacementScore"], places=6)
        self.assertAlmostEqual(left["completenessScore"], right["completenessScore"], places=6)

    def test_score_batch_matches_individual_score_calls(self):
        nodes, links = scoring_graph()
        scorer = BatchGraphScorer(nodes, links, device_str="cpu")
        pin_sets = [[], ["0_10"], ["0_11"], ["0_10", "0_11"]]

        batch_scores = scorer.score_batch(pin_sets)

        self.assertEqual(len(batch_scores), len(pin_sets))
        for pins, batch_score in zip(pin_sets, batch_scores):
            self.assert_scores_close(batch_score, scorer.score_fast(pins))

    def test_golden_scoring_fixture_preserves_scores_and_search_result(self):
        fixture = load_golden_scoring_fixture()
        scorer = BatchGraphScorer(fixture["nodes"], fixture["links"], device_str="cpu")

        pin_sets = [item["pinned_ids"] for item in fixture["expected_scores"]]
        batch_scores = scorer.score_batch(pin_sets)

        for expected, batch_score in zip(fixture["expected_scores"], batch_scores):
            individual_score = scorer.score_fast(expected["pinned_ids"])
            self.assert_scores_close(batch_score, individual_score)
            self.assertAlmostEqual(
                expected["replacementScore"],
                individual_score["replacementScore"],
                places=6,
            )
            self.assertAlmostEqual(
                expected["completenessScore"],
                individual_score["completenessScore"],
                places=6,
            )

        expected_search = fixture["expected_ia_pc"]
        search_result = scorer.build_circuit_ia_pc(
            expected_search["endpoint_ids"],
            max_steps=expected_search["max_steps"],
            pc_passes=expected_search["pc_passes"],
            pc_candidates=expected_search["pc_candidates"],
        )

        for key in ("pinnedIds", "iaFeatures", "pcFeatures", "totalFeatures"):
            self.assertEqual(search_result[key], expected_search[key])
        self.assertAlmostEqual(
            search_result["replacementScore"],
            expected_search["replacementScore"],
            places=6,
        )
        self.assertAlmostEqual(
            search_result["completenessScore"],
            expected_search["completenessScore"],
            places=6,
        )

    def test_unknown_pins_are_ignored(self):
        nodes, links = scoring_graph()
        scorer = BatchGraphScorer(nodes, links, device_str="cpu")

        self.assert_scores_close(scorer.score_fast(["missing-feature"]), scorer.score_fast([]))
        self.assert_scores_close(scorer.score_fast(["0_10", "missing-feature"]), scorer.score_fast(["0_10"]))

    def test_accepts_object_link_endpoints(self):
        string_nodes, string_links = scoring_graph()
        object_nodes, object_links = scoring_graph(object_endpoints=True)

        string_scorer = BatchGraphScorer(string_nodes, string_links, device_str="cpu")
        object_scorer = BatchGraphScorer(object_nodes, object_links, device_str="cpu")

        self.assert_scores_close(object_scorer.score_fast(["0_10"]), string_scorer.score_fast(["0_10"]))

    def test_prune_graph_keeps_non_feature_nodes_and_filters_links(self):
        nodes = [
            node("embed_0", "embedding", layer="E"),
            node("err_0", "mlp reconstruction error"),
            node("logit_target", "logit", token_prob=1.0, clerp=" Austin"),
            node("logit_other", "logit", token_prob=0.0, clerp=" Houston"),
        ]
        links = []
        for idx in range(60):
            feature_id = f"0_{idx}"
            nodes.append(
                node(
                    feature_id,
                    "cross layer transcoder",
                    feature=idx,
                    influence=float(60 - idx),
                )
            )
            weight = 1.0 if idx < 50 else 0.0
            links.append({"source": feature_id, "target": "logit_target", "weight": weight})
            links.append({"source": "embed_0", "target": feature_id, "weight": 0.1})

        pruned_nodes, pruned_links = prune_graph_for_scoring(nodes, links, " Austin", keep_ratio=0.1)
        kept_ids = {n["node_id"] for n in pruned_nodes}

        self.assertLess(len(pruned_nodes), len(nodes))
        self.assertTrue({"embed_0", "err_0", "logit_target", "logit_other"}.issubset(kept_ids))
        self.assertIn("0_0", kept_ids)
        self.assertNotIn("0_59", kept_ids)
        for link in pruned_links:
            self.assertIn(link["source"], kept_ids)
            self.assertIn(link["target"], kept_ids)

    def test_build_circuit_preserves_endpoints_and_reports_feature_counts(self):
        nodes, links = scoring_graph()
        scorer = BatchGraphScorer(nodes, links, device_str="cpu")

        result = scorer.build_circuit_ia_pc(
            ["embed_0", "logit_target"],
            max_steps=5,
            pc_passes=1,
            pc_candidates=5,
        )

        self.assertTrue({"embed_0", "logit_target"}.issubset(set(result["pinnedIds"])))
        self.assertEqual(len(result["pinnedIds"]), len(set(result["pinnedIds"])))
        feature_pins = [pin for pin in result["pinnedIds"] if pin in {"0_10", "0_11"}]
        self.assertEqual(result["totalFeatures"], len(feature_pins))
        self.assertEqual(result["iaFeatures"] + result["pcFeatures"], result["totalFeatures"])
        self.assertTrue(math.isfinite(result["replacementScore"]))
        self.assertTrue(math.isfinite(result["completenessScore"]))


if __name__ == "__main__":
    unittest.main()
