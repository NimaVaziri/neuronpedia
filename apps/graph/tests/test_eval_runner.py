import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval import eval_runner


def eval_graph():
    return {
        "metadata": {"prompt": "<bos>The capital of Texas is"},
        "nodes": [
            {
                "node_id": "embed_0",
                "feature_type": "embedding",
                "layer": "E",
                "ctx_idx": 0,
                "feature": 0,
                "token_prob": 0,
                "clerp": "Texas",
            },
            {
                "node_id": "0_42",
                "feature_type": "cross layer transcoder",
                "layer": "0",
                "ctx_idx": 3,
                "feature": 42,
                "token_prob": 0,
                "clerp": "Austin feature",
            },
            {
                "node_id": "1_7",
                "feature_type": "cross layer transcoder",
                "layer": "1",
                "ctx_idx": 4,
                "feature": 7,
                "token_prob": 0,
                "clerp": "Texas feature",
            },
            {
                "node_id": "logit_austin",
                "feature_type": "logit",
                "layer": "0",
                "ctx_idx": 0,
                "feature": 0,
                "token_prob": 0.8,
                "clerp": " Austin",
            },
            {
                "node_id": "logit_houston",
                "feature_type": "logit",
                "layer": "0",
                "ctx_idx": 0,
                "feature": 0,
                "token_prob": 0.2,
                "clerp": " Houston",
            },
        ],
        "links": [
            {"source": "embed_0", "target": "0_42", "weight": 0.4},
            {"source": "0_42", "target": "logit_austin", "weight": 0.7},
        ],
    }


class FakeScorer:
    calls = []

    def __init__(self, nodes, links, device_str):
        self.nodes = nodes
        self.links = links
        self.device_str = device_str

    def build_circuit_ia_pc(self, endpoints, alpha, max_steps, pc_passes):
        self.__class__.calls.append(
            {
                "nodes": self.nodes,
                "links": self.links,
                "device_str": self.device_str,
                "endpoints": endpoints,
                "alpha": alpha,
                "max_steps": max_steps,
                "pc_passes": pc_passes,
            }
        )
        return {
            "pinnedIds": [*endpoints, "0_42"],
            "iaFeatures": 1,
            "pcFeatures": 0,
            "totalFeatures": 1,
            "replacementScore": 0.4,
            "completenessScore": 0.8,
        }


class EvalRunnerTests(unittest.TestCase):
    def setUp(self):
        FakeScorer.calls = []

    def test_sanitizes_model_ids_and_resolves_model_paths(self):
        self.assertEqual(eval_runner.sanitize_model_id("google/gemma-2-2b"), "gemma-2-2b")
        self.assertEqual(eval_runner.sanitize_model_id("Org/Model v1.5!"), "model-v1-5")

        with patch.object(eval_runner, "MODEL_ID", "Qwen/Qwen3-4B"):
            self.assertTrue(eval_runner.get_model_eval_results_dir().endswith("eval/models/qwen3-4b/results"))
            self.assertTrue(eval_runner.get_model_eval_graphs_dir().endswith("eval/models/qwen3-4b/graphs"))

    def test_resolve_graph_url_honors_prefix_override(self):
        with patch.object(eval_runner, "EVAL_GRAPH_URL_PREFIX", "https://example.test/graphs"):
            self.assertEqual(
                eval_runner.resolve_graph_url("capital-france", "/graph-data/capital-france.json"),
                "https://example.test/graphs/capital-france.json",
            )

        with patch.object(eval_runner, "EVAL_GRAPH_URL_PREFIX", ""):
            self.assertEqual(
                eval_runner.resolve_graph_url("capital-france", "/graph-data/capital-france.json"),
                "/graph-data/capital-france.json",
            )

    def test_load_graph_prefers_local_override_before_fetching(self):
        graph = eval_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "capital-france.json"
            graph_path.write_text(json.dumps(graph))

            with (
                patch.object(eval_runner, "EVAL_GRAPH_DIR", tmpdir),
                patch("eval.eval_runner.urllib.request.urlopen") as urlopen,
            ):
                loaded = eval_runner.load_graph("capital-france", "https://example.test/remote.json")

        self.assertEqual(loaded, graph)
        urlopen.assert_not_called()

    def test_focus_on_target_zeroes_non_target_logits_without_mutating_original(self):
        graph = eval_graph()

        focused = eval_runner.focus_on_target(graph, " Austin")

        original_other = next(n for n in graph["nodes"] if n["node_id"] == "logit_houston")
        focused_other = next(n for n in focused["nodes"] if n["node_id"] == "logit_houston")
        focused_target = next(n for n in focused["nodes"] if n["node_id"] == "logit_austin")
        self.assertEqual(original_other["token_prob"], 0.2)
        self.assertEqual(focused_other["token_prob"], 0)
        self.assertEqual(focused_target["token_prob"], 0.8)

    def test_get_endpoints_returns_embeddings_and_matching_target_logit(self):
        self.assertEqual(eval_runner.get_endpoints(eval_graph(), " Austin"), ["embed_0", "logit_austin"])
        self.assertEqual(eval_runner.get_endpoints(eval_graph(), "missing"), ["embed_0"])

    def test_find_target_prob_returns_highest_matching_probability(self):
        logits = [
            {"top_logits": [{"token": " Austin", "prob": 0.1}]},
            None,
            {"top_logits": [{"token": " Austin", "prob": 0.4}, {"token": " Houston", "prob": 0.5}]},
        ]

        self.assertEqual(eval_runner.find_target_prob(logits, " Austin"), 0.4)
        self.assertEqual(eval_runner.find_target_prob(logits, " Dallas"), 0.0)

    def test_node_to_steer_feature_data_maps_feature_nodes_only(self):
        feature = next(n for n in eval_graph()["nodes"] if n["node_id"] == "0_42")
        logit = next(n for n in eval_graph()["nodes"] if n["node_id"] == "logit_austin")

        self.assertEqual(
            eval_runner.node_to_steer_feature_data(feature),
            {
                "layer": 0,
                "index": 42,
                "token_active_position": 3,
                "steer_position": 3,
                "delta": None,
                "ablate": True,
                "steer_generated_tokens": False,
            },
        )
        self.assertIsNone(eval_runner.node_to_steer_feature_data(logit))
        self.assertIsNone(eval_runner.node_to_steer_feature_data(None))

    def test_choose_keep_ratio_clamps_to_configured_bounds(self):
        graph = eval_graph()

        self.assertEqual(eval_runner.choose_keep_ratio(graph, feature_budget=1, min_keep_ratio=0.2, max_keep_ratio=0.6), 0.5)
        self.assertEqual(eval_runner.choose_keep_ratio(graph, feature_budget=100, min_keep_ratio=0.2, max_keep_ratio=0.6), 0.6)
        self.assertEqual(eval_runner.choose_keep_ratio(graph, feature_budget=0, min_keep_ratio=0.2, max_keep_ratio=0.6), 0.2)
        self.assertEqual(
            eval_runner.choose_keep_ratio({"nodes": [{"feature_type": "embedding"}]}, 10, 0.2, 0.6),
            0.6,
        )

    def test_resolve_scorer_device_prefers_cuda_then_mps_then_cpu(self):
        cuda_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        )
        mps_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        )
        cpu_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )

        self.assertEqual(eval_runner.resolve_scorer_device("auto", cuda_torch), "cuda")
        self.assertEqual(eval_runner.resolve_scorer_device("auto", mps_torch), "mps")
        self.assertEqual(eval_runner.resolve_scorer_device("auto", cpu_torch), "cpu")
        self.assertEqual(eval_runner.resolve_scorer_device("cpu", cuda_torch), "cpu")
        self.assertEqual(eval_runner.resolve_scorer_device("mps", cpu_torch), "mps")

    def test_build_circuit_for_prompt_focuses_prunes_and_configures_scorer(self):
        graph = eval_graph()

        def fake_prune(nodes, links, target_token, keep_ratio):
            self.assertEqual(target_token, " Austin")
            self.assertEqual(keep_ratio, 0.5)
            other_logit = next(n for n in nodes if n["node_id"] == "logit_houston")
            self.assertEqual(other_logit["token_prob"], 0)
            return nodes[:4], links[:1]

        result = eval_runner.build_circuit_for_prompt(
            copy.deepcopy(graph),
            target_token=" Austin",
            method="ia_pc",
            alpha=0.15,
            feature_budget=1,
            min_keep_ratio=0.2,
            max_keep_ratio=0.6,
            device_str="mps",
            scorer_cls=FakeScorer,
            prune_fn=fake_prune,
        )

        self.assertEqual(result["circuit"]["pinnedIds"], ["embed_0", "logit_austin", "0_42"])
        self.assertEqual(result["keep_ratio"], 0.5)
        self.assertEqual(result["pruned_nodes"], 4)
        self.assertEqual(result["pruned_links"], 1)
        self.assertEqual(result["device"], "mps")
        self.assertGreaterEqual(result["build_time"], 0)
        self.assertEqual(FakeScorer.calls[0]["device_str"], "mps")
        self.assertEqual(FakeScorer.calls[0]["endpoints"], ["embed_0", "logit_austin"])
        self.assertEqual(FakeScorer.calls[0]["alpha"], 0.15)
        self.assertEqual(FakeScorer.calls[0]["pc_passes"], 3)

    def test_build_circuit_for_prompt_disables_alpha_and_pc_for_c_only(self):
        eval_runner.build_circuit_for_prompt(
            copy.deepcopy(eval_graph()),
            target_token=" Austin",
            method="c_only",
            alpha=0.15,
            feature_budget=1,
            min_keep_ratio=0.2,
            max_keep_ratio=0.6,
            device_str="cpu",
            scorer_cls=FakeScorer,
            prune_fn=lambda nodes, links, _target, keep_ratio: (nodes, links),
        )

        self.assertEqual(FakeScorer.calls[0]["alpha"], 0.0)
        self.assertEqual(FakeScorer.calls[0]["pc_passes"], 0)


if __name__ == "__main__":
    unittest.main()
