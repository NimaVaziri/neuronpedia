import json
import queue
import unittest
import asyncio
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

from neuronpedia_graph import explorer


def graph_data():
    return {
        "metadata": {"prompt": "Dallas is"},
        "nodes": [
            {
                "node_id": "embedding-a",
                "feature_type": "embedding",
                "is_target_logit": False,
                "token_prob": 0,
                "clerp": "Dallas",
            },
            {
                "node_id": "feature-a",
                "feature_type": "cross layer transcoder",
                "is_target_logit": False,
                "token_prob": 0,
                "clerp": "Texas feature",
            },
            {
                "node_id": "feature-b",
                "feature_type": "cross layer transcoder",
                "is_target_logit": False,
                "token_prob": 0,
                "clerp": "Capital feature",
            },
            {
                "node_id": "logit-target",
                "feature_type": "logit",
                "is_target_logit": True,
                "token_prob": 0.7,
                "logitToken": " Austin",
                "clerp": " Austin",
            },
            {
                "node_id": "logit-other",
                "feature_type": "logit",
                "is_target_logit": False,
                "token_prob": 0.2,
                "logitToken": " Houston",
                "clerp": " Houston",
            },
        ],
        "links": [
            {"source": "feature-a", "target": "logit-target", "weight": -0.5},
            {"source": "feature-b", "target": "logit-target", "weight": 0.9},
            {"source": "embedding-a", "target": "feature-a", "weight": 0.4},
        ],
    }


def parse_sse_event(message):
    event_type = None
    data_lines = []
    for line in message.splitlines():
        if line.startswith("event: "):
            event_type = line[len("event: ") :]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: ") :])
    return event_type, json.loads("\n".join(data_lines))


class FakeBatchGraphScorer:
    calls = []

    def __init__(self, nodes, links, device_str):
        self.nodes = nodes
        self.links = links
        self.device_str = device_str

    def build_circuit_ia_pc(self, endpoint_ids, max_steps, pc_passes, progress_callback):
        self.calls.append(
            {
                "endpoint_ids": endpoint_ids,
                "max_steps": max_steps,
                "pc_passes": pc_passes,
                "device_str": self.device_str,
            }
        )
        progress_callback(1, max_steps, len(endpoint_ids), 0.4)
        if "feature-b" in endpoint_ids:
            return {
                "pinnedIds": ["embedding-a", "logit-target", "feature-b"],
                "replacementScore": 0.2,
                "completenessScore": 0.7,
                "totalFeatures": 1,
            }
        return {
            "pinnedIds": ["embedding-a", "logit-target", "feature-a"],
            "replacementScore": 0.5,
            "completenessScore": 0.3,
            "totalFeatures": 2,
        }


class DuplicateBatchGraphScorer:
    def __init__(self, nodes, links, device_str):
        pass

    def build_circuit_ia_pc(self, endpoint_ids, max_steps, pc_passes, progress_callback):
        return {
            "pinnedIds": ["embedding-a", "logit-target", "feature-a"],
            "replacementScore": 0.1,
            "completenessScore": 0.2,
            "totalFeatures": 1,
        }


class FailingFirstSeedBatchGraphScorer:
    calls = 0

    def __init__(self, nodes, links, device_str):
        pass

    def build_circuit_ia_pc(self, endpoint_ids, max_steps, pc_passes, progress_callback):
        self.__class__.calls += 1
        if self.__class__.calls == 1:
            raise RuntimeError("first seed failed")
        return {
            "pinnedIds": ["embedding-a", "logit-target", "feature-a"],
            "replacementScore": 0.2,
            "completenessScore": 0.3,
            "totalFeatures": 1,
        }


class ExplorerTests(unittest.TestCase):
    def setUp(self):
        FakeBatchGraphScorer.calls = []
        FailingFirstSeedBatchGraphScorer.calls = 0

    def collect_events(self, event_queue, done_sentinel):
        messages = []
        while True:
            item = event_queue.get_nowait()
            if item is done_sentinel:
                break
            messages.append(parse_sse_event(item))
        return messages

    def test_sse_event_encodes_json_and_pads_small_events(self):
        with redirect_stdout(StringIO()):
            message = explorer._sse_event("status", {"phase": "exploring"})

        event_type, data = parse_sse_event(message)

        self.assertEqual(event_type, "status")
        self.assertEqual(data, {"phase": "exploring"})
        self.assertGreaterEqual(len(message), 16384)
        self.assertIn("\n\n:", message)

    def test_sse_event_does_not_pad_large_events(self):
        large_payload = {"message": "x" * 17000}

        with redirect_stdout(StringIO()):
            message = explorer._sse_event("error", large_payload)

        event_type, data = parse_sse_event(message)

        self.assertEqual(event_type, "error")
        self.assertEqual(data, large_payload)
        self.assertNotIn("\n\n:", message)

    def test_prepare_job_inputs_masks_non_target_logits_and_orders_seeds(self):
        job = explorer.ExploreCircuitsJob(
            graph_data=graph_data(),
            target_logit_node_id="logit-target",
            endpoint_node_ids=["embedding-a", "missing-node", "logit-target"],
            num_seeds=2,
        )

        nodes, _links, endpoint_ids, seed_sets = explorer._prepare_job_inputs(job)

        other_logit = next(node for node in nodes if node["node_id"] == "logit-other")
        self.assertEqual(other_logit["token_prob"], 0)
        self.assertEqual(endpoint_ids, ["embedding-a", "logit-target"])
        self.assertEqual(
            seed_sets,
            [
                {"seeds": ["feature-b"], "label": "feature-b"},
                {"seeds": ["feature-a"], "label": "feature-a"},
            ],
        )

    def test_prepare_job_inputs_accepts_object_link_endpoints_and_default_target_logit(self):
        data = graph_data()
        data["links"] = [
            {"source": {"node_id": "feature-a"}, "target": {"node_id": "logit-target"}, "weight": 0.6},
            {"source": {"node_id": "feature-b"}, "target": {"node_id": "logit-target"}, "weight": -0.2},
        ]
        job = explorer.ExploreCircuitsJob(
            graph_data=data,
            endpoint_node_ids=["embedding-a", "logit-target"],
            num_seeds=5,
        )

        nodes, links, endpoint_ids, seed_sets = explorer._prepare_job_inputs(job)

        self.assertEqual(nodes[4]["token_prob"], 0.2)
        self.assertEqual(links, data["links"])
        self.assertEqual(endpoint_ids, ["embedding-a", "logit-target"])
        self.assertEqual(
            seed_sets,
            [
                {"seeds": ["feature-a"], "label": "feature-a"},
                {"seeds": ["feature-b"], "label": "feature-b"},
            ],
        )

    def test_prepare_job_inputs_returns_no_seeds_without_target_logit(self):
        data = graph_data()
        for node in data["nodes"]:
            node["is_target_logit"] = False
        job = explorer.ExploreCircuitsJob(
            graph_data=data,
            endpoint_node_ids=["embedding-a"],
            num_seeds=2,
        )

        _nodes, _links, endpoint_ids, seed_sets = explorer._prepare_job_inputs(job)

        self.assertEqual(endpoint_ids, ["embedding-a"])
        self.assertEqual(seed_sets, [])

    def test_run_exploration_streams_progress_groups_and_ranks_by_combined_score(self):
        job = explorer.ExploreCircuitsJob(
            graph_data=graph_data(),
            target_logit_node_id="logit-target",
            endpoint_node_ids=["embedding-a", "logit-target"],
            num_seeds=2,
            max_iterations=8,
            pc_passes=3,
            keep_ratio=0.5,
            grouping_model="haiku",
        )
        event_queue = queue.Queue()
        done_sentinel = object()

        with (
            patch("neuronpedia_graph.explorer.BatchGraphScorer", FakeBatchGraphScorer),
            patch("neuronpedia_graph.explorer.prune_graph_for_scoring", lambda nodes, links, _target, keep_ratio: (nodes, links)),
            patch(
                "neuronpedia_graph.explorer.auto_group_nodes",
                lambda _nodes, _links, pinned_ids, _prompt, grouping_model: (
                    [["Grouped", *pinned_ids]],
                    ["Grouped explanation"],
                    [{pinned_id: "Grouped because it was pinned." for pinned_id in pinned_ids}],
                    False,
                    None,
                ),
            ),
        ):
            with redirect_stdout(StringIO()):
                explorer._run_exploration(job, "cpu", event_queue, done_sentinel)

        messages = self.collect_events(event_queue, done_sentinel)

        event_types = [event_type for event_type, _data in messages]
        self.assertEqual(event_types[0], "status")
        self.assertIn("progress", event_types)
        self.assertEqual(event_types.count("circuit"), 2)
        self.assertEqual(event_types.count("grouped"), 2)
        self.assertEqual(event_types[-1], "done")

        done_data = messages[-1][1]
        self.assertEqual([circuit["rank"] for circuit in done_data["circuits"]], [1, 2])
        self.assertEqual(done_data["circuits"][0]["seed"], "feature-b")
        self.assertGreater(done_data["circuits"][0]["combined_score"], done_data["circuits"][1]["combined_score"])
        self.assertEqual(
            FakeBatchGraphScorer.calls[0],
            {
                "endpoint_ids": ["embedding-a", "logit-target", "feature-b"],
                "max_steps": 8,
                "pc_passes": 3,
                "device_str": "cpu",
            },
        )

    def test_run_exploration_deduplicates_identical_circuits(self):
        job = explorer.ExploreCircuitsJob(
            graph_data=graph_data(),
            target_logit_node_id="logit-target",
            endpoint_node_ids=["embedding-a", "logit-target"],
            num_seeds=2,
        )
        event_queue = queue.Queue()
        done_sentinel = object()

        with (
            patch("neuronpedia_graph.explorer.BatchGraphScorer", DuplicateBatchGraphScorer),
            patch("neuronpedia_graph.explorer.prune_graph_for_scoring", lambda nodes, links, _target, keep_ratio: (nodes, links)),
            patch(
                "neuronpedia_graph.explorer.auto_group_nodes",
                lambda _nodes, _links, pinned_ids, _prompt, grouping_model: ([], [], [], False, None),
            ),
        ):
            with redirect_stdout(StringIO()):
                explorer._run_exploration(job, "cpu", event_queue, done_sentinel)

        messages = self.collect_events(event_queue, done_sentinel)

        self.assertEqual([event_type for event_type, _data in messages].count("circuit"), 1)
        self.assertEqual(len(messages[-1][1]["circuits"]), 1)

    def test_run_exploration_continues_after_seed_scorer_error(self):
        job = explorer.ExploreCircuitsJob(
            graph_data=graph_data(),
            target_logit_node_id="logit-target",
            endpoint_node_ids=["embedding-a", "logit-target"],
            num_seeds=2,
        )
        event_queue = queue.Queue()
        done_sentinel = object()

        with (
            patch("neuronpedia_graph.explorer.BatchGraphScorer", FailingFirstSeedBatchGraphScorer),
            patch("neuronpedia_graph.explorer.prune_graph_for_scoring", lambda nodes, links, _target, keep_ratio: (nodes, links)),
            patch(
                "neuronpedia_graph.explorer.auto_group_nodes",
                lambda _nodes, _links, pinned_ids, _prompt, grouping_model: ([], [], [], False, None),
            ),
        ):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                explorer._run_exploration(job, "cpu", event_queue, done_sentinel)

        messages = self.collect_events(event_queue, done_sentinel)

        self.assertEqual(FailingFirstSeedBatchGraphScorer.calls, 2)
        self.assertEqual([event_type for event_type, _data in messages].count("circuit"), 1)
        self.assertEqual(messages[-1][0], "done")
        self.assertEqual(len(messages[-1][1]["circuits"]), 1)

    def test_run_exploration_streams_error_when_setup_fails(self):
        job = explorer.ExploreCircuitsJob(
            graph_data=graph_data(),
            target_logit_node_id="logit-target",
            endpoint_node_ids=["embedding-a", "logit-target"],
            num_seeds=1,
        )
        event_queue = queue.Queue()
        done_sentinel = object()

        with patch("neuronpedia_graph.explorer.prune_graph_for_scoring", side_effect=RuntimeError("prune failed")):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                explorer._run_exploration(job, "cpu", event_queue, done_sentinel)

        messages = self.collect_events(event_queue, done_sentinel)

        self.assertEqual(messages[-1][0], "error")
        self.assertEqual(messages[-1][1], {"message": "prune failed"})

    def test_run_exploration_with_no_seeds_returns_empty_done_event(self):
        data = graph_data()
        for node in data["nodes"]:
            node["is_target_logit"] = False
        job = explorer.ExploreCircuitsJob(
            graph_data=data,
            endpoint_node_ids=["embedding-a"],
            num_seeds=2,
        )
        event_queue = queue.Queue()
        done_sentinel = object()

        with patch("neuronpedia_graph.explorer.BatchGraphScorer") as scorer_cls:
            with redirect_stdout(StringIO()):
                explorer._run_exploration(job, "cpu", event_queue, done_sentinel)

        messages = self.collect_events(event_queue, done_sentinel)

        scorer_cls.assert_not_called()
        self.assertEqual(
            messages[0],
            ("status", {"phase": "exploring", "total_seeds": 0, "completed_seeds": 0, "circuits_found": 0}),
        )
        self.assertEqual(messages[-1], ("done", {"circuits": []}))

    def test_grouping_worker_emits_failure_payload_when_grouping_raises(self):
        grouping_queue = queue.Queue()
        event_queue = queue.Queue()
        grouping_stop = object()
        queued_result = {
            "circuit_id": "feature-a",
            "pinned_ids": ["feature-a"],
            "supernodes": [],
            "supernode_explanations": [],
            "supernode_member_reasons": [],
        }
        grouping_queue.put((queued_result, 1))
        grouping_queue.put(grouping_stop)

        with patch("neuronpedia_graph.explorer.auto_group_nodes", side_effect=RuntimeError("grouping failed")):
            with redirect_stdout(StringIO()):
                explorer._run_grouping_worker(
                    grouping_queue,
                    grouping_stop,
                    event_queue,
                    2,
                    graph_data()["nodes"],
                    graph_data()["links"],
                    "Dallas is",
                    "sonnet",
                )

        messages = [parse_sse_event(event_queue.get_nowait()), parse_sse_event(event_queue.get_nowait())]

        self.assertEqual(messages[0][0], "status")
        self.assertEqual(messages[0][1]["phase"], "grouping")
        self.assertEqual(messages[1][0], "grouped")
        self.assertEqual(messages[1][1]["circuit"]["grouping_failed"], True)
        self.assertEqual(messages[1][1]["circuit"]["grouping_error"], "grouping failed")

    def test_build_explore_circuits_response_streams_until_done_sentinel(self):
        async def collect_stream():
            response = explorer.build_explore_circuits_response(
                explorer.ExploreCircuitsJob(graph_data=graph_data()),
                "cpu",
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            return chunks

        def fake_run_exploration(_job, _device_str, event_queue, done_sentinel):
            event_queue.put(explorer._sse_event("done", {"circuits": []}))
            event_queue.put(done_sentinel)

        with patch("neuronpedia_graph.explorer._run_exploration", fake_run_exploration):
            with redirect_stdout(StringIO()):
                chunks = asyncio.run(collect_stream())

        self.assertEqual(len(chunks), 1)
        self.assertEqual(parse_sse_event(chunks[0]), ("done", {"circuits": []}))


if __name__ == "__main__":
    unittest.main()
