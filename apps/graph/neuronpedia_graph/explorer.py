import asyncio
import json
import queue
import threading
from dataclasses import dataclass, field
from typing import Any

from fastapi.responses import StreamingResponse

from neuronpedia_graph.grouping import auto_group_nodes, format_grouping_response
from neuronpedia_graph.scorer import BatchGraphScorer, prune_graph_for_scoring


@dataclass
class ExploreCircuitsJob:
    graph_data: dict[str, Any]
    target_logit_node_id: str | None = None
    endpoint_node_ids: list[str] = field(default_factory=list)
    num_seeds: int = 5
    max_iterations: int = 80
    pc_passes: int = 1
    keep_ratio: float = 0.4
    grouping_model: str = "sonnet"


def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    message = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    if len(message) < 16384:
        message += ":" + " " * (16384 - len(message) - 2) + "\n\n"
    print(f"[SSE] queuing {event_type} event ({len(message)} bytes)", flush=True)
    return message


def _prepare_job_inputs(job: ExploreCircuitsJob):
    nodes = [dict(node) for node in job.graph_data.get("nodes", [])]
    links = job.graph_data.get("links", [])

    if job.target_logit_node_id:
        for node in nodes:
            if node.get("feature_type") == "logit" and node["node_id"] != job.target_logit_node_id:
                node["token_prob"] = 0

    all_node_ids = {node["node_id"] for node in nodes}
    all_feature_ids = {
        node["node_id"] for node in nodes if node["feature_type"] == "cross layer transcoder"
    }
    endpoint_ids = [endpoint_id for endpoint_id in job.endpoint_node_ids if endpoint_id in all_node_ids]

    target_logit = next(
        (node for node in nodes if node["node_id"] == job.target_logit_node_id),
        next((node for node in nodes if node.get("is_target_logit")), None),
    )

    seed_sets = []
    if target_logit:
        incoming = []
        for link in links:
            target = link.get("target") if isinstance(link.get("target"), str) else link["target"]["node_id"]
            if target == target_logit["node_id"]:
                source = link.get("source") if isinstance(link.get("source"), str) else link["source"]["node_id"]
                if source in all_feature_ids:
                    incoming.append({"source": source, "weight": abs(link["weight"])})
        incoming.sort(key=lambda edge: edge["weight"], reverse=True)
        for edge in incoming[: job.num_seeds]:
            seed_sets.append({"seeds": [edge["source"]], "label": edge["source"]})

    return nodes, links, endpoint_ids, seed_sets


def _run_grouping_worker(
    grouping_queue: "queue.Queue[tuple[dict[str, Any], int] | object]",
    grouping_stop: object,
    event_queue: "queue.Queue[str | object]",
    total_seeds: int,
    nodes: list[dict[str, Any]],
    links: list[dict[str, Any]],
    prompt_text: str,
    grouping_model: str,
):
    while True:
        item = grouping_queue.get()
        try:
            if item is grouping_stop:
                return

            queued_result, rank = item
            event_queue.put(
                _sse_event(
                    "status",
                    {
                        "phase": "grouping",
                        "grouping_circuit": rank,
                        "circuit_id": queued_result["circuit_id"],
                        "total_to_group": total_seeds,
                    },
                )
            )
            (
                supernodes,
                explanations,
                member_reasons,
                grouping_failed,
                grouping_error,
            ) = auto_group_nodes(
                nodes,
                links,
                queued_result["pinned_ids"],
                prompt_text,
                grouping_model,
            )
            queued_result.update(
                format_grouping_response(
                    supernodes,
                    explanations,
                    member_reasons,
                    grouping_model,
                    grouping_failed,
                    grouping_error,
                )
            )
            event_queue.put(
                _sse_event(
                    "grouped",
                    {
                        "rank": rank,
                        "circuit_id": queued_result["circuit_id"],
                        "circuit": queued_result,
                    },
                )
            )
        except Exception as error:
            print(f"[explore-circuits] Grouping error for circuit {rank}: {error}", flush=True)
            queued_result.update(
                format_grouping_response(
                    queued_result.get("supernodes", []),
                    queued_result.get("supernode_explanations", []),
                    queued_result.get("supernode_member_reasons", []),
                    grouping_model,
                    True,
                    str(error),
                )
            )
            event_queue.put(
                _sse_event(
                    "grouped",
                    {
                        "rank": rank,
                        "circuit_id": queued_result["circuit_id"],
                        "circuit": queued_result,
                    },
                )
            )
        finally:
            grouping_queue.task_done()


def _run_exploration(
    job: ExploreCircuitsJob,
    device_str: str,
    event_queue: "queue.Queue[str | object]",
    done_sentinel: object,
):
    try:
        nodes, links, endpoint_ids, seed_sets = _prepare_job_inputs(job)
        total_seeds = len(seed_sets)
        prompt_text = job.graph_data.get("metadata", {}).get("prompt", "")

        grouping_queue: "queue.Queue[tuple[dict[str, Any], int] | object]" = queue.Queue()
        grouping_stop = object()
        grouping_thread = threading.Thread(
            target=_run_grouping_worker,
            args=(
                grouping_queue,
                grouping_stop,
                event_queue,
                total_seeds,
                nodes,
                links,
                prompt_text,
                job.grouping_model,
            ),
            daemon=True,
        )
        grouping_thread.start()

        event_queue.put(
            _sse_event(
                "status",
                {
                    "phase": "exploring",
                    "total_seeds": total_seeds,
                    "completed_seeds": 0,
                    "circuits_found": 0,
                },
            )
        )

        if total_seeds == 0:
            grouping_queue.put(grouping_stop)
            grouping_thread.join()
            event_queue.put(_sse_event("done", {"circuits": []}))
            print("[explore-circuits] Done: 0 circuits found", flush=True)
            return

        target_logit_node = next((node for node in nodes if node.get("is_target_logit")), None)
        target_token = target_logit_node.get("logitToken", "") if target_logit_node else ""
        pruned_nodes, pruned_links = prune_graph_for_scoring(
            nodes,
            links,
            target_token,
            keep_ratio=job.keep_ratio,
        )
        print(
            f"[explore-circuits] Pruned: {len(nodes)} -> {len(pruned_nodes)} nodes, "
            f"{len(links)} -> {len(pruned_links)} links",
            flush=True,
        )

        scorer = BatchGraphScorer(pruned_nodes, pruned_links, device_str=device_str)
        print(
            f"[explore-circuits] BatchGraphScorer built: {len(pruned_nodes)} nodes, {len(pruned_links)} links",
            flush=True,
        )

        circuits = []
        seen = set()

        print(
            f"[explore-circuits] Starting loop over {total_seeds} seeds, endpoints={len(endpoint_ids)}",
            flush=True,
        )
        for index, seed_set in enumerate(seed_sets):
            seed_endpoint_ids = list(endpoint_ids) + [
                seed for seed in seed_set["seeds"] if seed not in endpoint_ids
            ]

            event_queue.put(
                _sse_event(
                    "progress",
                    {
                        "current_seed": index + 1,
                        "total_seeds": total_seeds,
                        "step": 0,
                        "max_steps": job.max_iterations,
                        "features": 0,
                        "c_score": 0.0,
                    },
                )
            )

            def _progress(step, max_steps, features, c_score):
                event_queue.put(
                    _sse_event(
                        "progress",
                        {
                            "current_seed": index + 1,
                            "total_seeds": total_seeds,
                            "step": step,
                            "max_steps": max_steps,
                            "features": features,
                            "c_score": round(c_score, 3),
                        },
                    )
                )

            try:
                ia_pc_result = scorer.build_circuit_ia_pc(
                    seed_endpoint_ids,
                    max_steps=job.max_iterations,
                    pc_passes=job.pc_passes,
                    progress_callback=_progress,
                )
                print(
                    f"[explore-circuits] Seed {index + 1}/{total_seeds}: {ia_pc_result['totalFeatures']} features",
                    flush=True,
                )
            except Exception as error:
                print(f"[explore-circuits] ERROR on seed {index + 1}: {error}", flush=True)
                import traceback

                traceback.print_exc()
                continue

            result = {
                "pinned_ids": ia_pc_result["pinnedIds"],
                "replacement_score": ia_pc_result["replacementScore"],
                "completeness_score": ia_pc_result["completenessScore"],
                "combined_score": ia_pc_result["replacementScore"] + ia_pc_result["completenessScore"],
            }

            circuit_id = ",".join(sorted(result["pinned_ids"]))
            if circuit_id in seen:
                event_queue.put(
                    _sse_event(
                        "status",
                        {
                            "phase": "exploring",
                            "total_seeds": total_seeds,
                            "completed_seeds": index + 1,
                            "circuits_found": len(circuits),
                        },
                    )
                )
                continue

            seen.add(circuit_id)
            feature_count = ia_pc_result["totalFeatures"]
            result.update(
                {
                    "circuit_id": circuit_id,
                    "seed": seed_set["label"],
                    "node_count": feature_count,
                }
            )
            result.update(
                format_grouping_response([], [], [], job.grouping_model, False, None)
            )

            circuits.append(result)
            event_queue.put(
                _sse_event(
                    "circuit",
                    {
                        **result,
                        "completed_seeds": index + 1,
                        "total_seeds": total_seeds,
                        "circuits_found": len(circuits),
                    },
                )
            )
            grouping_queue.put((result, len(circuits)))

        grouping_queue.join()
        grouping_queue.put(grouping_stop)
        grouping_thread.join()

        circuits.sort(key=lambda circuit: circuit["combined_score"], reverse=True)

        for rank, circuit in enumerate(circuits, start=1):
            circuit["rank"] = rank

        event_queue.put(_sse_event("done", {"circuits": circuits[:100]}))
        print(f"[explore-circuits] Done: {len(circuits)} circuits found", flush=True)
    except Exception as error:
        print(f"[explore-circuits] STREAM ERROR: {error}", flush=True)
        import traceback

        traceback.print_exc()
        event_queue.put(_sse_event("error", {"message": str(error)}))
    finally:
        event_queue.put(done_sentinel)


def build_explore_circuits_response(job: ExploreCircuitsJob, device_str: str) -> StreamingResponse:
    event_queue: "queue.Queue[str | object]" = queue.Queue()
    done_sentinel = object()

    async def async_event_stream():
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_exploration, job, device_str, event_queue, done_sentinel)
        while True:
            while event_queue.empty():
                await asyncio.sleep(0.05)
            item = event_queue.get()
            if item is done_sentinel:
                break
            yield item

    return StreamingResponse(async_event_stream(), media_type="text/event-stream")
