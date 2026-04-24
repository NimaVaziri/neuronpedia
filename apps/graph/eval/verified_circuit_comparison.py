"""Compare verified circuits with CircuitExplorer circuits on cached verified graphs."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
GRAPH_APP_DIR = EVAL_DIR.parent
if str(GRAPH_APP_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_APP_DIR))

import eval_runner as eval_base
from neuronpedia_graph.scorer import BatchGraphScorer, prune_graph_for_scoring


VERIFIED_DIR = EVAL_DIR / "verified-circuits"
DEFAULT_FIXTURE = VERIFIED_DIR / "verified-circuits.json"
DEFAULT_GRAPH_DIR = VERIFIED_DIR / "graphs"
DEFAULT_RESULTS_DIR = VERIFIED_DIR / "results"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def pp(value: float) -> str:
    return f"{value * 100:.1f}pp"


def load_fixture(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return sorted(payload["circuits"], key=lambda circuit: circuit["order"])


def graph_path_for(graph_dir: Path, circuit: dict[str, Any]) -> Path:
    return graph_dir / f"{prefixless_id(circuit['graph']['slug'])}.json"


def prefixless_id(value: str) -> str:
    return value.removeprefix("gemma-")


def load_graph(graph_dir: Path, circuit: dict[str, Any]) -> dict[str, Any]:
    with graph_path_for(graph_dir, circuit).open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_model_id(model_id: str) -> str:
    if "/" in model_id:
        return model_id
    if model_id == "gemma-2-2b":
        return "google/gemma-2-2b"
    return model_id


def generate_graph(graph_dir: Path, circuit: dict[str, Any]) -> dict[str, Any]:
    graph_dir.mkdir(parents=True, exist_ok=True)
    view = circuit.get("view", {})
    body = json.dumps({
        "model_id": normalize_model_id(circuit["model_id"]),
        "prompt": circuit["prompt"],
        "batch_size": 1,
        "max_n_logits": 10,
        "desired_logit_prob": 0.95,
        "node_threshold": view.get("pruning_threshold") if view.get("pruning_threshold") is not None else 0.7,
        "edge_threshold": view.get("density_threshold") if view.get("density_threshold") is not None else 0.99,
        "slug_identifier": prefixless_id(circuit["graph"]["slug"]),
        "max_feature_nodes": 10000,
        "signed_url": None,
    }).encode()
    req = urllib.request.Request(
        f"{eval_base.STEER_URL}/generate-graph",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "x-secret-key": eval_base.STEER_SECRET},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        graph = json.loads(resp.read().decode("utf-8"))
    graph_path_for(graph_dir, circuit).write_text(json.dumps(graph), encoding="utf-8")
    return graph


def target_candidates(target_token: str) -> list[str]:
    stripped = target_token.strip()
    candidates = [target_token]
    if stripped:
        candidates.append(f" {stripped}" if target_token == stripped else stripped)
    return list(dict.fromkeys(candidates))


def find_best_target_prob(logits_by_token: list[dict[str, Any]], candidates: list[str]) -> float:
    return max((eval_base.find_target_prob(logits_by_token, target) for target in candidates), default=0.0)


def graph_stats(graph: dict[str, Any]) -> dict[str, int]:
    return {
        "nodes": len(graph["nodes"]),
        "links": len(graph["links"]),
        "feature_nodes": sum(1 for node in graph["nodes"] if node["feature_type"] == "cross layer transcoder"),
    }


def graph_ablation_limits(graph: dict[str, Any], feature_width: int) -> tuple[set[int], int, int]:
    clt_nodes = [node for node in graph["nodes"] if node["feature_type"] == "cross layer transcoder"]
    valid_layers = {int(node["layer"]) for node in clt_nodes}
    max_ctx_idx = max((int(node["ctx_idx"]) for node in graph["nodes"]), default=-1)
    return valid_layers, max_ctx_idx, feature_width - 1


def steer_feature_from_id(
    node_id: str,
    valid_layers: set[int],
    max_ctx_idx: int,
    max_feature_index: int,
) -> dict[str, Any] | None:
    parts = node_id.split("_")
    if len(parts) != 3 or parts[0] == "E":
        return None
    try:
        layer = int(parts[0])
        index = int(parts[1])
        ctx_idx = int(parts[2])
    except ValueError:
        return None
    if layer not in valid_layers:
        return None
    if ctx_idx < 0 or ctx_idx > max_ctx_idx:
        return None
    if index < 0 or index > max_feature_index:
        return None
    return {
        "layer": layer,
        "index": index,
        "token_active_position": ctx_idx,
        "steer_position": ctx_idx,
        "delta": None,
        "ablate": True,
        "steer_generated_tokens": False,
    }


def circuit_features(
    graph: dict[str, Any],
    pinned_ids: list[str],
    feature_width: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    node_by_id = {node["node_id"]: node for node in graph["nodes"]}
    graph_node_ids = set(node_by_id)
    valid_layers, max_ctx_idx, max_feature_index = graph_ablation_limits(graph, feature_width)
    features = []
    unsupported_ids = []

    for node_id in pinned_ids:
        node = node_by_id.get(node_id)
        feature = eval_base.node_to_steer_feature_data(node, node_id)
        if feature is None:
            feature = steer_feature_from_id(node_id, valid_layers, max_ctx_idx, max_feature_index)
        if feature is None:
            unsupported_ids.append(node_id)
        else:
            features.append(feature)

    return features, {
        "raw_feature_count": len(pinned_ids),
        "evaluated_feature_count": len(features),
        "unsupported_feature_count": len(unsupported_ids),
        "graph_matched_feature_count": sum(1 for node_id in pinned_ids if node_id in graph_node_ids),
        "unsupported_feature_ids": unsupported_ids,
    }


def complement_features(graph: dict[str, Any], pinned_ids: list[str]) -> list[dict[str, Any]]:
    pinned_id_set = set(pinned_ids)
    return [
        feature
        for feature in (
            eval_base.node_to_steer_feature_data(node)
            for node in graph["nodes"]
            if node["feature_type"] == "cross layer transcoder" and node["node_id"] not in pinned_id_set
        )
        if feature
    ]


def validate_circuit(
    graph: dict[str, Any],
    target_token: str,
    pinned_ids: list[str],
    feature_width: int,
) -> dict[str, Any]:
    prompt = graph["metadata"]["prompt"].replace("<bos>", "")
    pinned_features, coverage = circuit_features(graph, pinned_ids, feature_width)
    complement = complement_features(graph, pinned_ids)

    feature_sets = []
    indices = {}
    if pinned_features:
        indices["necessity"] = len(feature_sets)
        feature_sets.append(pinned_features)
    if complement:
        indices["sufficiency"] = len(feature_sets)
        feature_sets.append(complement)

    batched = eval_base.call_steer_batch(prompt, feature_sets)
    candidates = target_candidates(target_token)
    baseline_prob = find_best_target_prob(batched["DEFAULT_LOGITS_BY_TOKEN"], candidates)
    batched_results = batched.get("STEERED_RESULTS", [])

    nec_prob = baseline_prob
    if "necessity" in indices:
        nec_prob = find_best_target_prob(
            batched_results[indices["necessity"]]["STEERED_LOGITS_BY_TOKEN"],
            candidates,
        )

    suf_prob = 0.0
    if "sufficiency" in indices:
        suf_prob = find_best_target_prob(
            batched_results[indices["sufficiency"]]["STEERED_LOGITS_BY_TOKEN"],
            candidates,
        )

    return {
        **coverage,
        "target_candidates": candidates,
        "baseline": baseline_prob,
        "necessity": baseline_prob - nec_prob,
        "sufficiency": suf_prob / baseline_prob if baseline_prob > 0 else 0.0,
        "nec_prob": nec_prob,
        "suf_prob": suf_prob,
    }


def build_circuit_explorer_circuit(
    graph: dict[str, Any],
    target_token: str,
    scorer_device: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    focused_graph = eval_base.focus_on_target(graph, target_token)
    endpoints = eval_base.get_endpoints(focused_graph, target_token)
    keep_ratio = eval_base.choose_keep_ratio(
        focused_graph,
        args.feature_budget,
        args.min_keep_ratio,
        args.max_keep_ratio,
    )

    t0 = time.time()
    pruned_nodes, pruned_links = prune_graph_for_scoring(
        focused_graph["nodes"],
        focused_graph["links"],
        target_token,
        keep_ratio=keep_ratio,
    )
    scorer = BatchGraphScorer(pruned_nodes, pruned_links, device_str=scorer_device)
    circuit = scorer.build_circuit_ia_pc(
        endpoints,
        alpha=args.alpha,
        max_steps=args.ia_steps,
        pc_passes=args.pc_passes,
    )

    return {
        "pinned_ids": circuit["pinnedIds"],
        "feature_count": circuit["totalFeatures"],
        "ia_features": circuit["iaFeatures"],
        "pc_features": circuit["pcFeatures"],
        "replacement_score": circuit["replacementScore"],
        "completeness_score": circuit["completenessScore"],
        "build_time": time.time() - t0,
        "keep_ratio": keep_ratio,
        "pruned_nodes": len(pruned_nodes),
        "pruned_links": len(pruned_links),
    }


def select_circuits(circuits: list[dict[str, Any]], selection: str) -> list[dict[str, Any]]:
    if selection == "all":
        return circuits
    wanted = {circuit_id.strip() for circuit_id in selection.split(",") if circuit_id.strip()}
    selected = [circuit for circuit in circuits if circuit["id"] in wanted]
    missing = wanted - {circuit["id"] for circuit in selected}
    if missing:
        raise SystemExit(f"Unknown circuit IDs: {', '.join(sorted(missing))}")
    return selected


def median_metric(results: list[dict[str, Any]], family: str, metric: str) -> float:
    values = [
        result[family][metric]
        for result in results
        if result.get("status") == "ok" and metric in result.get(family, {})
    ]
    return median(values) if values else 0.0


def write_report(payload: dict[str, Any], report_path: Path) -> None:
    results = [result for result in payload["results"] if result.get("status") == "ok"]
    errors = [result for result in payload["results"] if result.get("status") != "ok"]

    lines = [
        "# Verified Circuits vs CircuitExplorer",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Model: `{payload['model_id']}`",
        f"- Steer URL: `{payload['steer_url']}`",
        f"- Scorer device: `{payload['scorer_device']}`",
        f"- CircuitExplorer IA steps: `{payload['circuit_explorer']['ia_steps']}`",
        f"- CircuitExplorer PC passes: `{payload['circuit_explorer']['pc_passes']}`",
        f"- Circuits evaluated: `{len(results)}`",
        f"- Errors: `{len(errors)}`",
        "",
        "## Median Scores",
        "",
        "| Circuit family | Median necessity | Median sufficiency |",
        "| --- | ---: | ---: |",
        f"| Verified circuits | {pp(payload['summary']['verified_median_necessity'])} | {pct(payload['summary']['verified_median_sufficiency'])} |",
        f"| CircuitExplorer circuits | {pp(payload['summary']['circuit_explorer_median_necessity'])} | {pct(payload['summary']['circuit_explorer_median_sufficiency'])} |",
        "",
        "## Per-Circuit Results",
        "",
        "| ID | Target | Verified Necessity | CircuitExplorer Necessity | Verified Sufficiency | CircuitExplorer Sufficiency | Verified Features | CircuitExplorer Features |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        verified = result["verified"]
        explorer = result["circuit_explorer"]
        graph = result["graph"]
        lines.append(
            "| "
            f"{prefixless_id(result['id'])} | "
            f"`{result['target_token']}` | "
            f"{pp(verified['necessity'])} | "
            f"{pp(explorer['necessity'])} | "
            f"{pct(verified['sufficiency'])} | "
            f"{pct(explorer['sufficiency'])} | "
            f"{verified['raw_feature_count']} | "
            f"{explorer['feature_count']} |"
        )

    if errors:
        lines.extend(["", "## Errors", ""])
        for error in errors:
            lines.append(f"- `{error['id']}`: {error['error']}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    circuits = select_circuits(load_fixture(args.fixture), args.circuits)
    scorer_device = eval_base.get_scorer_device()
    results = []

    print(f"Comparing {len(circuits)} verified circuits")
    print(f"Steer URL: {eval_base.STEER_URL} | Model: {eval_base.MODEL_ID} | Scorer device: {scorer_device}")
    print(f"CircuitExplorer: IA steps={args.ia_steps}, PC passes={args.pc_passes}")

    suite_start = time.time()
    for index, circuit in enumerate(circuits, start=1):
        prompt_start = time.time()
        circuit_id = circuit["id"]
        target_token = circuit["target_token"]
        graph_path = graph_path_for(args.graph_dir, circuit)
        print(f"[{index}/{len(circuits)}] {circuit_id}: {circuit['prompt'][:60]!r} -> {target_token!r}")

        try:
            if graph_path.exists():
                graph = load_graph(args.graph_dir, circuit)
            elif args.generate_missing_graphs:
                graph = generate_graph(args.graph_dir, circuit)
            else:
                raise FileNotFoundError(f"Missing graph: {graph_path}")

            stats = graph_stats(graph)

            verified_start = time.time()
            verified_eval = validate_circuit(
                graph,
                target_token,
                circuit["circuit"]["pinned_ids"],
                args.feature_width,
            )
            verified_eval["eval_time"] = time.time() - verified_start

            build = build_circuit_explorer_circuit(graph, target_token, scorer_device, args)
            explorer_start = time.time()
            explorer_eval = validate_circuit(
                graph,
                target_token,
                build["pinned_ids"],
                args.feature_width,
            )
            explorer_eval["eval_time"] = time.time() - explorer_start

            result = {
                "status": "ok",
                "id": circuit_id,
                "label": circuit["label"],
                "prompt": circuit["prompt"],
                "target_token": target_token,
                "graph_path": str(graph_path),
                "graph": stats,
                "verified": verified_eval,
                "circuit_explorer": {
                    **build,
                    "target_candidates": explorer_eval["target_candidates"],
                    "baseline": explorer_eval["baseline"],
                    "necessity": explorer_eval["necessity"],
                    "sufficiency": explorer_eval["sufficiency"],
                    "nec_prob": explorer_eval["nec_prob"],
                    "suf_prob": explorer_eval["suf_prob"],
                    "eval_time": explorer_eval["eval_time"],
                    "evaluated_feature_count": explorer_eval["evaluated_feature_count"],
                    "unsupported_feature_count": explorer_eval["unsupported_feature_count"],
                },
                "elapsed_time": time.time() - prompt_start,
            }
            print(
                "  "
                f"Verified: {verified_eval['evaluated_feature_count']}/{verified_eval['raw_feature_count']} eval features, "
                f"nec {pp(verified_eval['necessity'])}, suf {pct(verified_eval['sufficiency'])} | "
                f"CE: {build['feature_count']} features, nec {pp(explorer_eval['necessity'])}, "
                f"suf {pct(explorer_eval['sufficiency'])}"
            )
        except Exception as exc:
            result = {
                "status": "error",
                "id": circuit_id,
                "prompt": circuit.get("prompt"),
                "target_token": target_token,
                "graph_path": str(graph_path),
                "error": str(exc),
                "elapsed_time": time.time() - prompt_start,
            }
            print(f"  FAILED: {exc}")

        results.append(result)

    ok_results = [result for result in results if result.get("status") == "ok"]
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "fixture": str(args.fixture),
        "graph_dir": str(args.graph_dir),
        "model_id": eval_base.MODEL_ID,
        "steer_url": eval_base.STEER_URL,
        "scorer_device": scorer_device,
        "circuit_explorer": {
            "alpha": args.alpha,
            "ia_steps": args.ia_steps,
            "pc_passes": args.pc_passes,
            "feature_budget": args.feature_budget,
            "min_keep_ratio": args.min_keep_ratio,
            "max_keep_ratio": args.max_keep_ratio,
        },
        "summary": {
            "verified_median_necessity": median_metric(ok_results, "verified", "necessity"),
            "verified_median_sufficiency": median_metric(ok_results, "verified", "sufficiency"),
            "circuit_explorer_median_necessity": median_metric(ok_results, "circuit_explorer", "necessity"),
            "circuit_explorer_median_sufficiency": median_metric(ok_results, "circuit_explorer", "sufficiency"),
        },
        "wall_time": time.time() - suite_start,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--circuits", default="all", help="all or comma-separated circuit IDs")
    parser.add_argument("--generate-missing-graphs", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--ia-steps", type=int, default=40)
    parser.add_argument("--pc-passes", type=int, default=1)
    parser.add_argument("--feature-budget", type=int, default=450)
    parser.add_argument("--min-keep-ratio", type=float, default=0.2)
    parser.add_argument("--max-keep-ratio", type=float, default=0.4)
    parser.add_argument("--feature-width", type=int, default=16384)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output
    report_path = args.report or args.results_dir / f"verified-circuit-comparison-{timestamp}.md"

    payload = run(args)
    if output_path:
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload, report_path)

    ok_count = sum(1 for result in payload["results"] if result.get("status") == "ok")
    if output_path:
        print(f"\nWrote JSON: {output_path}")
    else:
        print("\nSkipped JSON output")
    print(f"Wrote report: {report_path}")
    print(f"Completed {ok_count}/{len(payload['results'])} circuits in {payload['wall_time']:.1f}s")


if __name__ == "__main__":
    main()
