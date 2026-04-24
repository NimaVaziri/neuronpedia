"""CircuitExplorer graph scoring and search.

This module intentionally has no model dependencies. It receives an attribution
graph, turns it into tensors, and scores candidate pin sets with Replacement (R)
and Completeness (C). CircuitExplorer also uses the scorer to run IA+PC search:
influence-aware greedy feature selection followed by pathway completion.
"""
import torch


FEATURE_TYPE_TRANSCODER = "cross layer transcoder"
FEATURE_TYPE_ERROR = "mlp reconstruction error"
FEATURE_TYPE_EMBEDDING = "embedding"
FEATURE_TYPE_LOGIT = "logit"

FEATURE_TYPE_SORT_ORDER = [
    FEATURE_TYPE_TRANSCODER,
    FEATURE_TYPE_ERROR,
    FEATURE_TYPE_EMBEDDING,
    FEATURE_TYPE_LOGIT,
]

MIN_DENOMINATOR = 1e-10
MAX_INFLUENCE_HOPS = 200
MAX_SUGGESTION_INFLUENCE_HOPS = 1000
INFLUENCE_STOP_THRESHOLD = 1e-15
MIN_FEATURES_AFTER_PRUNING = 50

DIRECT_TARGET_EDGE_WEIGHT = 2.0
INDIRECT_TARGET_EDGE_WEIGHT = 0.3
NODE_INFLUENCE_TIEBREAKER_WEIGHT = 0.1
EDGE_CONNECTIVITY_TIEBREAKER_WEIGHT = 0.05
SEARCH_NODE_INFLUENCE_WEIGHT = 0.01
SUGGESTION_OUTGOING_EDGE_WEIGHT = 2.0
PATHWAY_COMPLETION_INCOMING_EDGE_WEIGHT = 0.5


def _link_endpoint_id(link, endpoint_name):
    endpoint = link[endpoint_name]
    return endpoint if isinstance(endpoint, str) else endpoint["node_id"]


def _is_feature_node(node):
    return node.get("feature_type") == FEATURE_TYPE_TRANSCODER


def _sort_graph_node(node):
    feature_type = node["feature_type"]
    type_rank = (
        FEATURE_TYPE_SORT_ORDER.index(feature_type)
        if feature_type in FEATURE_TYPE_SORT_ORDER
        else len(FEATURE_TYPE_SORT_ORDER)
    )
    layer = node.get("layer", "0")
    layer_rank = 0 if layer == "E" else (int(layer) if str(layer).isdigit() else 999)
    return (type_rank, layer_rank, node.get("ctx_idx", 0), node.get("feature", 0) or 0)


@torch.jit.script
def _jit_score_single(
    base_adjacency: torch.Tensor,
    feature_node_indices: torch.Tensor,
    feature_error_node_indices: torch.Tensor,
    pinned_feature_mask: torch.Tensor,
    logit_weights: torch.Tensor,
    error_start: int,
    error_end: int,
    token_end: int,
) -> tuple[float, float]:
    """Score one feature pin mask. Kept TorchScript-friendly for the hot path."""
    min_denominator = 1e-10
    max_influence_hops = 200
    influence_stop_threshold = 1e-15
    node_count = base_adjacency.shape[0]
    adjacency = base_adjacency.clone()

    # Unpinned features are replaced by their matching reconstruction error node.
    for i in range(feature_node_indices.shape[0]):
        if pinned_feature_mask[i]:
            continue
        feature_node_index = feature_node_indices[i]
        error_node_index = feature_error_node_indices[i]
        if error_node_index >= 0:
            adjacency[:, error_node_index] = (
                adjacency[:, error_node_index] + adjacency[:, feature_node_index]
            )
        adjacency[feature_node_index, :] = torch.zeros(
            node_count, dtype=adjacency.dtype, device=adjacency.device
        )
        adjacency[:, feature_node_index] = torch.zeros(
            node_count, dtype=adjacency.dtype, device=adjacency.device
        )

    abs_adjacency = adjacency.abs()
    normalized_adjacency = abs_adjacency / abs_adjacency.sum(
        dim=1, keepdim=True
    ).clamp(min=min_denominator)

    current_walk = logit_weights @ normalized_adjacency
    influence = current_walk.clone()
    for _ in range(max_influence_hops):
        current_walk = current_walk @ normalized_adjacency
        if current_walk.abs().max() < influence_stop_threshold:
            break
        influence = influence + current_walk

    token_influence = influence[error_end:token_end].sum()
    error_influence = influence[error_start:error_end].sum()
    replacement_score = token_influence / (token_influence + error_influence).clamp(
        min=min_denominator
    )

    non_error_outflow = 1.0 - normalized_adjacency[:, error_start:error_end].sum(dim=1)
    outgoing_influence = influence + logit_weights
    completeness_score = (
        (non_error_outflow * outgoing_influence).sum()
        / outgoing_influence.sum().clamp(min=min_denominator)
    )
    return replacement_score.item(), completeness_score.item()


@torch.jit.script
def _jit_score_batch(
    base_adjacency: torch.Tensor,
    feature_node_indices: torch.Tensor,
    feature_error_node_indices: torch.Tensor,
    pinned_feature_masks: torch.Tensor,
    logit_weights: torch.Tensor,
    error_start: int,
    error_end: int,
    token_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score candidate masks together so greedy search pays setup cost once."""
    min_denominator = 1e-10
    max_influence_hops = 200
    influence_stop_threshold = 1e-15
    batch_size = pinned_feature_masks.shape[0]
    node_count = base_adjacency.shape[0]
    batch_adjacency = base_adjacency.unsqueeze(0).expand(batch_size, -1, -1).clone()

    for batch_index in range(batch_size):
        for i in range(feature_node_indices.shape[0]):
            if pinned_feature_masks[batch_index, i]:
                continue
            feature_node_index = feature_node_indices[i]
            error_node_index = feature_error_node_indices[i]
            if error_node_index >= 0:
                batch_adjacency[batch_index, :, error_node_index] = (
                    batch_adjacency[batch_index, :, error_node_index]
                    + batch_adjacency[batch_index, :, feature_node_index]
                )
            batch_adjacency[batch_index, feature_node_index, :] = torch.zeros(
                node_count, dtype=base_adjacency.dtype, device=base_adjacency.device
            )
            batch_adjacency[batch_index, :, feature_node_index] = torch.zeros(
                node_count, dtype=base_adjacency.dtype, device=base_adjacency.device
            )

    abs_batch_adjacency = batch_adjacency.abs()
    normalized_batch_adjacency = (
        abs_batch_adjacency
        / abs_batch_adjacency.sum(dim=2, keepdim=True).clamp(min=min_denominator)
    )
    batch_logit_weights = logit_weights.unsqueeze(0).expand(batch_size, -1)

    current_walk = torch.bmm(
        batch_logit_weights.unsqueeze(1), normalized_batch_adjacency
    ).squeeze(1)
    influence = current_walk.clone()
    for _ in range(max_influence_hops):
        current_walk = torch.bmm(current_walk.unsqueeze(1), normalized_batch_adjacency).squeeze(1)
        if current_walk.abs().max() < influence_stop_threshold:
            break
        influence = influence + current_walk

    token_influence = influence[:, error_end:token_end].sum(dim=1)
    error_influence = influence[:, error_start:error_end].sum(dim=1)
    replacement_scores = token_influence / (
        token_influence + error_influence
    ).clamp(min=min_denominator)

    non_error_outflow = 1.0 - normalized_batch_adjacency[:, :, error_start:error_end].sum(dim=2)
    outgoing_influence = influence + batch_logit_weights
    completeness_scores = (
        (non_error_outflow * outgoing_influence).sum(dim=1)
        / outgoing_influence.sum(dim=1).clamp(min=min_denominator)
    )
    return replacement_scores, completeness_scores


def prune_graph_for_scoring(nodes, links, target_token, keep_ratio=0.5):
    """Remove low-priority feature nodes before constructing dense scoring tensors.

    Dense scoring is O(n^2), so pruning has a large runtime effect. We always
    keep embeddings, reconstruction errors, and logits; feature nodes are ranked
    by target-logit proximity plus small influence/connectivity tie-breakers.
    """
    target_logit = None
    for node in nodes:
        if node.get("feature_type") == FEATURE_TYPE_LOGIT and target_token.strip() in (
            node.get("clerp") or ""
        ):
            target_logit = node
            break

    edges_from = {}
    for link in links:
        source_id = _link_endpoint_id(link, "source")
        target_id = _link_endpoint_id(link, "target")
        edges_from.setdefault(source_id, []).append((target_id, link["weight"]))

    feature_scores = {}
    target_logit_id = target_logit["node_id"] if target_logit else None
    features = [node for node in nodes if _is_feature_node(node)]

    for node in features:
        node_id = node["node_id"]
        score = 0.0

        if target_logit_id:
            for target_id, weight in edges_from.get(node_id, []):
                if target_id == target_logit_id:
                    score += abs(weight) * DIRECT_TARGET_EDGE_WEIGHT

            for target_id, weight in edges_from.get(node_id, []):
                if target_id == target_logit_id:
                    continue
                for second_hop_target_id, second_hop_weight in edges_from.get(target_id, []):
                    if second_hop_target_id == target_logit_id:
                        score += abs(weight * second_hop_weight) * INDIRECT_TARGET_EDGE_WEIGHT

        score += abs(node.get("influence", 0) or 0) * NODE_INFLUENCE_TIEBREAKER_WEIGHT

        total_edge_weight = sum(abs(weight) for _, weight in edges_from.get(node_id, []))
        score += total_edge_weight * EDGE_CONNECTIVITY_TIEBREAKER_WEIGHT

        feature_scores[node_id] = score

    num_features_to_keep = max(int(len(features) * keep_ratio), MIN_FEATURES_AFTER_PRUNING)
    ranked_features = sorted(feature_scores.items(), key=lambda item: -item[1])
    keep_ids = {feature_id for feature_id, _score in ranked_features[:num_features_to_keep]}

    keep_ids.update(node["node_id"] for node in nodes if not _is_feature_node(node))

    pruned_nodes = [node for node in nodes if node["node_id"] in keep_ids]
    pruned_links = []
    for link in links:
        source_id = _link_endpoint_id(link, "source")
        target_id = _link_endpoint_id(link, "target")
        if source_id in keep_ids and target_id in keep_ids:
            pruned_links.append(link)

    return pruned_nodes, pruned_links


class BatchGraphScorer:
    """Scores pin sets and builds IA+PC circuits from a static attribution graph."""

    def __init__(self, nodes_data, links_data, device_str="cpu"):
        self.device = torch.device(device_str)

        self.sorted_nodes = sorted(nodes_data, key=_sort_graph_node)
        self.node_id_to_idx = {n['node_id']: i for i, n in enumerate(self.sorted_nodes)}
        self.n = len(self.sorted_nodes)

        # Adjacency is indexed [target, source], matching attribution edge flow.
        base = torch.zeros(self.n, self.n, dtype=torch.float32)
        for link in links_data:
            source_id = _link_endpoint_id(link, 'source')
            target_id = _link_endpoint_id(link, 'target')
            source_index = self.node_id_to_idx.get(source_id)
            target_index = self.node_id_to_idx.get(target_id)
            if source_index is not None and target_index is not None:
                base[target_index, source_index] = link['weight']
        self.base_adjacency = base.to(self.device)

        self.n_features = sum(
            1 for node in self.sorted_nodes if node["feature_type"] == FEATURE_TYPE_TRANSCODER
        )
        self.n_errors = sum(
            1 for node in self.sorted_nodes if node["feature_type"] == FEATURE_TYPE_ERROR
        )
        self.n_tokens = sum(
            1 for node in self.sorted_nodes if node["feature_type"] == FEATURE_TYPE_EMBEDDING
        )
        self.n_logits = sum(
            1 for node in self.sorted_nodes if node["feature_type"] == FEATURE_TYPE_LOGIT
        )
        self.error_start = self.n_features
        self.error_end = self.error_start + self.n_errors
        self.token_end = self.error_end + self.n_tokens

        # Feature pinning replaces each unpinned feature with its same-layer/context error node.
        self.error_idx_by_key = {}
        for i, node in enumerate(self.sorted_nodes):
            if node['feature_type'] == FEATURE_TYPE_ERROR:
                self.error_idx_by_key[f"{node['layer']}|{node['ctx_idx']}"] = i

        self.feature_node_indices = []
        self.feature_ids = []
        self.feature_error_node_indices = []
        for i, node in enumerate(self.sorted_nodes):
            if node['feature_type'] == FEATURE_TYPE_TRANSCODER:
                self.feature_node_indices.append(i)
                self.feature_ids.append(node['node_id'])
                key = f"{node['layer']}|{node['ctx_idx']}"
                self.feature_error_node_indices.append(self.error_idx_by_key.get(key, -1))
        self.feature_node_indices_t = torch.tensor(
            self.feature_node_indices, dtype=torch.long, device=self.device
        )
        self.feature_error_node_indices_t = torch.tensor(
            self.feature_error_node_indices, dtype=torch.long, device=self.device
        )
        self.feature_id_set = set(self.feature_ids)
        self.feature_id_to_local = {fid: i for i, fid in enumerate(self.feature_ids)}
        self.num_features = len(self.feature_ids)

        logit_weights = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        logit_probs = [
            node["token_prob"]
            for node in self.sorted_nodes
            if node["feature_type"] == FEATURE_TYPE_LOGIT
        ]
        for i, probability in enumerate(logit_probs):
            logit_weights[self.n - self.n_logits + i] = probability
        self.logit_weights = logit_weights

        self.edges_from = {}
        self.edges_to = {}
        for link in links_data:
            source_id = _link_endpoint_id(link, 'source')
            target_id = _link_endpoint_id(link, 'target')
            self.edges_from.setdefault(source_id, []).append((target_id, abs(link['weight'])))
            self.edges_to.setdefault(target_id, []).append((source_id, abs(link['weight'])))

        target_logit = next(
            (
                node
                for node in self.sorted_nodes
                if node["feature_type"] == FEATURE_TYPE_LOGIT and node["token_prob"] > 0
            ),
            None,
        )
        self.target_influence = {}
        if target_logit:
            target_logit_id = target_logit['node_id']
            signed_edges_from = {}
            for link in links_data:
                source_id = _link_endpoint_id(link, 'source')
                target_id = _link_endpoint_id(link, 'target')
                signed_edges_from.setdefault(source_id, []).append((target_id, link['weight']))
            for node in self.sorted_nodes:
                if node['feature_type'] != FEATURE_TYPE_TRANSCODER:
                    continue
                score = 0.0
                for target_id, weight in signed_edges_from.get(node['node_id'], []):
                    if target_id == target_logit_id:
                        score += weight
                    else:
                        for second_hop_target_id, second_hop_weight in signed_edges_from.get(
                            target_id, []
                        ):
                            if second_hop_target_id == target_logit_id:
                                score += weight * second_hop_weight * INDIRECT_TARGET_EDGE_WEIGHT
                self.target_influence[node['node_id']] = score
        self.max_influence = max(
            (abs(v) for v in self.target_influence.values()),
            default=MIN_DENOMINATOR,
        )
        self.node_influence = {n['node_id']: n.get('influence', 0) or 0 for n in self.sorted_nodes}
        self._init_search_tensors(links_data)

    def _init_search_tensors(self, links_data):
        target_values = [self.target_influence.get(fid, 0) for fid in self.feature_ids]
        node_values = [self.node_influence.get(fid, 0) for fid in self.feature_ids]
        outgoing_abs = [
            sum(w for _, w in self.edges_from.get(fid, []))
            for fid in self.feature_ids
        ]

        self.target_influence_scores_t = torch.tensor(
            target_values, dtype=torch.float32, device=self.device
        )
        self.node_influence_scores_t = torch.tensor(
            node_values, dtype=torch.float32, device=self.device
        )
        self.feature_outgoing_abs_t = torch.tensor(
            outgoing_abs, dtype=torch.float32, device=self.device
        )
        self.search_priority_scores_t = (
            torch.clamp(
                self.target_influence_scores_t / max(self.max_influence, MIN_DENOMINATOR),
                min=0,
            )
            + self.node_influence_scores_t * SEARCH_NODE_INFLUENCE_WEIGHT
        )

        node_to_feature_local = torch.full((self.n,), -1, dtype=torch.long, device=self.device)
        if self.num_features:
            node_to_feature_local[self.feature_node_indices_t] = torch.arange(
                self.num_features, dtype=torch.long, device=self.device
            )
        self.node_to_feature_local_t = node_to_feature_local

        feature_edge_sources = []
        feature_edge_targets = []
        feature_edge_weights = []
        for link in links_data:
            source_id = _link_endpoint_id(link, 'source')
            target_id = _link_endpoint_id(link, 'target')
            source_local = self.feature_id_to_local.get(source_id)
            target_local = self.feature_id_to_local.get(target_id)
            if source_local is None or target_local is None:
                continue
            feature_edge_sources.append(source_local)
            feature_edge_targets.append(target_local)
            feature_edge_weights.append(abs(link['weight']))

        self.feature_edge_sources_t = torch.tensor(
            feature_edge_sources, dtype=torch.long, device=self.device
        )
        self.feature_edge_targets_t = torch.tensor(
            feature_edge_targets, dtype=torch.long, device=self.device
        )
        self.feature_edge_weights_t = torch.tensor(
            feature_edge_weights, dtype=torch.float32, device=self.device
        )

    def _build_pinned_mask(self, pinned_ids):
        mask = torch.zeros(self.num_features, dtype=torch.bool, device=self.device)
        for fid in pinned_ids:
            li = self.feature_id_to_local.get(fid)
            if li is not None:
                mask[li] = True
        return mask

    def _score_from_mask_jit(self, pinned_mask):
        r, c = _jit_score_single(
            self.base_adjacency,
            self.feature_node_indices_t,
            self.feature_error_node_indices_t,
            pinned_mask,
            self.logit_weights,
            self.error_start,
            self.error_end,
            self.token_end,
        )
        return r, c

    def _apply_pins_scatter(self, adj, pinned_mask):
        unpinned = ~pinned_mask
        unpinned_feature_nodes = self.feature_node_indices_t[unpinned]
        replacement_error_nodes = self.feature_error_node_indices_t[unpinned]
        has_replacement_error = replacement_error_nodes >= 0
        replaced_feature_nodes = unpinned_feature_nodes[has_replacement_error]
        target_error_nodes = replacement_error_nodes[has_replacement_error]
        if replaced_feature_nodes.numel() > 0:
            adj.scatter_add_(
                1,
                target_error_nodes.unsqueeze(0).expand(self.n, -1),
                adj[:, replaced_feature_nodes],
            )
        if unpinned_feature_nodes.numel() > 0:
            adj[unpinned_feature_nodes, :] = 0
            adj[:, unpinned_feature_nodes] = 0
        return adj

    def _solve_influence(self, norm_adj):
        current = self.logit_weights @ norm_adj
        influence = current.clone()
        for _ in range(MAX_SUGGESTION_INFLUENCE_HOPS):
            current = current @ norm_adj
            if current.abs().max() < INFLUENCE_STOP_THRESHOLD:
                break
            influence += current
        return influence

    def score_fast(self, pinned_ids=None, pinned_mask=None):
        """Return R/C scores without candidate suggestions."""
        if pinned_mask is None:
            pinned_mask = self._build_pinned_mask(pinned_ids or [])
        r, c = self._score_from_mask_jit(pinned_mask)
        return {
            'replacementScore': r if r == r else 0,
            'completenessScore': c if c == c else 0,
            'suggestedPinIds': [],
        }

    def score_with_suggestions(self, pinned_ids):
        """Return R/C scores plus feature IDs likely to improve completeness."""
        mask = self._build_pinned_mask(pinned_ids)
        adj = self.base_adjacency.clone()
        adj = self._apply_pins_scatter(adj, mask)
        abs_adj = adj.abs()
        norm = abs_adj / abs_adj.sum(dim=1, keepdim=True).clamp(min=MIN_DENOMINATOR)
        influence = self._solve_influence(norm)
        token_inf = influence[self.error_end:self.token_end].sum()
        error_inf = influence[self.error_start:self.error_end].sum()
        r = (token_inf / (token_inf + error_inf).clamp(min=MIN_DENOMINATOR)).item()
        non_err = 1.0 - norm[:, self.error_start:self.error_end].sum(dim=1)
        out_inf = influence + self.logit_weights
        c = ((non_err * out_inf).sum() / out_inf.sum().clamp(min=MIN_DENOMINATOR)).item()

        pinned_set = set(pinned_ids)
        suggested = []
        if pinned_ids:
            error_infs = [(self.sorted_nodes[i]['node_id'], influence[i].item(),
                          self.sorted_nodes[i]['layer'], self.sorted_nodes[i]['ctx_idx'])
                         for i in range(self.error_start, self.error_end)]
            error_infs.sort(key=lambda x: -x[1])
            top_keys = {f"{e[2]}|{e[3]}" for e in error_infs[:10]}
            cand_scores = {}
            for node in self.sorted_nodes:
                if node['feature_type'] != FEATURE_TYPE_TRANSCODER or node['node_id'] in pinned_set:
                    continue
                s = 0.0
                if f"{node['layer']}|{node['ctx_idx']}" in top_keys:
                    s += (
                        sum(w for _, w in self.edges_from.get(node['node_id'], []))
                        * SUGGESTION_OUTGOING_EDGE_WEIGHT
                    )
                s += self.node_influence.get(node['node_id'], 0) * NODE_INFLUENCE_TIEBREAKER_WEIGHT
                if s > 0:
                    cand_scores[node['node_id']] = s
            suggested = sorted(cand_scores, key=lambda x: -cand_scores[x])[:50]

        return {
            'replacementScore': r if r == r else 0,
            'completenessScore': c if c == c else 0,
            'suggestedPinIds': suggested,
        }

    def score(self, pinned_ids):
        return self.score_with_suggestions(pinned_ids)

    def score_batch(self, pinned_ids_list):
        B = len(pinned_ids_list)
        if B == 0:
            return []
        masks = torch.stack([self._build_pinned_mask(pins) for pins in pinned_ids_list])
        return self.score_batch_masks(masks)

    def score_batch_masks(self, masks):
        B = masks.shape[0]
        if B == 0:
            return []
        r_scores, c_scores = _jit_score_batch(
            self.base_adjacency,
            self.feature_node_indices_t,
            self.feature_error_node_indices_t,
            masks,
            self.logit_weights,
            self.error_start,
            self.error_end,
            self.token_end,
        )
        return [
            {
                'replacementScore': r_scores[b].item() if r_scores[b] == r_scores[b] else 0,
                'completenessScore': c_scores[b].item() if c_scores[b] == c_scores[b] else 0,
                'suggestedPinIds': [],
            }
            for b in range(B)
        ]

    def _score_mask_with_local_suggestions(self, pinned_mask):
        """Score a mask and return suggestions as local feature indices for search."""
        adj = self.base_adjacency.clone()
        adj = self._apply_pins_scatter(adj, pinned_mask)
        abs_adj = adj.abs()
        norm = abs_adj / abs_adj.sum(dim=1, keepdim=True).clamp(min=MIN_DENOMINATOR)
        influence = self._solve_influence(norm)
        token_inf = influence[self.error_end:self.token_end].sum()
        error_inf = influence[self.error_start:self.error_end].sum()
        r = (token_inf / (token_inf + error_inf).clamp(min=MIN_DENOMINATOR)).item()
        non_err = 1.0 - norm[:, self.error_start:self.error_end].sum(dim=1)
        out_inf = influence + self.logit_weights
        c = ((non_err * out_inf).sum() / out_inf.sum().clamp(min=MIN_DENOMINATOR)).item()

        suggested_local = torch.empty(0, dtype=torch.long, device=self.device)
        if pinned_mask.any().item() and self.n_errors > 0 and self.num_features > 0:
            error_scores = influence[self.error_start:self.error_end]
            top_count = min(10, error_scores.numel())
            if top_count > 0:
                _, top_error_offsets = torch.topk(error_scores, k=top_count)
                top_error_nodes = top_error_offsets + self.error_start
                related_to_top_error = (
                    self.feature_error_node_indices_t[:, None] == top_error_nodes[None, :]
                ).any(dim=1)
                candidate_scores = torch.where(
                    related_to_top_error,
                    self.feature_outgoing_abs_t * SUGGESTION_OUTGOING_EDGE_WEIGHT,
                    torch.zeros_like(self.feature_outgoing_abs_t),
                )
                candidate_scores = (
                    candidate_scores
                    + self.node_influence_scores_t * NODE_INFLUENCE_TIEBREAKER_WEIGHT
                )
                candidate_scores = candidate_scores.masked_fill(
                    pinned_mask | (candidate_scores <= 0),
                    -float("inf"),
                )
                top_suggestions = min(50, candidate_scores.numel())
                if top_suggestions > 0 and torch.isfinite(candidate_scores).any().item():
                    _, suggested_local = torch.topk(candidate_scores, k=top_suggestions)
                    suggested_local = suggested_local[
                        torch.isfinite(candidate_scores[suggested_local])
                    ]

        return {
            'replacementScore': r if r == r else 0,
            'completenessScore': c if c == c else 0,
            'suggestedLocal': suggested_local,
        }

    def _top_unpinned_by_score(self, scores, pinned_mask, k, positive_only=False):
        if k <= 0 or scores.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        masked_scores = scores.masked_fill(pinned_mask, -float("inf"))
        if positive_only:
            masked_scores = masked_scores.masked_fill(masked_scores <= 0, -float("inf"))
        if not torch.isfinite(masked_scores).any().item():
            return torch.empty(0, dtype=torch.long, device=self.device)
        _, idx = torch.topk(masked_scores, k=min(k, masked_scores.numel()))
        return idx[torch.isfinite(masked_scores[idx])]

    def _feature_ids_from_locals(self, local_indices):
        return [self.feature_ids[int(i)] for i in local_indices.detach().cpu().tolist()]

    def _pathway_completion_candidates(self, pinned_mask, pc_candidates):
        """Rank neighboring unpinned features for the PC pass."""
        if pc_candidates <= 0 or self.feature_edge_sources_t.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        weights = torch.zeros(self.num_features, dtype=torch.float32, device=self.device)
        outgoing = pinned_mask[self.feature_edge_sources_t]
        if outgoing.any().item():
            weights.scatter_add_(
                0,
                self.feature_edge_targets_t[outgoing],
                self.feature_edge_weights_t[outgoing],
            )
        incoming = pinned_mask[self.feature_edge_targets_t]
        if incoming.any().item():
            weights.scatter_add_(
                0,
                self.feature_edge_sources_t[incoming],
                self.feature_edge_weights_t[incoming] * PATHWAY_COMPLETION_INCOMING_EDGE_WEIGHT,
            )

        weights = weights.masked_fill(pinned_mask | (weights <= 0), -float("inf"))
        if not torch.isfinite(weights).any().item():
            return torch.empty(0, dtype=torch.long, device=self.device)
        _, candidates = torch.topk(weights, k=min(pc_candidates, weights.numel()))
        return candidates[torch.isfinite(weights[candidates])]

    def build_circuit_ia_pc(
        self,
        endpoint_ids,
        alpha=0.15,
        max_steps=80,
        stagnant_limit=8,
        pc_passes=1,
        pc_candidates=30,
        progress_callback=None,
    ):
        """Build one circuit with influence-aware greedy search plus PC.

        `endpoint_ids` may contain non-feature graph endpoints. They are kept in
        the returned `pinnedIds`, while scoring/search masks only track feature
        pins.
        """
        pinned = list(dict.fromkeys(endpoint_ids))
        pinned_set = set(pinned)
        pinned_mask = self._build_pinned_mask(pinned)

        promoting = self._top_unpinned_by_score(
            self.target_influence_scores_t, pinned_mask, 5, positive_only=True
        )
        for fid in self._feature_ids_from_locals(promoting):
            if fid not in pinned_set:
                pinned.append(fid)
                pinned_set.add(fid)
                pinned_mask[self.feature_id_to_local[fid]] = True

        scores = self._score_mask_with_local_suggestions(pinned_mask)
        best_c = scores["completenessScore"]
        stagnant = 0

        prev_checkpoint_c = best_c
        for step in range(max_steps):
            if step % 20 == 0 and step > 0:
                c_improvement = best_c - prev_checkpoint_c
                if c_improvement < 0.005 and step >= 40:
                    if progress_callback:
                        progress_callback(step, max_steps, len(pinned_set), best_c)
                    break
                prev_checkpoint_c = best_c

            candidate_mask = torch.zeros(self.num_features, dtype=torch.bool, device=self.device)
            suggested_local = scores["suggestedLocal"]
            if suggested_local.numel() > 0:
                candidate_mask[suggested_local] = True
                candidate_mask[pinned_mask] = False

            influence_ranked_candidates = self._top_unpinned_by_score(
                self.target_influence_scores_t,
                pinned_mask | candidate_mask,
                10,
                positive_only=True,
            )
            if influence_ranked_candidates.numel() > 0:
                candidate_mask[influence_ranked_candidates] = True

            if not candidate_mask.any().item():
                fallback = self._top_unpinned_by_score(
                    self.node_influence_scores_t, pinned_mask, 1, positive_only=False
                )
                if fallback.numel() == 0:
                    break
                candidate_mask[fallback] = True

            candidate_priority_scores = self.search_priority_scores_t.masked_fill(
                ~candidate_mask,
                -float("inf"),
            )
            _, ranked_candidates = torch.topk(
                candidate_priority_scores,
                k=min(5, int(candidate_mask.sum().item())),
            )
            ranked_candidates = ranked_candidates[
                torch.isfinite(candidate_priority_scores[ranked_candidates])
            ]
            if ranked_candidates.numel() == 0:
                break

            trial_masks = pinned_mask.unsqueeze(0).repeat(ranked_candidates.numel(), 1)
            trial_masks[
                torch.arange(ranked_candidates.numel(), device=self.device),
                ranked_candidates,
            ] = True
            batch = self.score_batch_masks(trial_masks)

            best_cand = None
            best_combined = -float("inf")
            best_trial_c = best_c
            for idx, local_idx_t in enumerate(ranked_candidates):
                local_idx = int(local_idx_t.item())
                trial_c = batch[idx]["completenessScore"]
                c_improvement = trial_c - best_c
                normalized_influence = max(
                    float(self.target_influence_scores_t[local_idx].item()) / self.max_influence,
                    0,
                )
                combined = c_improvement + alpha * normalized_influence
                if combined > best_combined and trial_c >= best_c * 0.995:
                    best_cand = local_idx
                    best_combined = combined
                    best_trial_c = trial_c

            if best_cand is not None and best_combined > 0:
                fid = self.feature_ids[best_cand]
                pinned.append(fid)
                pinned_set.add(fid)
                pinned_mask[best_cand] = True
                best_c = best_trial_c
                scores = self._score_mask_with_local_suggestions(pinned_mask)
                stagnant = 0
            else:
                stagnant += 1
                if stagnant >= stagnant_limit:
                    if progress_callback:
                        progress_callback(step + 1, max_steps, len(pinned_set), best_c)
                    break

            if progress_callback:
                progress_callback(step + 1, max_steps, len(pinned_set), best_c)

        ia_count = int(pinned_mask.sum().item())
        current_c = best_c
        for _ in range(pc_passes):
            candidates = self._pathway_completion_candidates(pinned_mask, pc_candidates)
            if candidates.numel() == 0:
                break

            trial_masks = pinned_mask.unsqueeze(0).repeat(candidates.numel(), 1)
            trial_masks[torch.arange(candidates.numel(), device=self.device), candidates] = True
            batch = self.score_batch_masks(trial_masks)

            added = 0
            for idx, local_idx_t in enumerate(candidates):
                local_idx = int(local_idx_t.item())
                if pinned_mask[local_idx].item():
                    continue
                if batch[idx]["completenessScore"] >= current_c * 0.98:
                    cid = self.feature_ids[local_idx]
                    pinned.append(cid)
                    pinned_set.add(cid)
                    pinned_mask[local_idx] = True
                    current_c = batch[idx]["completenessScore"]
                    added += 1
            if added == 0:
                break

        total_features = int(pinned_mask.sum().item())
        pc_count = total_features - ia_count
        final = self.score_fast(pinned_mask=pinned_mask)
        return {
            "pinnedIds": pinned,
            "iaFeatures": ia_count,
            "pcFeatures": pc_count,
            "totalFeatures": total_features,
            "replacementScore": final["replacementScore"],
            "completenessScore": final["completenessScore"],
        }
