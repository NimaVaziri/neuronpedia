import json
import os
import time
from typing import Any


def format_grouping_response(
    supernodes: list[list[str]],
    explanations: list[str],
    member_reasons: list[dict[str, str]],
    grouping_model: str,
    grouping_failed: bool,
    grouping_error: str | None,
) -> dict[str, Any]:
    return {
        "supernodes": supernodes,
        "supernode_explanations": explanations,
        "supernode_member_reasons": member_reasons,
        "grouping_model": grouping_model,
        "grouping_failed": grouping_failed,
        "grouping_error": grouping_error,
    }


def auto_group_nodes(nodes, links, pinned_ids, prompt="", grouping_model="sonnet"):
    """
    Use an Anthropic model to semantically group pinned feature nodes into supernodes.
    Falls back to no grouping if the API call fails.

    Returns:
      supernodes, explanations, member_reasons, grouping_failed, grouping_error
    """
    import anthropic

    _ = links
    anthropic_model = {
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-20250514",
    }.get(grouping_model, "claude-sonnet-4-20250514")

    pinned_set = set(pinned_ids)
    node_descriptions = []
    for node in nodes:
        if node["node_id"] not in pinned_set:
            continue
        if node["feature_type"] not in ("cross layer transcoder", "embedding", "logit"):
            continue
        node_descriptions.append(
            {
                "id": node["node_id"],
                "type": node["feature_type"],
                "layer": node.get("layer", "?"),
                "position": node.get("ctx_idx", "?"),
                "label": node.get("clerp", "") or "",
            }
        )

    if len(node_descriptions) < 3:
        return [], [], [], False, None

    node_list = "\n".join(
        f'  {desc["id"]} (L{desc["layer"]} pos{desc["position"]} {desc["type"]}): "{desc["label"]}"'
        for desc in node_descriptions
    )
    prompt_started_at = time.time()
    print(
        f"[auto_group_nodes] Starting {grouping_model} grouping for {len(node_descriptions)} nodes "
        f"({sum(1 for desc in node_descriptions if desc['type'] == 'cross layer transcoder')} feature, "
        f"{sum(1 for desc in node_descriptions if desc['type'] == 'embedding')} embedding, "
        f"{sum(1 for desc in node_descriptions if desc['type'] == 'logit')} logit) "
        f"with model {anthropic_model}",
        flush=True,
    )

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ANTHROPIC_API_KEY not set, skipping LLM grouping")
        return [], [], [], True, "ANTHROPIC_API_KEY not set"

    try:
        client = anthropic.Anthropic(api_key=api_key)
        api_started_at = time.time()
        response = client.messages.create(
            model=anthropic_model,
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": f"""Group ALL of these neural network features into semantic clusters. The features come from a circuit that processes: "{prompt}"

Features:
{node_list}

Group features that serve a similar role in the computation (e.g., features detecting the same concept, features at the same processing stage, features related to the same entity).

Rules:
- EVERY feature must be assigned to exactly one group — do not skip any
- Use the EXACT feature IDs as shown (e.g., "18_187_10") — do not modify them
- Each group needs a short label (1-3 words) describing the shared concept
- Each group needs a brief explanation (1-2 sentences) of why these features belong together
- Each member needs a brief reason (1 short sentence) explaining why that specific feature belongs in this group
- Group features that detect or represent the same concept together, even if they are at different layers
- Embedding and logit nodes should NOT be grouped with feature nodes

Return ONLY a JSON array of groups with this structure:
[{{"label": "Texas", "explanation": "These features all detect or represent the state of Texas and its geographic properties.", "members": [{{"id": "18_187_10", "reason": "Detects references to Texas as a US state"}}, {{"id": "18_1437_10", "reason": "Activates on Texas-related geographic context"}}]}}]

Return ONLY the JSON array, no other text."""
                }
            ],
        )
        api_elapsed = time.time() - api_started_at
        result_text = response.content[0].text.strip()
        print(
            f"[auto_group_nodes] Anthropic response received in {api_elapsed:.2f}s "
            f"with {len(result_text)} response chars",
            flush=True,
        )

        if "```" in result_text:
            parts = result_text.split("```")
            if len(parts) >= 3:
                result_text = parts[1]
            elif len(parts) == 2:
                result_text = parts[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()

        groups = None
        try:
            groups = json.loads(result_text)
        except json.JSONDecodeError:
            start = result_text.find("[")
            end = result_text.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    groups = json.loads(result_text[start : end + 1])
                except json.JSONDecodeError:
                    pass

        if groups is None:
            print("[auto_group_nodes] Failed to parse LLM response, falling back")
            return [], [], [], True, "Failed to parse grouping model response"

        valid_ids = {desc["id"] for desc in node_descriptions}
        supernodes = []
        supernode_explanations = []
        supernode_member_reasons = []
        for group in groups:
            if isinstance(group, dict):
                label = str(group.get("label", ""))
                explanation = str(group.get("explanation", ""))
                raw_members = group.get("members", [])
                member_ids = []
                reasons = {}
                for member in raw_members:
                    if isinstance(member, dict):
                        member_id = member.get("id", "")
                        if member_id in valid_ids:
                            member_ids.append(member_id)
                            reasons[member_id] = member.get("reason", "")
                    elif isinstance(member, str) and member in valid_ids:
                        member_ids.append(member)
            elif isinstance(group, list) and len(group) >= 3:
                label = str(group[0])
                explanation = ""
                member_ids = [member_id for member_id in group[1:] if member_id in valid_ids]
                reasons = {}
            else:
                continue

            if len(member_ids) >= 2:
                supernodes.append([label] + member_ids)
                supernode_explanations.append(explanation)
                supernode_member_reasons.append(reasons)

        total_elapsed = time.time() - prompt_started_at
        print(
            f"LLM grouped {sum(len(supernode) - 1 for supernode in supernodes)} nodes into {len(supernodes)} groups "
            f"in {total_elapsed:.2f}s",
            flush=True,
        )
        return supernodes, supernode_explanations, supernode_member_reasons, False, None

    except Exception as error:
        print(f"LLM grouping failed: {error}, falling back to no grouping")
        return [], [], [], True, str(error)
