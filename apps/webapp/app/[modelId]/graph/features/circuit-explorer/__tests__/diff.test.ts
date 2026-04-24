import { describe, expect, test } from 'vitest';
import { CLTGraphNode } from '../../../graph-types';
import { computeCircuitDiff } from '../diff';
import { ExploredCircuit } from '../types';

function node(node_id: string, feature_type = 'cross layer transcoder'): CLTGraphNode {
  return {
    node_id,
    feature: 0,
    layer: '1',
    ctx_idx: 0,
    feature_type,
    token_prob: 0,
    is_target_logit: false,
    run_idx: 0,
    reverse_ctx_idx: 0,
    jsNodeId: node_id,
    clerp: node_id,
  };
}

function circuit(overrides: Partial<ExploredCircuit>): ExploredCircuit {
  const pinnedIds = overrides.pinned_ids || [];

  return {
    rank: overrides.rank || 1,
    pinned_ids: pinnedIds,
    replacement_score: 0,
    completeness_score: 0,
    combined_score: 0,
    node_count: pinnedIds.length,
    seed: '',
    circuit_id: overrides.circuit_id || pinnedIds.join(','),
    supernodes: overrides.supernodes || [],
    supernode_explanations: [],
    supernode_member_reasons: [],
    grouping_model: 'sonnet',
    grouping_failed: false,
    grouping_error: null,
    ready: true,
    isActiveGrouping: false,
  };
}

describe('computeCircuitDiff', () => {
  test('returns null until both selected ranks are valid', () => {
    const circuits = [circuit({ rank: 1, pinned_ids: ['feature-a'] })];
    const selectedGraph = { nodes: [node('feature-a')] };

    expect(computeCircuitDiff(circuits, [null, 1], selectedGraph)).toBeNull();
    expect(computeCircuitDiff(circuits, [1, 2], selectedGraph)).toBeNull();
  });

  test('compares only real feature nodes from the selected graph', () => {
    const selectedGraph = {
      nodes: [node('shared'), node('only-a'), node('only-b'), node('logit-1', 'logit')],
    };
    const circuits = [
      circuit({ rank: 1, pinned_ids: ['shared', 'only-a', 'logit-1', 'missing-node'] }),
      circuit({ rank: 2, pinned_ids: ['shared', 'only-b', 'logit-1'] }),
    ];

    const diff = computeCircuitDiff(circuits, [1, 2], selectedGraph);

    expect(diff?.shared).toEqual(['shared']);
    expect(diff?.onlyA).toEqual(['only-a']);
    expect(diff?.onlyB).toEqual(['only-b']);
  });

  test('matches supernodes by overlapping members and reports unmatched groups', () => {
    const selectedGraph = { nodes: [node('a'), node('b'), node('c'), node('d')] };
    const circuits = [
      circuit({
        rank: 1,
        pinned_ids: ['a', 'b'],
        supernodes: [
          ['Geography', 'a', 'b'],
          ['Sports', 'c'],
        ],
      }),
      circuit({
        rank: 2,
        pinned_ids: ['a', 'd'],
        supernodes: [
          ['Places', 'a', 'd'],
          ['Numbers', 'd'],
        ],
      }),
    ];

    const diff = computeCircuitDiff(circuits, [1, 2], selectedGraph);

    expect(diff?.supernodeDiff).toEqual([
      {
        label: 'Geography',
        inA: true,
        inB: true,
        membersA: ['a', 'b'],
        membersB: ['a', 'd'],
      },
      {
        label: 'Sports',
        inA: true,
        inB: false,
        membersA: ['c'],
        membersB: [],
      },
      {
        label: 'Numbers',
        inA: false,
        inB: true,
        membersA: [],
        membersB: ['d'],
      },
    ]);
  });
});
