import { describe, expect, test } from 'vitest';
import { CLTGraph, CLTGraphLink, CLTGraphNode } from '../../../graph-types';
import { buildExploreGraphData, buildGroupingGraphData } from '../graph-payload';

function node(overrides: Partial<CLTGraphNode>): CLTGraphNode {
  return {
    node_id: overrides.node_id || 'node-1',
    feature: overrides.feature || 0,
    layer: overrides.layer || '1',
    ctx_idx: overrides.ctx_idx || 0,
    feature_type: overrides.feature_type || 'cross layer transcoder',
    token_prob: overrides.token_prob || 0,
    is_target_logit: overrides.is_target_logit || false,
    run_idx: 0,
    reverse_ctx_idx: 0,
    jsNodeId: overrides.jsNodeId || overrides.node_id || 'node-1',
    clerp: overrides.clerp || '',
    influence: overrides.influence,
  };
}

function graph(overrides: Partial<CLTGraph> = {}): CLTGraph {
  return {
    metadata: {
      slug: 'test-graph',
      scan: 'test-scan',
      prompt_tokens: ['Dallas', ' is'],
      prompt: 'Dallas is',
    },
    qParams: {
      linkType: '',
      pinnedIds: [],
      clickedId: '',
      supernodes: [],
      sg_pos: '',
    },
    nodes: [
      node({
        node_id: 'feature-a',
        feature: 123,
        layer: '12',
        ctx_idx: 3,
        clerp: 'remote label',
        influence: 0.4,
      }),
      node({
        node_id: 'logit-a',
        feature_type: 'logit',
        token_prob: 0.8,
        is_target_logit: true,
        clerp: ' Austin',
      }),
    ],
    links: [
      { source: 'feature-a', target: 'logit-a', weight: 0.5 },
      {
        source: { node_id: 'embedding-a' },
        target: { node_id: 'feature-a' },
        weight: -0.25,
      } as unknown as CLTGraphLink,
    ],
    ...overrides,
  };
}

describe('buildExploreGraphData', () => {
  test('serializes scoring payload fields and applies override labels', () => {
    const selectedGraph = graph();

    const payload = buildExploreGraphData(selectedGraph, (graphNode) =>
      graphNode.node_id === 'feature-a' ? 'local override label' : undefined,
    );

    expect(payload.metadata).toBe(selectedGraph.metadata);
    expect(payload.nodes).toEqual([
      {
        node_id: 'feature-a',
        feature: 123,
        layer: '12',
        ctx_idx: 3,
        feature_type: 'cross layer transcoder',
        token_prob: 0,
        is_target_logit: false,
        influence: 0.4,
        clerp: 'local override label',
      },
      {
        node_id: 'logit-a',
        feature: 0,
        layer: '1',
        ctx_idx: 0,
        feature_type: 'logit',
        token_prob: 0.8,
        is_target_logit: true,
        influence: undefined,
        clerp: ' Austin',
      },
    ]);
    expect(payload.links).toEqual([
      { source: 'feature-a', target: 'logit-a', weight: 0.5 },
      { source: 'embedding-a', target: 'feature-a', weight: -0.25 },
    ]);
  });
});

describe('buildGroupingGraphData', () => {
  test('sends only grouping-relevant node fields and serializes endpoint refs', () => {
    const payload = buildGroupingGraphData(graph(), () => undefined);

    expect(payload.nodes).toEqual([
      {
        node_id: 'feature-a',
        feature_type: 'cross layer transcoder',
        clerp: 'remote label',
        layer: '12',
        ctx_idx: 3,
      },
      {
        node_id: 'logit-a',
        feature_type: 'logit',
        clerp: ' Austin',
        layer: '1',
        ctx_idx: 0,
      },
    ]);
    expect(payload.links).toEqual([
      { source: 'feature-a', target: 'logit-a', weight: 0.5 },
      { source: 'embedding-a', target: 'feature-a', weight: -0.25 },
    ]);
  });
});
