import { useGraphContext } from '@/components/provider/graph-provider';
import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, test, vi } from 'vitest';
import { CLTGraph, CLTGraphNode } from '../../../graph-types';
import { useCircuitExplorerContext } from '../context';
import { useExploreCircuits } from '../use-explore-circuits';

vi.mock('@/components/provider/graph-provider', () => ({
  useGraphContext: vi.fn(),
}));

vi.mock('../context', () => ({
  useCircuitExplorerContext: vi.fn(),
}));

const encoder = new TextEncoder();

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
  };
}

function graph(): CLTGraph {
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
      node({ node_id: 'embedding-a', feature_type: 'embedding' }),
      node({ node_id: 'feature-a', clerp: 'remote label' }),
      node({ node_id: 'logit-a', feature_type: 'logit', is_target_logit: true, token_prob: 0.7 }),
    ],
    links: [{ source: 'feature-a', target: 'logit-a', weight: 0.5 }],
  };
}

function mockReader(chunks: string[]) {
  const reads = chunks.map((chunk) => Promise.resolve({ done: false, value: encoder.encode(chunk) }));
  reads.push(Promise.resolve({ done: true, value: undefined }));

  return {
    read: vi.fn(() => reads.shift() as Promise<ReadableStreamReadResult<Uint8Array>>),
  };
}

function mockFetchResponse(chunks: string[], ok = true, text = '') {
  const reader = mockReader(chunks);
  vi.mocked(fetch).mockResolvedValue({
    ok,
    text: vi.fn().mockResolvedValue(text),
    body: ok
      ? {
          getReader: () => reader,
        }
      : null,
  } as unknown as Response);
  return reader;
}

describe('useExploreCircuits', () => {
  const dispatch = vi.fn();
  const setIsExploreCircuitsModalOpen = vi.fn();
  const selectedGraph = graph();

  beforeEach(() => {
    vi.restoreAllMocks();
    vi.stubGlobal('fetch', vi.fn());
    dispatch.mockReset();
    setIsExploreCircuitsModalOpen.mockReset();

    vi.mocked(useCircuitExplorerContext).mockReturnValue({
      isExploreCircuitsModalOpen: true,
      exploredCircuits: [],
      exploreProgress: null,
      dispatch,
      setIsExploreCircuitsModalOpen,
    });
    vi.mocked(useGraphContext).mockReturnValue({
      selectedGraph,
      selectedModelId: 'gemma-2-2b',
      selectedSourceSetName: 'gemmascope',
      getOverrideClerpForNode: (graphNode: CLTGraphNode) =>
        graphNode.node_id === 'feature-a' ? 'local override label' : undefined,
    } as ReturnType<typeof useGraphContext>);
  });

  test('posts graph data and dispatches chunked SSE events', async () => {
    mockFetchResponse([
      'event: status\r\n',
      'data: {"phase":"exploring","total_seeds":2,"completed_seeds":0,"circuits_found":0}\r\n\r\n',
      'event: progress\n',
      'data: {"current_seed":1,"total_seeds":2,"step":3,"max_steps":8,"features":4,"c_score":0.42}\n\n',
      'event: circuit\n',
      'data: {"pinned_ids":["feature-a"],"replacement_score":0.2,"completeness_score":0.3}\n\n',
      'event: grouped\n',
      'data: {"circuit_id":"feature-a","circuit":{"pinned_ids":["feature-a"],"supernodes":[]}}\n\n',
      'event: done\n',
      'data: {"circuits":[{"pinned_ids":["feature-a"],"replacement_score":0.2,"completeness_score":0.3}]}\n\n',
    ]);

    const { result } = renderHook(() =>
      useExploreCircuits({
        activeLogitId: 'logit-a',
        numSeeds: 2,
        maxSteps: 8,
        pcPasses: 3,
        keepRatio: 0.5,
        groupingModel: 'haiku',
      }),
    );

    await act(async () => {
      await result.current.handleExplore();
    });

    expect(fetch).toHaveBeenCalledTimes(1);
    const [url, request] = vi.mocked(fetch).mock.calls[0];
    const body = JSON.parse((request as RequestInit).body as string);

    expect(url).toBe('/api/graph/explore-circuits');
    expect(body).toMatchObject({
      model_id: 'gemma-2-2b',
      modelId: 'gemma-2-2b',
      sourceSetName: 'gemmascope',
      target_logit_node_id: 'logit-a',
      endpoint_node_ids: ['embedding-a', 'logit-a'],
      num_seeds: 2,
      max_iterations: 8,
      pc_passes: 3,
      keep_ratio: 0.5,
      grouping_model: 'haiku',
    });
    expect(
      body.graph_data.nodes.find((payloadNode: { node_id: string }) => payloadNode.node_id === 'feature-a'),
    ).toMatchObject({
      clerp: 'local override label',
    });

    expect(dispatch).toHaveBeenNthCalledWith(1, { type: 'START_EXPLORATION' });
    expect(setIsExploreCircuitsModalOpen).toHaveBeenCalledWith(false);
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'STATUS_EVENT',
        data: expect.objectContaining({ phase: 'exploring', total_seeds: 2 }),
      }),
    );
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'PROGRESS_EVENT',
        data: expect.objectContaining({ step: 3, c_score: 0.42 }),
      }),
    );
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'CIRCUIT_EVENT',
        data: expect.objectContaining({ pinned_ids: ['feature-a'] }),
        defaultGroupingModel: 'haiku',
      }),
    );
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'GROUPED_EVENT',
        data: expect.objectContaining({ circuit_id: 'feature-a' }),
      }),
    );
    expect(dispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'DONE_EVENT',
        data: expect.objectContaining({ circuits: [expect.objectContaining({ pinned_ids: ['feature-a'] })] }),
      }),
    );
    expect(dispatch).toHaveBeenLastCalledWith({ type: 'CLEAR_PROGRESS' });
    expect(result.current.error).toBeNull();
    expect(result.current.isExploring).toBe(false);
  });

  test('surfaces non-OK response text and does not close the modal', async () => {
    mockFetchResponse([], false, 'graph server unavailable');

    const { result } = renderHook(() =>
      useExploreCircuits({
        activeLogitId: null,
        numSeeds: 2,
        maxSteps: 8,
        pcPasses: 1,
        keepRatio: 0.4,
        groupingModel: 'sonnet',
      }),
    );

    await act(async () => {
      await result.current.handleExplore();
    });

    expect(result.current.error).toBe('Exploration failed: graph server unavailable');
    expect(result.current.isExploring).toBe(false);
    expect(setIsExploreCircuitsModalOpen).not.toHaveBeenCalled();
    expect(dispatch).toHaveBeenNthCalledWith(1, { type: 'START_EXPLORATION' });
    expect(dispatch).toHaveBeenLastCalledWith({ type: 'CLEAR_PROGRESS' });
  });

  test('ignores malformed SSE events and clears progress after stream completion', async () => {
    mockFetchResponse(['event: circuit\n', 'data: not-json\n\n', 'event: done\n', 'data: {"circuits":[]}\n\n']);

    const { result } = renderHook(() =>
      useExploreCircuits({
        activeLogitId: null,
        numSeeds: 2,
        maxSteps: 8,
        pcPasses: 1,
        keepRatio: 0.4,
        groupingModel: 'sonnet',
      }),
    );

    await act(async () => {
      await result.current.handleExplore();
    });

    expect(dispatch).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'CIRCUIT_EVENT' }));
    expect(dispatch).toHaveBeenCalledWith(expect.objectContaining({ type: 'DONE_EVENT' }));
    expect(dispatch).toHaveBeenLastCalledWith({ type: 'CLEAR_PROGRESS' });
    expect(result.current.error).toBeNull();
  });
});
