// Lightweight worker to compute subgraph scores off the main thread.
// Duplicates the core scoring logic to avoid bringing in DOM/d3 dependencies.

import { CLTGraphLink, CLTGraphNode, ErrorEdgeAnalysis, ErrorNodeInfluence, GraphScores } from './graph-types';

type WorkerGraph = {
  nodes: CLTGraphNode[];
  links: CLTGraphLink[];
};

export type WorkerScoreResult = GraphScores;

function normalizeMatrix(matrix: number[][]): number[][] {
  return matrix.map((row) => {
    const absRow = row.map((val) => Math.abs(val));
    const sum = absRow.reduce((acc, val) => acc + val, 0);
    const clampedSum = Math.max(sum, 1e-10);
    return absRow.map((val) => val / clampedSum);
  });
}

function computeInfluence(A: number[][], logitWeights: number[], maxIter: number = 1000): number[] {
  let currentInfluence = new Array(A[0].length).fill(0);
  for (let j = 0; j < A[0].length; j += 1) {
    for (let i = 0; i < logitWeights.length; i += 1) {
      currentInfluence[j] += logitWeights[i] * A[i][j];
    }
  }

  const influence = [...currentInfluence];
  let iterations = 0;

  while (currentInfluence.some((val) => val !== 0)) {
    if (iterations >= maxIter) {
      throw new Error(`Influence computation failed to converge after ${iterations} iterations`);
    }

    const newInfluence = new Array(A[0].length).fill(0);
    for (let j = 0; j < A[0].length; j += 1) {
      for (let i = 0; i < currentInfluence.length; i += 1) {
        newInfluence[j] += currentInfluence[i] * A[i][j];
      }
    }

    currentInfluence = newInfluence;
    for (let i = 0; i < influence.length; i += 1) {
      influence[i] += currentInfluence[i];
    }
    iterations += 1;
  }

  return influence;
}

function reconstructAdjacencyMatrix(
  nodes: CLTGraphNode[],
  edges: CLTGraphLink[],
): { matrix: number[][]; sortedNodes: CLTGraphNode[] } {
  const featureTypeOrder = [
    'cross layer transcoder',
    'lorsa',
    'mlp reconstruction error',
    'lorsa error',
    'embedding',
    'logit',
  ];

  function getSortKey(node: CLTGraphNode) {
    let typePriority = featureTypeOrder.indexOf(node.feature_type);
    if (typePriority === -1) typePriority = featureTypeOrder.length;
    const layerNum = node.layer === 'E' ? 0 : Number.isNaN(parseInt(node.layer, 10)) ? 999 : parseInt(node.layer, 10);
    return [typePriority, layerNum, node.ctx_idx, node.feature || 0];
  }

  const sortedNodes = [...nodes].sort((a, b) => {
    const keyA = getSortKey(a);
    const keyB = getSortKey(b);
    for (let i = 0; i < Math.max(keyA.length, keyB.length); i += 1) {
      const valA = (keyA as number[])[i] || 0;
      const valB = (keyB as number[])[i] || 0;
      if (valA !== valB) return valA - valB;
    }
    return 0;
  });

  const nodeIdToIdx: Record<string, number> = {};
  sortedNodes.forEach((node, idx) => {
    nodeIdToIdx[node.node_id] = idx;
  });

  const nNodes = sortedNodes.length;
  const adjacencyMatrix: number[][] = Array(nNodes)
    .fill(null)
    .map(() => Array(nNodes).fill(0));

  edges.forEach((edge) => {
    const srcIdx = nodeIdToIdx[edge.source];
    const dstIdx = nodeIdToIdx[edge.target];
    if (srcIdx !== undefined && dstIdx !== undefined) {
      adjacencyMatrix[dstIdx][srcIdx] = edge.weight;
    }
  });

  return { matrix: adjacencyMatrix, sortedNodes };
}

export function computeGraphScoresFromGraphData(
  graphData: WorkerGraph,
  pinnedIds: string[] = [],
): WorkerScoreResult {
  const hasLorsa = graphData.nodes.some((n) => n.feature_type === 'lorsa' || n.feature_type === 'lorsa error');
  if (hasLorsa) {
    console.warn(
      'Score computation skipped: graph contains lorsa nodes which are not yet supported by the scoring algorithm.',
    );
    return {
      replacementScore: -1,
      completenessScore: -1,
      errorNodeInfluences: [],
      suggestedPinIds: [],
      errorEdgeAnalysis: { leakingPins: [], unexplainedPins: [], bridgeCandidates: [] },
    };
  }

  const graphNodesToUse = graphData.nodes;

  const { matrix: adjacencyMatrix, sortedNodes } = reconstructAdjacencyMatrix(graphNodesToUse, graphData.links);

  // Helper to zero a row and its corresponding column by index
  const zeroRowAndColumn = (idx: number) => {
    for (let c = 0; c < adjacencyMatrix.length; c += 1) {
      adjacencyMatrix[idx][c] = 0;
    }
    for (let r = 0; r < adjacencyMatrix.length; r += 1) {
      adjacencyMatrix[r][idx] = 0;
    }
  };

  // If pinnedIds provided, merge non-pinned feature nodes into their corresponding error nodes
  if (pinnedIds.length > 0) {
    const pinnedSet = new Set(pinnedIds);

    // Build a lookup from (layer, ctx_idx) to error node index in sortedNodes
    const errorIndexByKey: Record<string, number> = {};
    sortedNodes.forEach((node, idx) => {
      if (node.feature_type === 'mlp reconstruction error') {
        const key = `${node.layer}|${node.ctx_idx}`;
        errorIndexByKey[key] = idx;
      }
    });

    sortedNodes.forEach((node, featureIdx) => {
      if (node.feature_type !== 'cross layer transcoder') return;
      if (pinnedSet.has(node.node_id)) return; // keep pinned features

      const key = `${node.layer}|${node.ctx_idx}`;
      const errorIdx = errorIndexByKey[key];
      if (errorIdx === undefined) {
        // No matching error node found; just zero out the feature node
        zeroRowAndColumn(featureIdx);
        return;
      }

      // Merge outgoing edges (column) of feature into error node's column
      for (let r = 0; r < adjacencyMatrix.length; r += 1) {
        adjacencyMatrix[r][errorIdx] += adjacencyMatrix[r][featureIdx];
      }

      // Zero both incoming and outgoing edges for the feature node
      zeroRowAndColumn(featureIdx);
    });
  }

  // Derive counts from sortedNodes to align with adjacency order
  const nFeatures = sortedNodes.filter((n) => n.feature_type === 'cross layer transcoder').length;
  const nErrorNodes = sortedNodes.filter((n) => n.feature_type === 'mlp reconstruction error').length;
  const nTokens = sortedNodes.filter((n) => n.feature_type === 'embedding').length;
  const nLogits = sortedNodes.filter((n) => n.feature_type === 'logit').length;

  const errorStart = nFeatures;
  const errorEnd = errorStart + nErrorNodes;
  const tokenEnd = errorEnd + nTokens;

  // Align logit weights with sortedNodes order so that the last indices match the correct logits
  const logitWeights = new Array(adjacencyMatrix.length).fill(0);
  const sortedLogitProbs = sortedNodes.filter((n) => n.feature_type === 'logit').map((n) => n.token_prob);
  for (let i = 0; i < nLogits; i += 1) {
    logitWeights[adjacencyMatrix.length - nLogits + i] = sortedLogitProbs[i];
  }

  const normalizedMatrix = normalizeMatrix(adjacencyMatrix);
  const nodeInfluence = computeInfluence(normalizedMatrix, logitWeights);

  const tokenInfluence = nodeInfluence.slice(errorEnd, tokenEnd).reduce((sum, val) => sum + val, 0);
  const errorInfluence = nodeInfluence.slice(errorStart, errorEnd).reduce((sum, val) => sum + val, 0);
  const replacementScore = tokenInfluence / (tokenInfluence + errorInfluence);

  const nonErrorFractions = normalizedMatrix.map(
    (row) => 1 - row.slice(errorStart, errorEnd).reduce((sum, val) => sum + val, 0),
  );
  const outputInfluence = nodeInfluence.map((val, i) => val + logitWeights[i]);
  const completenessScore =
    nonErrorFractions.map((fraction, i) => fraction * outputInfluence[i]).reduce((sum, val) => sum + val, 0) /
    outputInfluence.reduce((sum, val) => sum + val, 0);

  // Per-error-node influence breakdown
  const errorNodeInfluences: ErrorNodeInfluence[] = [];
  for (let i = errorStart; i < errorEnd; i += 1) {
    const node = sortedNodes[i];
    errorNodeInfluences.push({
      nodeId: node.node_id,
      influence: nodeInfluence[i],
      layer: node.layer,
      ctxIdx: node.ctx_idx,
    });
  }
  errorNodeInfluences.sort((a, b) => b.influence - a.influence);

  // Suggested pins: unpinned feature nodes near highest-influence error nodes
  const suggestedPinIds: string[] = [];
  if (pinnedIds.length > 0) {
    const pinnedSet = new Set(pinnedIds);
    const topErrorIds = new Set(errorNodeInfluences.slice(0, 10).map((e) => e.nodeId));
    const topErrorKeys = new Set(
      errorNodeInfluences.slice(0, 10).map((e) => `${e.layer}|${e.ctxIdx}`),
    );

    // Build edge index for fast lookup
    const edgesFromNode: Record<string, Array<{ target: string; weight: number }>> = {};
    const edgesToNode: Record<string, Array<{ source: string; weight: number }>> = {};
    graphData.links.forEach((link) => {
      const srcId = typeof link.source === 'string' ? link.source : (link.source as any).node_id;
      const tgtId = typeof link.target === 'string' ? link.target : (link.target as any).node_id;
      if (!edgesFromNode[srcId]) edgesFromNode[srcId] = [];
      edgesFromNode[srcId].push({ target: tgtId, weight: Math.abs(link.weight) });
      if (!edgesToNode[tgtId]) edgesToNode[tgtId] = [];
      edgesToNode[tgtId].push({ source: srcId, weight: Math.abs(link.weight) });
    });

    // Score unpinned features by: layer/ctxIdx match with error nodes + edge connectivity
    const candidateScores: Array<{ nodeId: string; score: number }> = [];
    graphData.nodes.forEach((node) => {
      if (node.feature_type !== 'cross layer transcoder') return;
      if (pinnedSet.has(node.node_id)) return;

      let score = 0;
      const key = `${node.layer}|${node.ctx_idx}`;

      // Bonus for sharing layer/ctxIdx with a top error node (merged features)
      if (topErrorKeys.has(key)) {
        const outEdges = edgesFromNode[node.node_id] || [];
        score += outEdges.reduce((s, e) => s + e.weight, 0) * 2;
      }

      // Bonus for having edges to/from top error nodes
      const outEdges = edgesFromNode[node.node_id] || [];
      outEdges.forEach((e) => {
        if (topErrorIds.has(e.target)) score += e.weight;
      });
      const inEdges = edgesToNode[node.node_id] || [];
      inEdges.forEach((e) => {
        if (topErrorIds.has(e.source)) score += e.weight;
      });

      // Also consider node influence as a tiebreaker
      score += (node.influence ?? 0) * 0.1;

      if (score > 0) {
        candidateScores.push({ nodeId: node.node_id, score });
      }
    });

    candidateScores.sort((a, b) => b.score - a.score);
    candidateScores.slice(0, 50).forEach((c) => {
      suggestedPinIds.push(c.nodeId);
    });
  }

  // Error edge analysis: find leaking pins, unexplained pins, and bridge candidates
  const errorEdgeAnalysis: ErrorEdgeAnalysis = {
    leakingPins: [],
    unexplainedPins: [],
    bridgeCandidates: [],
  };

  if (pinnedIds.length > 0) {
    const pinnedSet = new Set(pinnedIds);

    // Build edge index from original graph data
    const edgesFrom: Record<string, Array<{ target: string; weight: number }>> = {};
    const edgesTo: Record<string, Array<{ source: string; weight: number }>> = {};
    graphData.links.forEach((link) => {
      const srcId = typeof link.source === 'string' ? link.source : (link.source as any).node_id;
      const tgtId = typeof link.target === 'string' ? link.target : (link.target as any).node_id;
      if (!edgesFrom[srcId]) edgesFrom[srcId] = [];
      edgesFrom[srcId].push({ target: tgtId, weight: Math.abs(link.weight) });
      if (!edgesTo[tgtId]) edgesTo[tgtId] = [];
      edgesTo[tgtId].push({ source: srcId, weight: Math.abs(link.weight) });
    });

    const errorNodeIdSet = new Set(graphData.nodes.filter((n) => n.feature_type === 'mlp reconstruction error').map((n) => n.node_id));

    // 1. Leaking pins: pinned features with strong outgoing edges to error nodes
    pinnedIds.forEach((pinnedId) => {
      (edgesFrom[pinnedId] || []).forEach((edge) => {
        if (errorNodeIdSet.has(edge.target)) {
          errorEdgeAnalysis.leakingPins.push({
            pinnedNodeId: pinnedId,
            errorNodeId: edge.target,
            weight: edge.weight,
          });
        }
      });
    });
    errorEdgeAnalysis.leakingPins.sort((a, b) => b.weight - a.weight);
    errorEdgeAnalysis.leakingPins = errorEdgeAnalysis.leakingPins.slice(0, 10);

    // 2. Unexplained pins: pinned features receiving strong incoming edges from error nodes
    pinnedIds.forEach((pinnedId) => {
      (edgesTo[pinnedId] || []).forEach((edge) => {
        if (errorNodeIdSet.has(edge.source)) {
          errorEdgeAnalysis.unexplainedPins.push({
            pinnedNodeId: pinnedId,
            errorNodeId: edge.source,
            weight: edge.weight,
          });
        }
      });
    });
    errorEdgeAnalysis.unexplainedPins.sort((a, b) => b.weight - a.weight);
    errorEdgeAnalysis.unexplainedPins = errorEdgeAnalysis.unexplainedPins.slice(0, 10);

    // 3. Bridge candidates: unpinned features that connect pinned nodes through error nodes
    // For each error node involved in leaking/unexplained edges, find unpinned features
    // at the same layer/ctxIdx that could "replace" the error node's role
    const involvedErrorIds = new Set([
      ...errorEdgeAnalysis.leakingPins.map((e) => e.errorNodeId),
      ...errorEdgeAnalysis.unexplainedPins.map((e) => e.errorNodeId),
    ]);

    const bridgeScores: Record<string, number> = {};
    graphData.nodes.forEach((node) => {
      if (node.feature_type !== 'cross layer transcoder') return;
      if (pinnedSet.has(node.node_id)) return;

      let score = 0;

      // Check if this feature has edges to/from pinned nodes that pass near involved error nodes
      const nodeKey = `${node.layer}|${node.ctx_idx}`;
      const outEdges = edgesFrom[node.node_id] || [];
      const inEdges = edgesTo[node.node_id] || [];

      // Score by connections to pinned nodes (would reduce error dependency)
      outEdges.forEach((e) => {
        if (pinnedSet.has(e.target)) score += e.weight * 1.5;
      });
      inEdges.forEach((e) => {
        if (pinnedSet.has(e.source)) score += e.weight * 1.5;
      });

      // Bonus for sharing layer/ctxIdx with involved error nodes
      involvedErrorIds.forEach((errId) => {
        const errNode = graphData.nodes.find((n) => n.node_id === errId);
        if (errNode && `${errNode.layer}|${errNode.ctx_idx}` === nodeKey) {
          score += 2.0;
        }
      });

      // Bonus for edges to/from involved error nodes
      outEdges.forEach((e) => {
        if (involvedErrorIds.has(e.target)) score += e.weight;
      });
      inEdges.forEach((e) => {
        if (involvedErrorIds.has(e.source)) score += e.weight;
      });

      if (score > 0) {
        bridgeScores[node.node_id] = (bridgeScores[node.node_id] || 0) + score;
      }
    });

    errorEdgeAnalysis.bridgeCandidates = Object.entries(bridgeScores)
      .map(([nodeId, score]) => ({ nodeId, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 50);
  }

  return {
    replacementScore: Number.isNaN(replacementScore) ? 0 : replacementScore,
    completenessScore: Number.isNaN(completenessScore) ? 0 : completenessScore,
    errorNodeInfluences: errorNodeInfluences.slice(0, 10),
    suggestedPinIds,
    errorEdgeAnalysis,
  };
}

// Guard worker setup — only runs in Web Worker context, not Node.js
if (typeof self !== 'undefined' && typeof self.onmessage !== 'undefined') {
  console.log('Worker script loaded');
}

if (typeof self !== 'undefined') {
// @ts-ignore — self is only available in Web Worker context
self.onmessage = (ev: MessageEvent) => {
  console.log('Worker received message:', ev.data ? 'has data' : 'null data');

  try {
    // Handle Chrome's stricter Web Worker message handling
    if (!ev.data) {
      // Chrome may send null data during worker initialization - ignore these messages
      return;
    }

    const { data } = ev;

    // Validate that we have the expected message structure

    if (typeof data !== 'object' || !data.hasOwnProperty('graph') || !data.hasOwnProperty('requestId')) {
      console.error('Worker: Invalid message format', data);
      // @ts-ignore
      self.postMessage({ error: 'Invalid message format received by worker' });
      return;
    }

    console.log('Worker: starting computation');
    const { requestId, graph, pinnedIds } = data as { requestId: number; graph: WorkerGraph; pinnedIds: string[] };
    const scores = computeGraphScoresFromGraphData(graph, pinnedIds);
    console.log('Worker: computation complete, sending results:', scores);
    // @ts-ignore
    self.postMessage({ requestId, ...scores });
  } catch (err) {
    console.error('Worker error:', err);
    // @ts-ignore
    self.postMessage({ error: (err as Error)?.message || 'Unknown worker error' });
  }
};
}
