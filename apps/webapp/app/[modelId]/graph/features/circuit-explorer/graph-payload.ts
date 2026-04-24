import { CLTGraph, CLTGraphLink, CLTGraphNode } from '../../graph-types';
import { GraphLinkEndpointRef } from './types';

type GraphNodeLabelResolver = (node: CLTGraphNode) => string | undefined;
type GraphLinkLike = Omit<CLTGraphLink, 'source' | 'target'> & {
  source: GraphLinkEndpointRef;
  target: GraphLinkEndpointRef;
};

function getLinkNodeId(endpoint: GraphLinkEndpointRef): string {
  return typeof endpoint === 'string' ? endpoint : endpoint.node_id;
}

function serializeLinks(links: CLTGraph['links']) {
  return links.map((link) => {
    const linkRef = link as GraphLinkLike;
    return {
      source: getLinkNodeId(linkRef.source),
      target: getLinkNodeId(linkRef.target),
      weight: link.weight,
    };
  });
}

export function buildExploreGraphData(graph: CLTGraph, getNodeLabel: GraphNodeLabelResolver) {
  return {
    nodes: graph.nodes.map((node) => ({
      node_id: node.node_id,
      feature: node.feature,
      layer: node.layer,
      ctx_idx: node.ctx_idx,
      feature_type: node.feature_type,
      token_prob: node.token_prob,
      is_target_logit: node.is_target_logit,
      influence: node.influence,
      clerp: getNodeLabel(node) || node.clerp || '',
    })),
    links: serializeLinks(graph.links),
    metadata: graph.metadata,
  };
}

export function buildGroupingGraphData(graph: CLTGraph, getNodeLabel: GraphNodeLabelResolver) {
  return {
    nodes: graph.nodes.map((node) => ({
      node_id: node.node_id,
      feature_type: node.feature_type,
      clerp: getNodeLabel(node) || node.clerp || '',
      layer: node.layer,
      ctx_idx: node.ctx_idx,
    })),
    links: serializeLinks(graph.links),
  };
}
