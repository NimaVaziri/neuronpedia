import { CLTGraph } from './graph-types';
import { getIndexFromFeatureAndGraph, getLayerFromFeatureAndGraph } from './utils';

export type CircuitCausalityResult = {
  necessity: number;
  baselineProb: number;
  relativeNecessity: number;
  sufficiency: number;
  baselineLogits: Array<{ token: string; prob: number }>;
  necessityLogits: Array<{ token: string; prob: number }>;
  sufficiencyLogits: Array<{ token: string; prob: number }>;
};

export async function validatePinnedCircuit({
  selectedGraph,
  selectedModelId,
  selectedSourceSetName,
  pinnedIds,
}: {
  selectedGraph: CLTGraph;
  selectedModelId: string;
  selectedSourceSetName: string | null | undefined;
  pinnedIds: string[];
}): Promise<CircuitCausalityResult> {
  const pinnedIdSet = new Set(pinnedIds);
  const featureNodes = selectedGraph.nodes.filter((n) => n.feature_type === 'cross layer transcoder');

  const buildFeatures = (nodeIds: string[]) =>
    nodeIds
      .map((nodeId) => {
        const node = selectedGraph.nodes.find((n) => n.node_id === nodeId || n.nodeId === nodeId);
        if (!node || node.feature_type !== 'cross layer transcoder') return null;
        return {
          layer: getLayerFromFeatureAndGraph(selectedModelId, node, selectedGraph),
          index: getIndexFromFeatureAndGraph(selectedModelId, node, selectedGraph),
          token_active_position: node.ctx_idx,
          steer_position: node.ctx_idx,
          steer_generated_tokens: false,
          delta: null,
          ablate: true,
        };
      })
      .filter((f): f is NonNullable<typeof f> => f !== null);

  const prompt = selectedGraph.metadata.prompt.replaceAll('<bos>', '');
  const baseRequest = {
    modelId: selectedGraph.metadata.scan,
    sourceSetName: selectedSourceSetName || null,
    prompt,
    nTokens: 1,
    topK: 50,
    freezeAttention: false,
    temperature: 0.8,
    freqPenalty: 0,
    seed: null,
    steeredOutputOnly: false,
  };

  const targetLogit = selectedGraph.nodes.find((n) => n.is_target_logit && n.feature_type === 'logit');
  const targetToken = targetLogit?.logitToken || '';

  const postSteerLogits = async (features: ReturnType<typeof buildFeatures>) => {
    const response = await fetch('/api/steer-logits', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...baseRequest, features }),
    });
    const text = await response.text();
    if (!response.ok) {
      throw new Error(text || `steer-logits failed with ${response.status}`);
    }
    return JSON.parse(text);
  };

  const baseline = await postSteerLogits([]);
  const necessity = await postSteerLogits(buildFeatures([...pinnedIdSet]));
  const complementIds = featureNodes.filter((n) => !pinnedIdSet.has(n.node_id)).map((n) => n.node_id);
  const sufficiency = await postSteerLogits(buildFeatures(complementIds));

  const getTopLogits = (data: any, useSteered: boolean): Array<{ token: string; prob: number }> => {
    if (!data) return [];
    const logits = useSteered ? data.STEERED_LOGITS_BY_TOKEN : data.DEFAULT_LOGITS_BY_TOKEN;
    const first = logits?.find((l: any) => l.top_logits?.length > 0);
    return (first?.top_logits || []).slice(0, 5).map((l: any) => ({ token: l.token, prob: l.prob }));
  };

  const getProbForToken = (topLogits: Array<{ token: string; prob: number }>) => {
    const match = topLogits.find(
      (l) => l.token === targetToken || l.token.trim() === targetToken || l.token === targetToken?.trim(),
    );
    return match?.prob || 0;
  };

  const baselineLogits = getTopLogits(baseline, false);
  const necessityLogits = getTopLogits(necessity, true);
  const sufficiencyLogits = getTopLogits(sufficiency, true);

  const baselineProb = getProbForToken(baselineLogits);
  const necessityProb = getProbForToken(necessityLogits);
  const sufficiencyProb = getProbForToken(sufficiencyLogits);
  const necessityDrop = baselineProb - necessityProb;

  return {
    necessity: necessityDrop,
    baselineProb,
    relativeNecessity: baselineProb > 0 ? necessityDrop / baselineProb : 0,
    sufficiency: baselineProb > 0 ? sufficiencyProb / baselineProb : 0,
    baselineLogits,
    necessityLogits,
    sufficiencyLogits,
  };
}
