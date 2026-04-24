'use client';

import CustomTooltip from '@/components/custom-tooltip';
import { GRAPH_PREFETCH_ACTIVATIONS_COUNT, useGraphContext } from '@/components/provider/graph-provider';
import { useGraphStateContext } from '@/components/provider/graph-state-provider';
import { NeuronWithPartialRelations } from '@/prisma/generated/zod';
import { ChevronDown, ChevronUp, CircleCheck, CircleAlert, PinIcon, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CLTGraphNode } from './graph-types';
import { getIndexFromCantorValue, getIndexFromFeatureAndGraph, getLayerFromFeatureAndGraph, MODELS_TO_CALCULATE_REPLACEMENT_SCORES } from './utils';

const REPLACEMENT_THRESHOLD = 0.5;
const COMPLETENESS_THRESHOLD = 0.8;

function scoreColor(value: number, threshold: number): string {
  if (value >= threshold) return 'bg-emerald-500';
  if (value >= threshold * 0.6) return 'bg-amber-500';
  return 'bg-red-400';
}

function scoreTextColor(value: number, threshold: number): string {
  if (value >= threshold) return 'text-emerald-700';
  if (value >= threshold * 0.6) return 'text-amber-700';
  return 'text-red-500';
}

function ScoreBar({
  label,
  tooltip,
  graphValue,
  subgraphValue,
  threshold,
  hasPins,
}: {
  label: string;
  tooltip: string;
  graphValue: number;
  subgraphValue: number;
  threshold: number;
  hasPins: boolean;
}) {
  const displayValue = hasPins ? subgraphValue : graphValue;
  const pct = Math.min(displayValue * 100, 100);

  return (
    <div className="flex flex-col gap-y-1">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-x-1">
          <span className="text-[10px] font-semibold text-slate-600">{label}</span>
          <CustomTooltip
            trigger={
              <div className="flex h-3.5 w-3.5 cursor-pointer items-center justify-center rounded-full bg-slate-200 text-[8px] font-bold text-slate-500">
                ?
              </div>
            }
            side="left"
          >
            <div className="text-[11px] text-slate-700">{tooltip}</div>
          </CustomTooltip>
        </div>
        <div className="flex items-center gap-x-1.5">
          {hasPins && (
            <span className="text-[9px] text-slate-400">graph: {graphValue.toFixed(3)}</span>
          )}
          <span className={`text-[11px] font-bold ${scoreTextColor(displayValue, threshold)}`}>
            {displayValue.toFixed(3)}
          </span>
        </div>
      </div>
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-slate-200">
        <div
          className={`h-full rounded-full transition-all duration-500 ease-out ${scoreColor(displayValue, threshold)}`}
          style={{ width: `${pct}%` }}
        />
        {/* Threshold marker */}
        <div
          className="absolute top-0 h-full w-px bg-slate-500"
          style={{ left: `${threshold * 100}%` }}
        />
      </div>
      <div className="flex justify-between text-[8px] text-slate-400">
        <span>0</span>
        <span>1</span>
      </div>
    </div>
  );
}

/** Resolve the best available label for a node, using featureDetailNP explanations if present. */
function getNodeLabel(node: CLTGraphNode, getOverrideClerp: (n: CLTGraphNode) => string | undefined): string {
  const override = getOverrideClerp(node);
  if (override && override.length > 0) return override;
  if (node.clerp && node.clerp.length > 0) return node.clerp;
  // Fallback: show layer/feature as readable identifier
  const idx = node.feature != null ? getIndexFromCantorValue(node.feature) : node.feature;
  return `L${node.layer} F${idx ?? '?'}`;
}

export default function FidelityDashboard() {
  const {
    selectedModelId,
    selectedGraph,
    selectedSourceSetName,
    visState,
    graphScores,
    subgraphScores,
    togglePin,
    getOverrideClerpForNode,
    updateVisStateField,
  } = useGraphContext();
  const { updateClickedState } = useGraphStateContext();
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [circuitInference, setCircuitInference] = useState<{
    targetToken: string;
    baselineProb: number;
    circuitProb: number;
    topLogits: Array<{ token: string; prob: number }>;
  } | null>(null);
  const [isRunningInference, setIsRunningInference] = useState(false);

  // Clear circuit inference when pins change
  useEffect(() => {
    setCircuitInference(null);
  }, [visState.pinnedIds]);

  const runCircuitInference = async () => {
    if (!selectedGraph || visState.pinnedIds.length === 0 || isRunningInference) return;
    setIsRunningInference(true);

    const pinnedSet = new Set(visState.pinnedIds);
    const featureNodes = selectedGraph.nodes.filter((n) => n.feature_type === 'cross layer transcoder');
    const targetLogit = selectedGraph.nodes.find((n) => n.is_target_logit && n.feature_type === 'logit');
    const targetToken = targetLogit?.logitToken || '';
    const numLogitNodes = selectedGraph.nodes.filter((n) => n.feature_type === 'logit').length;

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

    const baseRequest = {
      modelId: selectedGraph.metadata.scan,
      sourceSetName: selectedSourceSetName || null,
      prompt: selectedGraph.metadata.prompt.replaceAll('<bos>', ''),
      nTokens: 1,
      topK: Math.max(numLogitNodes, 10),
      freezeAttention: false,
      temperature: 0.8,
      freqPenalty: 0,
      seed: null,
      steeredOutputOnly: false,
    };

    try {
      // Ablate complement — keep only circuit features
      const complementIds = featureNodes.filter((n) => !pinnedSet.has(n.node_id) && !pinnedSet.has(n.nodeId || '')).map((n) => n.node_id);
      const res = await fetch('/api/steer-logits', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...baseRequest, features: buildFeatures(complementIds) }),
      });
      const text = await res.text();
      if (!res.ok) throw new Error(text);
      const data = JSON.parse(text);

      // Get baseline prob from DEFAULT logits
      const baselineFirst = data.DEFAULT_LOGITS_BY_TOKEN?.find((l: any) => l.top_logits?.length > 0);
      const baselineProb = baselineFirst?.top_logits?.find(
        (l: any) => l.token === targetToken || l.token.trim() === targetToken || l.token === targetToken?.trim(),
      )?.prob || 0;

      // Get circuit prob from STEERED logits
      const steeredFirst = data.STEERED_LOGITS_BY_TOKEN?.find((l: any) => l.top_logits?.length > 0);
      const circuitProb = steeredFirst?.top_logits?.find(
        (l: any) => l.token === targetToken || l.token.trim() === targetToken || l.token === targetToken?.trim(),
      )?.prob || 0;

      const topLogits = (steeredFirst?.top_logits || []).slice(0, 5).map((l: any) => ({ token: l.token, prob: l.prob }));

      setCircuitInference({ targetToken, baselineProb, circuitProb, topLogits });
    } catch (err) {
      console.error('Circuit inference error:', err);
    } finally {
      setIsRunningInference(false);
    }
  };

  // Drag state
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const dragRef = useRef<{ startX: number; startY: number; startPosX: number; startPosY: number } | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    // Only drag from the header area — ignore if target is a button
    if ((e.target as HTMLElement).closest('button')) return;
    e.preventDefault();
    const panel = panelRef.current;
    if (!panel) return;
    const rect = panel.getBoundingClientRect();
    const parentRect = panel.offsetParent?.getBoundingClientRect() || { left: 0, top: 0 };
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startPosX: rect.left - parentRect.left,
      startPosY: rect.top - parentRect.top,
    };
    panel.setPointerCapture(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.startX;
    const dy = e.clientY - dragRef.current.startY;
    setPosition({
      x: dragRef.current.startPosX + dx,
      y: dragRef.current.startPosY + dy,
    });
  }, []);

  const onPointerUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  // Async-fetched labels for suggested pins that lack explanations
  const [fetchedLabels, setFetchedLabels] = useState<Record<string, string>>({});
  const fetchedIdsRef = useRef<Set<string>>(new Set());

  // Dismissed suggestion IDs
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());

  // Resolve suggested pin node objects (all of them, filtered by dismissed)
  const suggestedPinNodes = useMemo(() => {
    if (!selectedGraph || subgraphScores.suggestedPinIds.length === 0) return [];
    return subgraphScores.suggestedPinIds
      .filter((id) => !dismissedIds.has(id))
      .map((nodeId) => selectedGraph.nodes.find((n) => n.node_id === nodeId))
      .filter((n): n is CLTGraphNode => n !== undefined);
  }, [selectedGraph, subgraphScores.suggestedPinIds, dismissedIds]);

  // Fetch explanations for nodes that don't have labels
  useEffect(() => {
    if (!selectedGraph || suggestedPinNodes.length === 0) return;

    const sourceSet =
      selectedGraph.metadata.feature_details?.neuronpedia_source_set ||
      selectedGraph.metadata.info?.neuronpedia_source_set;
    if (!sourceSet) return;

    // Fetch for nodes missing featureDetailNP (no activations/explanations) or missing a label
    const needsFetch = suggestedPinNodes.filter((node) => {
      if (fetchedIdsRef.current.has(node.node_id)) return false;
      if (!node.featureDetailNP) return true;
      const label = getOverrideClerpForNode(node);
      return !label || label.length === 0;
    });

    if (needsFetch.length === 0) return;

    // Mark as in-flight to avoid duplicate fetches
    needsFetch.forEach((n) => fetchedIdsRef.current.add(n.node_id));

    const features = needsFetch.map((node) => ({
      modelId: selectedModelId,
      layer: `${node.layer}-${sourceSet}`,
      index: getIndexFromCantorValue(node.feature),
      maxActsToReturn: GRAPH_PREFETCH_ACTIVATIONS_COUNT,
    }));

    fetch('/api/features', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(features),
    })
      .then((resp) => resp.json())
      .then((results: NeuronWithPartialRelations[]) => {
        const newLabels: Record<string, string> = {};
        needsFetch.forEach((node, i) => {
          const detail = results[i];
          if (detail?.explanations && detail.explanations.length > 0) {
            const desc = detail.explanations[0].description || '';
            if (desc.length > 0) {
              newLabels[node.node_id] = desc;
              // Also populate the node's featureDetailNP for future use
              // eslint-disable-next-line no-param-reassign
              node.featureDetailNP = detail;
            }
          }
        });
        if (Object.keys(newLabels).length > 0) {
          setFetchedLabels((prev) => ({ ...prev, ...newLabels }));
        }
      })
      .catch(() => {
        // Silently fail — labels will show fallback
      });
  }, [suggestedPinNodes, selectedGraph, selectedModelId, getOverrideClerpForNode]);

  // Build final suggested pins with labels and node refs
  const suggestedPins = useMemo(
    () =>
      suggestedPinNodes.map((node) => {
        const fetched = fetchedLabels[node.node_id];
        const label = fetched || getNodeLabel(node, getOverrideClerpForNode);
        return { nodeId: node.node_id, label, node };
      }),
    [suggestedPinNodes, fetchedLabels, getOverrideClerpForNode],
  );

  if (!selectedGraph || !MODELS_TO_CALCULATE_REPLACEMENT_SCORES.has(selectedModelId)) {
    return null;
  }

  const hasPins = visState.pinnedIds.length > 0;
  const activeScores = hasPins ? subgraphScores : graphScores;
  const isVerified =
    hasPins &&
    activeScores.replacementScore >= REPLACEMENT_THRESHOLD &&
    activeScores.completenessScore >= COMPLETENESS_THRESHOLD;

  return (
    <div
      ref={panelRef}
      className="absolute z-20 w-56 rounded-lg border border-slate-200 bg-white/95 shadow-lg backdrop-blur-sm"
      style={
        position
          ? { left: position.x, top: position.y, bottom: 'auto', right: 'auto' }
          : { bottom: 8, left: 8 }
      }
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      {/* Header — draggable */}
      <div
        className="flex w-full cursor-grab items-center justify-between px-3 py-2 active:cursor-grabbing"
        onPointerDown={onPointerDown}
      >
        <button
          type="button"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="flex flex-1 items-center justify-between"
        >
        <div className="flex items-center gap-x-1.5">
          {isVerified ? (
            <CircleCheck className="h-3.5 w-3.5 text-emerald-600" />
          ) : (
            <CircleAlert className="h-3.5 w-3.5 text-slate-400" />
          )}
          <span className="text-[11px] font-bold text-slate-700">Fidelity</span>
        </div>
        <div className="flex items-center gap-x-1.5">
          {hasPins && (
            <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[9px] font-medium text-slate-500">
              {visState.pinnedIds.length} pinned
            </span>
          )}
          {isCollapsed ? (
            <ChevronUp className="h-3 w-3 text-slate-400" />
          ) : (
            <ChevronDown className="h-3 w-3 text-slate-400" />
          )}
        </div>
        </button>
      </div>

      {/* Body */}
      {!isCollapsed && (
        <div className="flex flex-col gap-y-3 border-t border-slate-100 px-3 pb-3 pt-2">
          <ScoreBar
            label="Replacement"
            tooltip="Fraction of end-to-end causal influence from input tokens to output logits that flows through feature nodes rather than error nodes. Higher means your pinned features capture more of the model's reasoning."
            graphValue={graphScores.replacementScore}
            subgraphValue={subgraphScores.replacementScore}
            threshold={REPLACEMENT_THRESHOLD}
            hasPins={hasPins}
          />
          <ScoreBar
            label="Completeness"
            tooltip="Fraction of incoming influence to each node (weighted by output influence) that comes from interpretable features rather than error nodes. Higher means the circuit is more self-contained with fewer unexplained inputs."
            graphValue={graphScores.completenessScore}
            subgraphValue={subgraphScores.completenessScore}
            threshold={COMPLETENESS_THRESHOLD}
            hasPins={hasPins}
          />

          {/* Circuit inference — test what the model predicts using only circuit features */}
          {hasPins && ['gemma-2-2b', 'qwen3-4b'].includes(selectedModelId) && (
            <div className="flex flex-col gap-y-1 border-t border-slate-100 pt-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-x-1">
                  <span className="text-[9px] font-semibold text-slate-500">Circuit Inference</span>
                  <CustomTooltip
                    trigger={
                      <div className="flex h-3.5 w-3.5 cursor-pointer items-center justify-center rounded-full bg-slate-200 text-[8px] font-bold text-slate-500">
                        ?
                      </div>
                    }
                    side="left"
                  >
                    <div className="text-[11px] text-slate-700">
                      Runs the model with only the pinned circuit features active (everything else ablated) to test if the circuit alone produces the expected output.
                    </div>
                  </CustomTooltip>
                </div>
                <button
                  type="button"
                  onClick={runCircuitInference}
                  disabled={isRunningInference}
                  className="rounded bg-slate-100 px-1.5 py-0.5 text-[8px] font-medium text-slate-500 hover:bg-slate-200 disabled:opacity-50"
                >
                  {isRunningInference ? 'Running...' : circuitInference ? 'Re-run' : 'Run'}
                </button>
              </div>
              {circuitInference && (
                <div className="flex flex-col gap-y-0.5">
                  <div className="flex items-center justify-between text-[9px]">
                    <span className="text-slate-500">Target: {circuitInference.targetToken}</span>
                    <span className="font-mono">
                      <span className="text-slate-400">{(circuitInference.baselineProb * 100).toFixed(1)}%</span>
                      <span className="text-slate-400"> → </span>
                      <span className={circuitInference.circuitProb >= circuitInference.baselineProb * 0.5 ? 'font-medium text-emerald-600' : 'text-red-500'}>
                        {(circuitInference.circuitProb * 100).toFixed(1)}%
                      </span>
                    </span>
                  </div>
                  <div className="flex flex-col gap-y-0 text-[8px]">
                    {circuitInference.topLogits.map((l, i) => (
                      <div key={i} className="flex justify-between text-slate-500">
                        <span className="truncate">{l.token}</span>
                        <span className="ml-1 font-mono">{(l.prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Suggested pins */}
          {hasPins && !isVerified && suggestedPins.length > 0 && (
            <div className="flex flex-col gap-y-1">
              <div className="flex items-center gap-x-1">
                <span className="text-[9px] font-semibold text-slate-500">
                  Suggested Pins ({suggestedPins.length})
                </span>
                <CustomTooltip
                  trigger={
                    <div className="flex h-3.5 w-3.5 cursor-pointer items-center justify-center rounded-full bg-slate-200 text-[8px] font-bold text-slate-500">
                      ?
                    </div>
                  }
                  side="left"
                >
                  <div className="text-[11px] text-slate-700">
                    Features near high-influence error nodes that are likely to improve the circuit&apos;s completeness score when pinned. These are the features the scoring algorithm identifies as most likely to fill gaps in the circuit.
                  </div>
                </CustomTooltip>
              </div>
              <div className="flex max-h-32 flex-col gap-y-1 overflow-y-auto">
                {suggestedPins.map(({ nodeId, label, node }) => (
                  <div
                    key={nodeId}
                    className="flex flex-shrink-0 items-center gap-x-0.5 rounded border border-slate-200 transition-colors hover:bg-slate-50"
                  >
                    <button
                      type="button"
                      onClick={() => updateClickedState(node)}
                      className="flex min-w-0 flex-1 items-center px-2 py-1 text-left text-[9px] text-slate-600"
                      title="Highlight in graph"
                    >
                      <span className="truncate">{label}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => togglePin(nodeId)}
                      className="flex-shrink-0 px-1 py-1 text-slate-400 transition-colors hover:text-sky-600"
                      title="Pin this feature"
                    >
                      <PinIcon className="h-3 w-3" />
                    </button>
                    <button
                      type="button"
                      onClick={() => setDismissedIds((prev) => new Set([...prev, nodeId]))}
                      className="flex-shrink-0 px-1 py-1 text-slate-300 transition-colors hover:text-red-400"
                      title="Dismiss suggestion"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {!hasPins && (
            <div className="text-center text-[9px] text-slate-400">
              Pin nodes to see subgraph fidelity scores
            </div>
          )}

          {/* Node legend with error node toggle */}
          <div className="flex items-center justify-between border-t border-slate-100 pt-2">
            <div className="flex items-center gap-x-3">
              <div className="flex items-center gap-x-1">
                <span className="text-[10px]" style={{ color: '#000', WebkitTextStroke: '1px #000' }}>●</span>
                <span className="text-[9px] text-slate-500">Feature</span>
              </div>
              <div className="flex items-center gap-x-1">
                <span className="text-[10px]" style={{ color: '#fb923c', WebkitTextStroke: '1px #ea580c' }}>◆</span>
                <span className="text-[9px] text-slate-500">Error node</span>
              </div>
            </div>
            <button
              type="button"
              className="relative inline-flex h-3.5 w-6 flex-shrink-0 cursor-pointer items-center rounded-full transition-colors"
              style={{ backgroundColor: visState.showErrorNodes ? '#fb923c' : '#d1d5db' }}
              onClick={() => updateVisStateField('showErrorNodes', !visState.showErrorNodes)}
              title={visState.showErrorNodes ? 'Hide error nodes' : 'Show error nodes'}
            >
              <span
                className="inline-block h-2.5 w-2.5 rounded-full bg-white shadow transition-transform"
                style={{ transform: visState.showErrorNodes ? 'translateX(12px)' : 'translateX(2px)' }}
              />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
