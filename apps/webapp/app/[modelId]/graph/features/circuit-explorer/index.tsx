'use client';

import CustomTooltip from '@/components/custom-tooltip';
import { useGlobalContext } from '@/components/provider/global-provider';
import { useGraphContext } from '@/components/provider/graph-provider';
import { useGraphStateContext } from '@/components/provider/graph-state-provider';
import { Button } from '@/components/shadcn/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/shadcn/dialog';
import { LoadingSquare } from '@/components/svg/loading-square';
import { Check, ExternalLink, Loader2, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { CircuitCausalityResult, validatePinnedCircuit } from '../../causal-validation';
import CircuitCausalityResultView from '../../circuit-causality-result';
import { CLTGraphNode } from '../../graph-types';
import { useCircuitExplorerContext } from './context';
import { computeCircuitDiff } from './diff';
import { buildGroupingGraphData } from './graph-payload';
import { ExploredCircuit, GroupingModel, GroupingResponsePayload } from './types';
import { useExploreCircuits } from './use-explore-circuits';

/**
 * Dialog for triggering exploration. Shows loading state.
 * Once results arrive, closes itself and stores circuits in modal context.
 */
export default function ExploreCircuitsModal() {
  const { isExploreCircuitsModalOpen, setIsExploreCircuitsModalOpen, exploredCircuits } = useCircuitExplorerContext();
  const { selectedGraph } = useGraphContext();

  // If circuits already exist, just close the modal to reveal the results panel
  useEffect(() => {
    if (isExploreCircuitsModalOpen && exploredCircuits.length > 0) {
      setIsExploreCircuitsModalOpen(false);
    }
  }, [isExploreCircuitsModalOpen, exploredCircuits.length, setIsExploreCircuitsModalOpen]);

  const [selectedLogitId, setSelectedLogitId] = useState<string | null>(null);
  const [numSeeds, setNumSeeds] = useState(5);
  const [maxSteps, setMaxSteps] = useState(40);
  const [pcPasses, setPcPasses] = useState(1);
  const [keepRatio, setKeepRatio] = useState(0.4);
  const [groupingModel, setGroupingModel] = useState<'sonnet' | 'haiku'>('sonnet');

  // Get all logit nodes sorted by probability
  const logitNodes = (selectedGraph?.nodes || [])
    .filter((n) => n.feature_type === 'logit')
    .sort((a, b) => (b.token_prob || 0) - (a.token_prob || 0));

  // Default to target logit or highest prob
  const activeLogitId =
    selectedLogitId || logitNodes.find((n) => n.is_target_logit)?.node_id || logitNodes[0]?.node_id || null;
  const { error, handleExplore, isExploring } = useExploreCircuits({
    activeLogitId,
    numSeeds,
    maxSteps,
    pcPasses,
    keepRatio,
    groupingModel,
  });

  return (
    <Dialog open={isExploreCircuitsModalOpen} onOpenChange={setIsExploreCircuitsModalOpen}>
      <DialogContent className="max-w-md bg-white">
        <DialogHeader>
          <DialogTitle>Circuit Explorer</DialogTitle>
        </DialogHeader>

        <div className="flex flex-col gap-y-3">
          {!isExploring && (
            <div className="flex flex-col items-center gap-y-3 py-4">
              <div className="flex flex-col gap-y-2 rounded-md bg-slate-50 px-3 py-2.5 text-xs leading-relaxed text-slate-500">
                <p>
                  Automatically discovers circuits using{' '}
                  <strong className="text-slate-700">Influence-Aware search + Pathway Completion (IA+PC)</strong> - a
                  two-phase strategy that finds potentially causally important circuits without model inference during
                  exploration, with option to verify causality via circuit ablation.
                </p>
                <div className="flex flex-col gap-y-1.5">
                  <div className="flex items-start gap-x-2">
                    <span className="mt-0.5 text-sky-500">&#9672;</span>
                    <span>
                      <strong className="text-slate-700">Influence-Aware search</strong> iteratively builds a circuit by
                      adding one feature at a time. At each step, it picks the feature that most improves the{' '}
                      <em>Completeness score</em> (how well the circuit&apos;s features explain each other&apos;s
                      activations) while favoring features that positively influence the target logit. This finds
                      features with fully interpretable causal chains - not just high-influence leaves.
                    </span>
                  </div>
                  <div className="flex items-start gap-x-2">
                    <span className="mt-0.5 text-sky-500">&#9672;</span>
                    <span>
                      <strong className="text-slate-700">Pathway Completion</strong> runs after the search converges. It
                      looks at features adjacent to the circuit (downstream and upstream neighbors) and adds those that
                      don&apos;t degrade the Completeness score. This captures computation the greedy search missed -
                      features that the circuit&apos;s features feed into - closing the gap between necessity and
                      sufficiency.
                    </span>
                  </div>
                  <div className="flex items-start gap-x-2">
                    <span className="mt-0.5 text-violet-400">&#9672;</span>
                    <span>
                      Test each circuit&apos;s causal impact via{' '}
                      <strong className="text-violet-600">&apos;Test Causality&apos;</strong> in the Explored Circuits
                      panel.
                    </span>
                  </div>
                </div>
                <div className="flex flex-col gap-y-2 border-t border-slate-200 pt-2">
                  <div className="flex items-center justify-between">
                    <div className="flex flex-col gap-y-0.5">
                      <div className="flex items-center gap-x-1">
                        <span className="text-[11px] font-semibold text-slate-600">Starting features</span>
                        <CustomTooltip
                          trigger={
                            <div className="flex h-3.5 w-3.5 cursor-pointer items-center justify-center rounded-full bg-slate-200 text-[8px] font-bold text-slate-500">
                              ?
                            </div>
                          }
                          side="right"
                        >
                          <div className="text-xs text-slate-700">
                            Number of features to start circuit exploration from. Each starting feature is selected from
                            those with the strongest direct connections to the target logit. This also roughly sets how
                            many candidate circuits are produced: one seed generally yields one circuit. More starting
                            features = more circuits and longer exploration time (~30s each).
                          </div>
                        </CustomTooltip>
                      </div>
                      <span className="text-[10px] font-medium text-sky-600">
                        Also controls how many circuits get produced.
                      </span>
                    </div>
                    <select
                      value={numSeeds}
                      onChange={(e) => setNumSeeds(Number(e.target.value))}
                      className="rounded border border-slate-200 py-0.5 pl-1.5 pr-5 text-[11px] text-slate-700"
                    >
                      <option value={1}>1</option>
                      <option value={3}>3</option>
                      <option value={5}>5</option>
                      <option value={10}>10</option>
                      <option value={20}>20</option>
                    </select>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-x-1">
                      <span className="text-[11px] font-semibold text-slate-600">IA Steps</span>
                      <CustomTooltip
                        trigger={
                          <div className="flex h-3.5 w-3.5 cursor-pointer items-center justify-center rounded-full bg-slate-200 text-[8px] font-bold text-slate-500">
                            ?
                          </div>
                        }
                        side="right"
                      >
                        <div className="text-xs text-slate-700">
                          Max greedy search steps per seed. More steps = more features per circuit but diminishing
                          returns after ~80. Each step adds one feature that improves the C score.
                        </div>
                      </CustomTooltip>
                    </div>
                    <select
                      value={maxSteps}
                      onChange={(e) => setMaxSteps(Number(e.target.value))}
                      className="rounded border border-slate-200 py-0.5 pl-1.5 pr-5 text-[11px] text-slate-700"
                    >
                      <option value={40}>40 (fast)</option>
                      <option value={80}>80</option>
                      <option value={120}>120</option>
                      <option value={200}>200 (thorough)</option>
                    </select>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-x-1">
                      <span className="text-[11px] font-semibold text-slate-600">PC Passes</span>
                      <CustomTooltip
                        trigger={
                          <div className="flex h-3.5 w-3.5 cursor-pointer items-center justify-center rounded-full bg-slate-200 text-[8px] font-bold text-slate-500">
                            ?
                          </div>
                        }
                        side="right"
                      >
                        <div className="text-xs text-slate-700">
                          Pathway Completion passes after the greedy search. Each pass adds downstream features that
                          don&apos;t degrade the C score, improving sufficiency. More passes = higher sufficiency but
                          more features.
                        </div>
                      </CustomTooltip>
                    </div>
                    <select
                      value={pcPasses}
                      onChange={(e) => setPcPasses(Number(e.target.value))}
                      className="rounded border border-slate-200 py-0.5 pl-1.5 pr-5 text-[11px] text-slate-700"
                    >
                      <option value={0}>0 (none)</option>
                      <option value={1}>1 (default)</option>
                      <option value={2}>2</option>
                      <option value={3}>3 (thorough)</option>
                    </select>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-x-1">
                      <span className="text-[11px] font-semibold text-slate-600">Graph Pruning</span>
                      <CustomTooltip
                        trigger={
                          <div className="flex h-3.5 w-3.5 cursor-pointer items-center justify-center rounded-full bg-slate-200 text-[8px] font-bold text-slate-500">
                            ?
                          </div>
                        }
                        side="right"
                      >
                        <div className="text-xs text-slate-700">
                          Fraction of features to keep before search. Features are ranked by target influence + edge
                          connectivity. Lower = faster search and often better circuits (fewer noise features). All
                          non-feature nodes (embeddings, error nodes, logits) are always kept.
                        </div>
                      </CustomTooltip>
                    </div>
                    <select
                      value={keepRatio}
                      onChange={(e) => setKeepRatio(Number(e.target.value))}
                      className="rounded border border-slate-200 py-0.5 pl-1.5 pr-5 text-[11px] text-slate-700"
                    >
                      <option value={0.2}>20% (aggressive)</option>
                      <option value={0.4}>40% (default)</option>
                      <option value={0.6}>60%</option>
                      <option value={0.8}>80%</option>
                      <option value={1.0}>100% (no pruning)</option>
                    </select>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-x-1">
                      <span className="text-[11px] font-semibold text-slate-600">LL Grouping</span>
                      <CustomTooltip
                        trigger={
                          <div className="flex h-3.5 w-3.5 cursor-pointer items-center justify-center rounded-full bg-slate-200 text-[8px] font-bold text-slate-500">
                            ?
                          </div>
                        }
                        side="right"
                      >
                        <div className="text-xs text-slate-700">
                          Choose which Anthropic model groups discovered circuit features into supernodes. Sonnet is
                          slower but usually stronger; Haiku is faster and cheaper.
                        </div>
                      </CustomTooltip>
                    </div>
                    <select
                      value={groupingModel}
                      onChange={(e) => setGroupingModel(e.target.value as 'sonnet' | 'haiku')}
                      className="rounded border border-slate-200 py-0.5 pl-1.5 pr-5 text-[11px] text-slate-700"
                    >
                      <option value="sonnet">Sonnet</option>
                      <option value="haiku">Haiku</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Logit selector */}
              {logitNodes.length > 0 && (
                <div className="flex w-full flex-col gap-y-1">
                  <span className="text-xs font-medium text-slate-500">Target logit:</span>
                  <div className="flex max-h-32 flex-col gap-y-0.5 overflow-y-auto">
                    {logitNodes.map((n) => (
                      <button
                        key={n.node_id}
                        type="button"
                        onClick={() => setSelectedLogitId(n.node_id)}
                        className={`flex items-center justify-between rounded px-2 py-1.5 text-left text-xs transition-colors ${
                          activeLogitId === n.node_id
                            ? 'bg-sky-100 font-semibold text-sky-700'
                            : 'bg-slate-50 text-slate-600 hover:bg-slate-100'
                        }`}
                      >
                        <span>{n.clerp || n.node_id}</span>
                        {n.is_target_logit && (
                          <span className="ml-1 rounded bg-slate-200 px-1 py-0.5 text-[10px] text-slate-500">
                            default
                          </span>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <Button onClick={handleExplore} disabled={!selectedGraph}>
                Start Exploration
              </Button>
            </div>
          )}

          {isExploring && (
            <div className="flex flex-col items-center gap-y-3 py-6">
              <LoadingSquare className="h-6 w-6" />
              <p className="text-sm text-slate-500">Starting exploration...</p>
            </div>
          )}

          {error && <div className="rounded-md bg-red-50 p-3 text-sm text-red-700">{error}</div>}
        </div>
      </DialogContent>
    </Dialog>
  );
}

/**
 * Inline panel showing ranked circuits. Renders in the layout next to the subgraph.
 */
export function ExploreResultsPanel({ embedded = false, onClose }: { embedded?: boolean; onClose?: () => void }) {
  const { exploredCircuits, exploreProgress, dispatch } = useCircuitExplorerContext();
  const { setVisState, selectedGraph, selectedModelId, selectedSourceSetName, getOverrideClerpForNode } =
    useGraphContext();
  const { clearClickedState, clearHoverState } = useGraphStateContext();
  const { setFeatureModalFeature, setFeatureModalOpen, getSource } = useGlobalContext();
  const [appliedRank, setAppliedRank] = useState<number | null>(null);
  const [reviewedRanks, setReviewedRanks] = useState<Set<number>>(new Set());
  const [reviewingCircuit, setReviewingCircuit] = useState<ExploredCircuit | null>(null);
  const [acceptedSupernodes, setAcceptedSupernodes] = useState<Set<number>>(new Set());
  const [validationResults, setValidationResults] = useState<Record<number, CircuitCausalityResult>>({});
  const [validatingRank, setValidatingRank] = useState<number | null>(null);
  const [diffSelection, setDiffSelection] = useState<[number | null, number | null]>([null, null]);
  const [retryingRanks, setRetryingRanks] = useState<Set<number>>(new Set());
  const [groupingNow, setGroupingNow] = useState(() => Date.now());

  useEffect(() => {
    const hasPendingGrouping = exploredCircuits.some((circuit) => !circuit.ready && circuit.groupingStartedAt);
    if (!hasPendingGrouping) return undefined;
    const interval = window.setInterval(() => setGroupingNow(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, [exploredCircuits]);

  if (exploredCircuits.length === 0 && !exploreProgress) return null;

  const currentSeed = exploreProgress?.currentSeed ?? (exploreProgress?.completedSeeds || 0) + 1;
  const buildStep = exploreProgress?.buildStep;
  const buildMaxSteps = exploreProgress?.buildMaxSteps || 80;
  const buildFeatures = exploreProgress?.buildFeatures ?? 0;
  const buildCScore = exploreProgress?.buildCScore ?? 0;
  const buildProgressPercent = buildStep != null ? (buildStep / buildMaxSteps) * 100 : 5;

  function applyCircuit(circuit: ExploredCircuit, supernodes: string[][]) {
    clearClickedState();
    clearHoverState();
    setVisState({
      pinnedIds: circuit.pinned_ids,
      clerps: [],
      sufficiencyAddedIds: circuit.sufficiency_added_ids || [],
      subgraph: {
        supernodes,
        sticky: true,
        dagrefy: true,
        activeGrouping: {
          isActive: false,
          selectedNodeIds: new Set<string>(),
        },
      },
    });
    setAppliedRank(circuit.rank);
    setReviewingCircuit(null);
  }

  const startReview = (circuit: ExploredCircuit) => {
    setReviewedRanks((prev) => new Set([...prev, circuit.rank]));
    if (circuit.supernodes?.length > 0) {
      setReviewingCircuit(circuit);
      // Accept all by default
      setAcceptedSupernodes(new Set(circuit.supernodes.map((_, i) => i)));
    } else {
      applyCircuit(circuit, []);
    }
  };

  const toggleSupernode = (index: number) => {
    setAcceptedSupernodes((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  const confirmReview = () => {
    if (!reviewingCircuit) return;
    const accepted = (reviewingCircuit.supernodes || []).filter((_, i) => acceptedSupernodes.has(i));
    applyCircuit(reviewingCircuit, accepted);
  };

  const getNode = (nodeId: string): CLTGraphNode | undefined => selectedGraph?.nodes.find((n) => n.node_id === nodeId);

  const getNodeLabel = (nodeId: string) => {
    const node = getNode(nodeId);
    return node?.clerp || node?.ppClerp || nodeId;
  };

  const openFeatureDetail = (nodeId: string) => {
    const node = getNode(nodeId);
    if (node?.featureDetailNP) {
      setFeatureModalFeature({
        ...node.featureDetailNP,
        source: getSource(node.featureDetailNP.modelId, node.featureDetailNP.layer),
      });
      setFeatureModalOpen(true);
    }
  };

  const validateCircuit = async (circuit: ExploredCircuit) => {
    if (!selectedGraph || validatingRank !== null) return;
    setValidatingRank(circuit.rank);

    try {
      const validation = await validatePinnedCircuit({
        selectedGraph,
        selectedModelId,
        selectedSourceSetName,
        pinnedIds: circuit.pinned_ids || [],
      });
      setValidationResults((prev) => ({
        ...prev,
        [circuit.rank]: validation,
      }));
    } catch (err) {
      console.error('Validation error:', err);
    } finally {
      setValidatingRank(null);
    }
  };

  const retryGrouping = async (circuit: ExploredCircuit) => {
    if (!selectedGraph) return;
    const circuitGroupingModel: GroupingModel = circuit.grouping_model === 'haiku' ? 'haiku' : 'sonnet';

    setRetryingRanks((prev) => new Set(prev).add(circuit.rank));
    dispatch({ type: 'GROUPING_RETRY_STARTED', rank: circuit.rank, eventAt: Date.now() });

    try {
      const response = await fetch('/api/graph/group-nodes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelId: selectedModelId,
          sourceSetName: selectedSourceSetName,
          graphData: buildGroupingGraphData(selectedGraph, getOverrideClerpForNode),
          pinnedIds: circuit.pinned_ids,
          prompt: selectedGraph.metadata.prompt.replaceAll('<bos>', ''),
          groupingModel: circuitGroupingModel,
        }),
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const grouped = (await response.json()) as GroupingResponsePayload;
      dispatch({ type: 'GROUPING_RETRY_FINISHED', rank: circuit.rank, grouped });
    } catch (error) {
      console.error('Retry grouping failed:', error);
    } finally {
      setRetryingRanks((prev) => {
        const next = new Set(prev);
        next.delete(circuit.rank);
        return next;
      });
    }
  };

  // Diff helpers
  const toggleDiffSelection = (rank: number) => {
    setDiffSelection(([a, b]) => {
      if (a === rank) return [b, null]; // deselect A
      if (b === rank) return [a, null]; // deselect B
      if (a === null) return [rank, b];
      if (b === null) return [a, rank];
      return [rank, null]; // replace both, start fresh
    });
  };
  const diff = computeCircuitDiff(exploredCircuits, diffSelection, selectedGraph);

  const panelClassName = embedded
    ? 'flex h-full w-full flex-col overflow-hidden bg-white'
    : 'hidden h-full w-full flex-col overflow-hidden border-l border-slate-200 bg-white sm:flex sm:w-[47%] sm:min-w-[47%] sm:max-w-[47%]';

  // Diff panel view
  if (diff) {
    return (
      <div className={panelClassName}>
        <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2">
          <div className="flex flex-col">
            <span className="text-base font-bold text-slate-700">Circuit Diff</span>
            <span className="text-xs text-slate-400">
              Circuit #{diff.circuitA.rank} vs #{diff.circuitB.rank}
            </span>
          </div>
          <button
            type="button"
            onClick={() => setDiffSelection([null, null])}
            className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
            title="Close diff"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-2">
          {/* Summary */}
          <div className="mb-3 flex items-center gap-x-3 text-xs">
            <span className="rounded bg-emerald-100 px-1.5 py-0.5 text-emerald-700">{diff.shared.length} shared</span>
            <span className="rounded bg-sky-100 px-1.5 py-0.5 text-sky-700">
              {diff.onlyA.length} only in #{diff.circuitA.rank}
            </span>
            <span className="rounded bg-amber-100 px-1.5 py-0.5 text-amber-700">
              {diff.onlyB.length} only in #{diff.circuitB.rank}
            </span>
          </div>

          {/* Supernode diff */}
          {diff.supernodeDiff.length > 0 && (
            <div className="mb-3">
              <div className="mb-1 text-[11px] font-semibold uppercase text-slate-500">Supernode Diff</div>
              {diff.supernodeDiff.map((sn, i) => (
                <div
                  key={i}
                  className={`mb-1 rounded border p-1.5 text-[10px] ${
                    sn.inA && sn.inB
                      ? 'border-emerald-200 bg-emerald-50'
                      : sn.inA
                        ? 'border-sky-200 bg-sky-50'
                        : 'border-amber-200 bg-amber-50'
                  }`}
                >
                  <div className="flex items-center gap-x-1">
                    <span className="font-bold text-slate-700">{sn.label}</span>
                    {sn.inA && sn.inB && <span className="text-emerald-600">shared</span>}
                    {sn.inA && !sn.inB && <span className="text-sky-600">only #{diff.circuitA.rank}</span>}
                    {!sn.inA && sn.inB && <span className="text-amber-600">only #{diff.circuitB.rank}</span>}
                  </div>
                  {sn.inA && sn.inB && (
                    <div className="mt-0.5 text-[10px] text-slate-400">
                      A: {sn.membersA.length} members | B: {sn.membersB.length} members | Overlap:{' '}
                      {sn.membersA.filter((m) => sn.membersB.includes(m)).length}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Shared features */}
          {diff.shared.length > 0 && (
            <div className="mb-3">
              <div className="mb-1 text-[10px] font-semibold uppercase text-emerald-600">
                Shared Features ({diff.shared.length})
              </div>
              {diff.shared.map((id) => {
                const node = getNode(id);
                const explanation = node?.featureDetailNP?.explanations?.[0]?.description;
                return (
                  <div key={id} className="mb-0.5 flex items-center gap-x-1 rounded bg-emerald-50 px-1.5 py-0.5">
                    <span className="flex-shrink-0 truncate text-[10px] font-medium text-emerald-700">
                      {getNodeLabel(id)}
                    </span>
                    {node?.featureDetailNP && (
                      <button
                        type="button"
                        onClick={() => openFeatureDetail(id)}
                        className="flex-shrink-0 rounded p-0.5 text-emerald-400 hover:text-emerald-600"
                      >
                        <ExternalLink className="h-2 w-2" />
                      </button>
                    )}
                    {explanation && (
                      <span className="truncate text-[11px] italic text-emerald-600/70">{explanation}</span>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Only in A */}
          {diff.onlyA.length > 0 && (
            <div className="mb-3">
              <div className="mb-1 text-[11px] font-semibold uppercase text-sky-600">
                Only in Circuit #{diff.circuitA.rank} ({diff.onlyA.length})
              </div>
              {diff.onlyA.map((id) => {
                const node = getNode(id);
                const explanation = node?.featureDetailNP?.explanations?.[0]?.description;
                return (
                  <div key={id} className="mb-0.5 flex items-center gap-x-1 rounded bg-sky-50 px-1.5 py-0.5">
                    <span className="flex-shrink-0 truncate text-[10px] font-medium text-sky-700">
                      {getNodeLabel(id)}
                    </span>
                    {node?.featureDetailNP && (
                      <button
                        type="button"
                        onClick={() => openFeatureDetail(id)}
                        className="flex-shrink-0 rounded p-0.5 text-sky-400 hover:text-sky-600"
                      >
                        <ExternalLink className="h-2 w-2" />
                      </button>
                    )}
                    {explanation && <span className="truncate text-[11px] italic text-sky-600/70">{explanation}</span>}
                  </div>
                );
              })}
            </div>
          )}

          {/* Only in B */}
          {diff.onlyB.length > 0 && (
            <div className="mb-3">
              <div className="mb-1 text-[11px] font-semibold uppercase text-amber-600">
                Only in Circuit #{diff.circuitB.rank} ({diff.onlyB.length})
              </div>
              {diff.onlyB.map((id) => {
                const node = getNode(id);
                const explanation = node?.featureDetailNP?.explanations?.[0]?.description;
                return (
                  <div key={id} className="mb-0.5 flex items-center gap-x-1 rounded bg-amber-50 px-1.5 py-0.5">
                    <span className="flex-shrink-0 truncate text-[10px] font-medium text-amber-700">
                      {getNodeLabel(id)}
                    </span>
                    {node?.featureDetailNP && (
                      <button
                        type="button"
                        onClick={() => openFeatureDetail(id)}
                        className="flex-shrink-0 rounded p-0.5 text-amber-400 hover:text-amber-600"
                      >
                        <ExternalLink className="h-2 w-2" />
                      </button>
                    )}
                    {explanation && (
                      <span className="truncate text-[11px] italic text-amber-600/70">{explanation}</span>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Supernode review view
  if (reviewingCircuit) {
    const supernodes: string[][] = reviewingCircuit.supernodes || [];
    const explanations: string[] = reviewingCircuit.supernode_explanations || [];
    const memberReasons: Record<string, string>[] = reviewingCircuit.supernode_member_reasons || [];

    return (
      <div className={panelClassName}>
        <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2">
          <div className="flex flex-col">
            <span className="text-base font-bold text-slate-700">Review Supernodes</span>
            <span className="text-xs text-slate-400">
              Circuit #{reviewingCircuit.rank} — {supernodes.length} suggested groups
            </span>
          </div>
          <button
            type="button"
            onClick={() => setReviewingCircuit(null)}
            className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
            title="Back to circuit list"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-2 py-1">
          {supernodes.map((sn, i) => {
            const label = sn[0];
            const memberIds = sn.slice(1);
            const explanation = explanations[i] || '';
            const reasons = memberReasons[i] || {};
            const isAccepted = acceptedSupernodes.has(i);

            return (
              <div
                key={i}
                className={`mb-2 rounded-md border p-2 transition-colors ${
                  isAccepted ? 'border-emerald-300 bg-emerald-50' : 'border-slate-200 bg-slate-50 opacity-60'
                }`}
              >
                <div className="flex items-start justify-between gap-x-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-x-2">
                      <span className="text-sm font-bold text-slate-700">{label}</span>
                      <span className="text-xs text-slate-400">{memberIds.length} features</span>
                    </div>
                    {explanation && <p className="mt-0.5 text-xs leading-snug text-slate-500">{explanation}</p>}
                  </div>
                  <button
                    type="button"
                    onClick={() => toggleSupernode(i)}
                    className={`flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full transition-colors ${
                      isAccepted
                        ? 'bg-emerald-500 text-white hover:bg-emerald-600'
                        : 'bg-slate-200 text-slate-400 hover:bg-slate-300'
                    }`}
                    title={isAccepted ? 'Reject this group' : 'Accept this group'}
                  >
                    <Check className="h-3 w-3" />
                  </button>
                </div>
                <div className="mt-1.5 flex flex-col gap-y-1">
                  {memberIds.map((id) => {
                    const nodeLabel = getNodeLabel(id);
                    const reason = reasons[id];
                    const node = getNode(id);
                    const hasDetail = !!node?.featureDetailNP;
                    return (
                      <div key={id} className="flex flex-col rounded bg-white/60 px-1.5 py-0.5">
                        <div className="flex items-start gap-x-1">
                          <span className="text-[11px] font-medium leading-snug text-slate-600">
                            {nodeLabel}
                            {hasDetail && (
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  openFeatureDetail(id);
                                }}
                                className="ml-0.5 inline-flex rounded p-0.5 align-middle text-sky-500 hover:bg-sky-100 hover:text-sky-700"
                                title="View feature activations"
                              >
                                <ExternalLink className="h-2.5 w-2.5" />
                              </button>
                            )}
                          </span>
                        </div>
                        {reason && <span className="text-[10px] italic leading-snug text-slate-400">{reason}</span>}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex items-center justify-between border-t border-slate-200 px-3 py-2">
          <span className="text-xs text-slate-400">
            {acceptedSupernodes.size}/{supernodes.length} groups accepted
          </span>
          <div className="flex gap-x-2">
            <button
              type="button"
              onClick={() => applyCircuit(reviewingCircuit, [])}
              className="rounded bg-slate-100 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-200"
            >
              Apply without groups
            </button>
            <button
              type="button"
              onClick={confirmReview}
              className="rounded bg-sky-600 px-2 py-1 text-xs font-medium text-white hover:bg-sky-700"
            >
              Apply with {acceptedSupernodes.size} groups
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Circuit list view
  return (
    <div className={panelClassName}>
      <div className="flex flex-col border-b border-slate-200">
        <div className="flex items-center justify-between px-3 py-2">
          <div className="flex items-center gap-x-2">
            <span className="text-base font-bold text-slate-700">Explored Circuits</span>
            <span className="rounded bg-slate-100 px-1.5 py-0.5 text-xs font-medium text-slate-500">
              {exploredCircuits.length}
            </span>
            {Object.keys(validationResults).length > 0 && (
              <span className="text-[10px] text-slate-400">
                tested circuits are re-ranked by necessity drop; untested circuits stay in original order
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={() => {
              onClose?.();
            }}
            className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        {exploreProgress && (
          <div className="px-3 pb-2">
            {(exploreProgress.phase === 'exploring' || exploreProgress.phase === 'building') && (
              <div className="flex flex-col gap-y-1">
                <div className="text-xs text-slate-500">
                  {exploreProgress.phase === 'building' ? (
                    <span>
                      Building circuit {currentSeed}/{exploreProgress.totalSeeds}...
                    </span>
                  ) : (
                    <span>
                      Exploring: {exploreProgress.completedSeeds}/{exploreProgress.totalSeeds} seeds (
                      {exploreProgress.circuitsFound} unique,{' '}
                      {exploreProgress.completedSeeds - exploreProgress.circuitsFound} duplicates)
                    </span>
                  )}
                </div>
                <div className="w-full rounded-full bg-slate-200">
                  <div
                    className="h-1 rounded-full bg-sky-500 transition-all"
                    style={{
                      width: `${exploreProgress.totalSeeds > 0 ? (exploreProgress.completedSeeds / exploreProgress.totalSeeds) * 100 : 0}%`,
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-1">
        {(() => {
          const circuitsInOriginalOrder = [...exploredCircuits].sort((a, b) => a.rank - b.rank);
          const validatedIndices = circuitsInOriginalOrder
            .map((circuit, index) => (validationResults[circuit.rank] ? index : -1))
            .filter((index) => index >= 0);
          const validatedCircuits = validatedIndices
            .map((index) => circuitsInOriginalOrder[index])
            .sort(
              (a, b) =>
                (validationResults[b.rank]?.relativeNecessity || 0) -
                (validationResults[a.rank]?.relativeNecessity || 0),
            );

          const displayCircuits = [...circuitsInOriginalOrder];
          validatedIndices.forEach((index, i) => {
            displayCircuits[index] = validatedCircuits[i];
          });

          return displayCircuits;
        })().map((circuit) => {
          const isReady = !!circuit.ready;
          const isReviewed = reviewedRanks.has(circuit.rank);
          const isApplied = appliedRank === circuit.rank;
          const validation = validationResults[circuit.rank];
          const isValidating = validatingRank === circuit.rank;
          const diffSlot = diffSelection[0] === circuit.rank ? 'A' : diffSelection[1] === circuit.rank ? 'B' : null;
          const isActiveGroupingCircuit = !isReady && !!circuit.isActiveGrouping;
          const groupingFailed = !!circuit.grouping_failed;
          const groupingModelLabel = circuit.grouping_model === 'haiku' ? 'Haiku' : 'Sonnet';
          const groupingFeatureCount = circuit.pinned_ids?.length || circuit.node_count || 0;
          const groupingTimeoutMs = Math.min(
            180000,
            Math.max(60000, 60000 + Math.max(0, groupingFeatureCount - 50) * 800),
          );
          const groupingTimeoutSeconds = Math.round(groupingTimeoutMs / 1000);
          const groupingElapsedMs =
            isActiveGroupingCircuit && circuit.groupingStartedAt ? groupingNow - circuit.groupingStartedAt : 0;
          const groupingTimedOut = isActiveGroupingCircuit && groupingElapsedMs >= groupingTimeoutMs;
          const isRetryingGrouping = retryingRanks.has(circuit.rank);
          const sufficiencyAddedCount = circuit.sufficiency_added_ids?.length ?? 0;
          return (
            <div
              key={circuit.rank}
              className={`mb-1 rounded-md border transition-colors ${
                !isReady
                  ? exploreProgress?.phase === 'grouping' && circuit.rank <= (exploreProgress.groupingCircuit ?? 0)
                    ? 'border-amber-200 bg-amber-50/50'
                    : 'border-slate-100 opacity-50'
                  : isApplied
                    ? 'border-sky-300 bg-sky-50'
                    : isReviewed
                      ? 'border-slate-300 bg-slate-50'
                      : 'border-slate-200'
              }`}
            >
              <div className="flex w-full items-center justify-between px-3 py-2 text-left">
                <div className="flex items-center gap-x-3">
                  <span
                    className={`w-6 text-center text-base font-bold ${isReviewed ? 'text-slate-500' : 'text-slate-400'}`}
                  >
                    {isReviewed && !isApplied ? '✓' : ''}
                    {circuit.rank}
                  </span>
                  <div className="flex flex-col">
                    <div className="flex items-center gap-x-2 text-sm">
                      <span className="font-medium text-slate-700">{circuit.node_count} nodes</span>
                      {isReady && circuit.supernodes?.length > 0 && (
                        <span className="text-slate-400">{circuit.supernodes.length} groups</span>
                      )}
                      {isReady && groupingFailed && <span className="text-amber-600">grouping failed</span>}
                      <span className="text-slate-400">C={circuit.completeness_score.toFixed(3)}</span>
                    </div>
                    <span className="text-xs text-slate-400">seed: {circuit.seed}</span>
                  </div>
                </div>
                <div className="flex items-center gap-x-2">
                  {!isReady && (
                    <span className="flex items-center gap-x-1 rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700">
                      {isActiveGroupingCircuit && !groupingTimedOut && <Loader2 className="h-3 w-3 animate-spin" />}
                      {groupingTimedOut
                        ? `${groupingModelLabel} response taking too long`
                        : isActiveGroupingCircuit
                          ? `Grouping features using ${groupingModelLabel}...`
                          : `Queued for ${groupingModelLabel} grouping...`}
                    </span>
                  )}
                  {isApplied && <Check className="h-4 w-4 text-sky-600" />}
                </div>
              </div>
              {!isReady && groupingTimedOut && (
                <div className="flex items-center justify-between gap-x-2 border-t border-amber-200 bg-amber-50 px-3 py-2">
                  <span className="text-xs text-amber-800">
                    {groupingModelLabel} has not returned a grouping response after {groupingTimeoutSeconds} seconds.
                  </span>
                  <button
                    type="button"
                    disabled={isRetryingGrouping}
                    onClick={() => retryGrouping(circuit)}
                    className="rounded bg-amber-200 px-2 py-1 text-xs font-medium text-amber-900 transition-colors hover:bg-amber-300 disabled:opacity-50"
                  >
                    {isRetryingGrouping ? 'Retrying...' : 'Retry'}
                  </button>
                </div>
              )}
              {isReady && groupingFailed && (
                <div className="flex items-center justify-between gap-x-2 border-t border-amber-200 bg-amber-50 px-3 py-2">
                  <span className="text-xs text-amber-800">
                    {circuit.grouping_error || `${groupingModelLabel} grouping failed.`}
                  </span>
                  <button
                    type="button"
                    disabled={isRetryingGrouping}
                    onClick={() => retryGrouping(circuit)}
                    className="rounded bg-amber-200 px-2 py-1 text-xs font-medium text-amber-900 transition-colors hover:bg-amber-300 disabled:opacity-50"
                  >
                    {isRetryingGrouping ? 'Retrying...' : 'Retry Grouping'}
                  </button>
                </div>
              )}
              {/* View Circuit + Compare + Test buttons */}
              {isReady && (
                <div className="flex items-center gap-x-1 border-t border-slate-100 px-3 py-1.5">
                  <button
                    type="button"
                    onClick={() => startReview(circuit)}
                    className={`flex-1 rounded py-1 text-xs font-semibold transition-colors ${
                      isApplied
                        ? 'bg-sky-100 text-sky-700'
                        : 'bg-sky-50 text-sky-600 hover:bg-sky-100 hover:text-sky-700'
                    }`}
                  >
                    {isApplied ? 'Viewing Circuit' : 'View Circuit'}
                  </button>
                  {!validation && (
                    <div className="flex items-center gap-x-1">
                      <CustomTooltip
                        trigger={
                          <div className="flex h-4 w-4 cursor-pointer items-center justify-center rounded-full bg-violet-200 text-[9px] font-bold text-violet-600">
                            ?
                          </div>
                        }
                        side="top"
                      >
                        <div className="text-xs text-slate-700">
                          Tests this circuit via actual model inference: ablates the circuit to measure necessity
                          (prediction drop when removed) and ablates everything else to measure sufficiency (prediction
                          retained when isolated).
                        </div>
                      </CustomTooltip>
                      <button
                        type="button"
                        disabled={isValidating}
                        onClick={(e) => {
                          e.stopPropagation();
                          validateCircuit(circuit);
                        }}
                        className={`rounded px-2 py-1 text-xs font-semibold transition-colors ${
                          isValidating
                            ? 'bg-violet-100 text-violet-500'
                            : 'bg-violet-100 text-violet-700 hover:bg-violet-200 hover:text-violet-800'
                        } disabled:opacity-50`}
                      >
                        {isValidating ? (
                          <span className="flex items-center gap-x-1">
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Testing...
                          </span>
                        ) : (
                          'Test Causality'
                        )}
                      </button>
                    </div>
                  )}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleDiffSelection(circuit.rank);
                    }}
                    className={`rounded px-2 py-1 text-xs font-medium transition-colors ${
                      diffSlot
                        ? diffSlot === 'A'
                          ? 'bg-sky-200 text-sky-800'
                          : 'bg-amber-200 text-amber-800'
                        : 'bg-slate-100 text-slate-500 hover:bg-slate-200'
                    }`}
                    title={diffSlot ? 'Deselect from comparison' : 'Select for comparison'}
                  >
                    {diffSlot || 'Compare'}
                  </button>
                </div>
              )}
              {/* Validation results */}
              {isReady && validation && (
                <div className="border-t border-slate-100 px-3 py-1">
                  <CircuitCausalityResultView validation={validation} />
                  {sufficiencyAddedCount > 0 && (
                    <div className="mt-1 border-t border-slate-100 pt-1">
                      <div className="text-[10px] font-semibold text-emerald-600">
                        +{sufficiencyAddedCount} features added by sufficiency refinement:
                      </div>
                      <div className="mt-0.5 flex flex-wrap gap-1">
                        {circuit.sufficiency_added_ids?.map((id: string) => {
                          const node = selectedGraph?.nodes.find((n: CLTGraphNode) => n.node_id === id);
                          const label = node?.clerp || node?.ppClerp || id;
                          return (
                            <span
                              key={id}
                              className="inline-block rounded border border-emerald-200 bg-emerald-50 px-1 py-0.5 text-[11px] text-emerald-700"
                              title={id}
                            >
                              {label}
                            </span>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
        {/* Show placeholder row for circuit currently being built */}
        {exploreProgress &&
          (exploreProgress.phase === 'building' ||
            exploreProgress.phase === 'starting' ||
            (exploreProgress.phase === 'exploring' &&
              (exploreProgress.completedSeeds || 0) < (exploreProgress.totalSeeds || 0))) && (
            <div className="mb-2 rounded-md border-2 border-sky-300 bg-sky-50 p-3">
              <div className="flex items-center gap-x-3">
                <Loader2 className="h-5 w-5 shrink-0 animate-spin text-sky-500" />
                <div className="flex flex-1 flex-col gap-y-1">
                  <div className="text-base font-semibold text-sky-700">
                    {exploreProgress.phase === 'starting'
                      ? 'Starting exploration...'
                      : buildStep != null
                        ? `Building circuit ${currentSeed}/${exploreProgress.totalSeeds} — step ${buildStep}/${buildMaxSteps}, ${buildFeatures} features, C=${buildCScore}`
                        : `Building circuit ${currentSeed}/${exploreProgress.totalSeeds}...`}
                  </div>
                  <div className="h-2 w-full rounded-full bg-sky-200">
                    <div
                      className="h-2 rounded-full bg-sky-500 transition-all duration-300"
                      style={{
                        width: `${buildProgressPercent}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
      </div>
    </div>
  );
}
