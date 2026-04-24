'use client';

import { useGraphContext } from '@/components/provider/graph-provider';
import { useCallback, useState } from 'react';
import { useCircuitExplorerContext } from './context';
import { buildExploreGraphData } from './graph-payload';
import {
  ExploreCircuitEventData,
  ExploreDoneEventData,
  ExploreGroupedEventData,
  ExploreProgressEventData,
  ExploreStatusEventData,
  GroupingModel,
} from './types';

type UseExploreCircuitsParams = {
  activeLogitId: string | null;
  numSeeds: number;
  maxSteps: number;
  pcPasses: number;
  keepRatio: number;
  groupingModel: GroupingModel;
};

export function useExploreCircuits({
  activeLogitId,
  numSeeds,
  maxSteps,
  pcPasses,
  keepRatio,
  groupingModel,
}: UseExploreCircuitsParams) {
  const { dispatch, setIsExploreCircuitsModalOpen } = useCircuitExplorerContext();
  const { selectedGraph, selectedModelId, selectedSourceSetName, getOverrideClerpForNode } = useGraphContext();

  const [isExploring, setIsExploring] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleExplore = useCallback(async () => {
    if (!selectedGraph) return;
    const graphModelId =
      selectedModelId || selectedGraph.metadata.scan || selectedGraph.metadata.neuronpedia_internal_model?.id || '';

    setIsExploring(true);
    setError(null);
    dispatch({ type: 'START_EXPLORATION' });

    const embeddingNodes = selectedGraph.nodes.filter((node) => node.feature_type === 'embedding');
    const endpointIds = [...embeddingNodes.map((node) => node.node_id), ...(activeLogitId ? [activeLogitId] : [])];

    try {
      const response = await fetch('/api/graph/explore-circuits', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: graphModelId,
          modelId: graphModelId,
          sourceSetName: selectedSourceSetName,
          graph_data: buildExploreGraphData(selectedGraph, getOverrideClerpForNode),
          target_logit_node_id: activeLogitId,
          endpoint_node_ids: endpointIds,
          num_seeds: numSeeds,
          max_iterations: maxSteps,
          pc_passes: pcPasses,
          keep_ratio: keepRatio,
          grouping_model: groupingModel,
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        setError(`Exploration failed: ${text}`);
        setIsExploring(false);
        return;
      }

      setIsExploreCircuitsModalOpen(false);

      const reader = response.body?.getReader();
      if (!reader) {
        setError('No response stream');
        setIsExploring(false);
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let pendingEventType = '';
      let pendingData = '';

      const handleSseEvent = (eventType: string, rawData: string) => {
        if (!eventType || !rawData) return;

        try {
          const data = JSON.parse(rawData);
          if (eventType === 'status') {
            dispatch({ type: 'STATUS_EVENT', data: data as ExploreStatusEventData, eventAt: Date.now() });
          } else if (eventType === 'progress') {
            dispatch({ type: 'PROGRESS_EVENT', data: data as ExploreProgressEventData });
          } else if (eventType === 'circuit') {
            dispatch({
              type: 'CIRCUIT_EVENT',
              data: data as ExploreCircuitEventData,
              defaultGroupingModel: groupingModel,
              eventAt: Date.now(),
            });
          } else if (eventType === 'grouped') {
            dispatch({
              type: 'GROUPED_EVENT',
              data: data as ExploreGroupedEventData,
              eventAt: Date.now(),
            });
          } else if (eventType === 'done') {
            dispatch({ type: 'DONE_EVENT', data: data as ExploreDoneEventData });
          }
        } catch {
          // ignore malformed events
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          const normalizedLine = line.replace(/\r$/, '');
          if (normalizedLine === '') {
            handleSseEvent(pendingEventType, pendingData);
            pendingEventType = '';
            pendingData = '';
          } else if (normalizedLine.startsWith('event: ')) {
            pendingEventType = normalizedLine.slice(7).trim();
          } else if (normalizedLine.startsWith('data: ')) {
            pendingData += (pendingData ? '\n' : '') + normalizedLine.slice(6);
          }
        }
      }

      handleSseEvent(pendingEventType, pendingData);
    } catch (err) {
      setError(`Exploration failed: ${(err as Error).message}`);
    } finally {
      setIsExploring(false);
      dispatch({ type: 'CLEAR_PROGRESS' });
    }
  }, [
    activeLogitId,
    dispatch,
    getOverrideClerpForNode,
    groupingModel,
    selectedGraph,
    keepRatio,
    maxSteps,
    numSeeds,
    pcPasses,
    selectedModelId,
    selectedSourceSetName,
    setIsExploreCircuitsModalOpen,
  ]);

  return {
    error,
    handleExplore,
    isExploring,
  };
}
