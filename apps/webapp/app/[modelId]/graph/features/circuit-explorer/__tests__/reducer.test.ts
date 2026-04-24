import { describe, expect, test } from 'vitest';
import { circuitExplorerReducer, initialCircuitExplorerState } from '../reducer';
import { CircuitExplorerState, ExploredCircuit } from '../types';

function exploredCircuit(overrides: Partial<ExploredCircuit>): ExploredCircuit {
  const pinnedIds = overrides.pinned_ids || [];
  const combinedScore =
    overrides.combined_score ?? (overrides.replacement_score || 0) + (overrides.completeness_score || 0);

  return {
    rank: overrides.rank || 1,
    pinned_ids: pinnedIds,
    replacement_score: overrides.replacement_score || 0,
    completeness_score: overrides.completeness_score || 0,
    combined_score: combinedScore,
    node_count: overrides.node_count || pinnedIds.length,
    seed: overrides.seed || '',
    circuit_id: overrides.circuit_id || [...pinnedIds].sort().join(','),
    supernodes: overrides.supernodes || [],
    supernode_explanations: overrides.supernode_explanations || [],
    supernode_member_reasons: overrides.supernode_member_reasons || [],
    grouping_model: overrides.grouping_model || 'sonnet',
    grouping_failed: overrides.grouping_failed || false,
    grouping_error: overrides.grouping_error || null,
    ready: overrides.ready || false,
    isActiveGrouping: overrides.isActiveGrouping || false,
    groupingStartedAt: overrides.groupingStartedAt,
    debugLastEvent: overrides.debugLastEvent,
    debugLastEventAt: overrides.debugLastEventAt,
    sufficiency_added_ids: overrides.sufficiency_added_ids,
    feature_scores: overrides.feature_scores,
  };
}

describe('circuitExplorerReducer', () => {
  test('starts a new exploration with default progress and no stale circuits', () => {
    const previousState: CircuitExplorerState = {
      isExploreCircuitsModalOpen: true,
      exploredCircuits: [exploredCircuit({ pinned_ids: ['old-feature'] })],
      exploreProgress: {
        phase: 'grouping',
        completedSeeds: 2,
        totalSeeds: 3,
        circuitsFound: 1,
        bestScore: 0.8,
      },
    };

    const nextState = circuitExplorerReducer(previousState, { type: 'START_EXPLORATION' });

    expect(nextState.isExploreCircuitsModalOpen).toBe(true);
    expect(nextState.exploredCircuits).toEqual([]);
    expect(nextState.exploreProgress).toEqual({
      phase: 'starting',
      completedSeeds: 0,
      totalSeeds: 0,
      circuitsFound: 0,
      bestScore: 0,
    });
  });

  test('adds discovered circuits with derived ids, scores, and progress', () => {
    const startedState = circuitExplorerReducer(initialCircuitExplorerState, { type: 'START_EXPLORATION' });

    const nextState = circuitExplorerReducer(startedState, {
      type: 'CIRCUIT_EVENT',
      data: {
        pinned_ids: ['feature-b', 'feature-a'],
        replacement_score: 0.4,
        completeness_score: 0.3,
        completed_seeds: 2,
        total_seeds: 5,
        circuits_found: 1,
      },
      defaultGroupingModel: 'haiku',
      eventAt: 100,
    });

    expect(nextState.exploredCircuits).toHaveLength(1);
    expect(nextState.exploredCircuits[0]).toMatchObject({
      rank: 1,
      pinned_ids: ['feature-b', 'feature-a'],
      replacement_score: 0.4,
      completeness_score: 0.3,
      combined_score: 0.7,
      node_count: 2,
      circuit_id: 'feature-a,feature-b',
      grouping_model: 'haiku',
      ready: false,
      isActiveGrouping: false,
      debugLastEvent: 'circuit discovered rank=1',
      debugLastEventAt: 100,
    });
    expect(nextState.exploreProgress).toMatchObject({
      phase: 'exploring',
      completedSeeds: 2,
      totalSeeds: 5,
      circuitsFound: 1,
      bestScore: 0.7,
    });
  });

  test('marks the active grouping circuit and merges grouped results by circuit id', () => {
    const stateWithCircuit: CircuitExplorerState = {
      ...initialCircuitExplorerState,
      exploredCircuits: [
        exploredCircuit({ rank: 1, pinned_ids: ['feature-a', 'feature-b'], circuit_id: 'feature-a,feature-b' }),
      ],
      exploreProgress: null,
    };

    const groupingState = circuitExplorerReducer(stateWithCircuit, {
      type: 'STATUS_EVENT',
      data: {
        phase: 'grouping',
        grouping_circuit: 1,
        total_to_group: 3,
        circuit_id: 'feature-a,feature-b',
      },
      eventAt: 200,
    });

    expect(groupingState.exploreProgress).toMatchObject({
      phase: 'grouping',
      groupingCircuit: 1,
      totalToGroup: 3,
    });
    expect(groupingState.exploredCircuits[0]).toMatchObject({
      isActiveGrouping: true,
      groupingStartedAt: 200,
      debugLastEvent: 'status:grouping rank=1 at=200',
      debugLastEventAt: 200,
    });

    const groupedState = circuitExplorerReducer(groupingState, {
      type: 'GROUPED_EVENT',
      data: {
        circuit_id: 'feature-a,feature-b',
        circuit: {
          pinned_ids: ['feature-a', 'feature-b'],
          supernodes: [['Geography', 'feature-a', 'feature-b']],
          supernode_explanations: ['Location-related features.'],
          supernode_member_reasons: [{ 'feature-a': 'Mentions a city.', 'feature-b': 'Mentions a state.' }],
          grouping_model: 'haiku',
          grouping_failed: false,
          grouping_error: null,
        },
      },
      eventAt: 250,
    });

    expect(groupedState.exploredCircuits[0]).toMatchObject({
      rank: 1,
      circuit_id: 'feature-a,feature-b',
      ready: true,
      isActiveGrouping: false,
      groupingStartedAt: 200,
      grouping_model: 'haiku',
      supernodes: [['Geography', 'feature-a', 'feature-b']],
      debugLastEvent: 'grouped at=250',
      debugLastEventAt: 250,
    });
  });

  test('merges final done-event scores without dropping existing grouped metadata', () => {
    const groupedCircuit = exploredCircuit({
      rank: 1,
      pinned_ids: ['feature-a', 'feature-b'],
      circuit_id: 'feature-a,feature-b',
      supernodes: [['Geography', 'feature-a']],
      supernode_explanations: ['Location-related features.'],
      ready: true,
      groupingStartedAt: 200,
      debugLastEvent: 'grouped at=250',
      debugLastEventAt: 250,
    });

    const doneState = circuitExplorerReducer(
      {
        ...initialCircuitExplorerState,
        exploredCircuits: [groupedCircuit],
        exploreProgress: {
          phase: 'grouping',
          completedSeeds: 3,
          totalSeeds: 3,
          circuitsFound: 1,
          bestScore: 0.6,
        },
      },
      {
        type: 'DONE_EVENT',
        data: {
          circuits: [
            {
              pinned_ids: ['feature-b', 'feature-a'],
              replacement_score: 0.55,
              completeness_score: 0.35,
              node_count: 2,
            },
          ],
        },
      },
    );

    expect(doneState.exploreProgress).toBeNull();
    expect(doneState.exploredCircuits[0]).toMatchObject({
      rank: 1,
      circuit_id: 'feature-a,feature-b',
      replacement_score: 0.55,
      completeness_score: 0.35,
      combined_score: 0.9,
      ready: true,
      isActiveGrouping: false,
      supernodes: [['Geography', 'feature-a']],
      groupingStartedAt: 200,
      debugLastEvent: 'grouped at=250',
      debugLastEventAt: 250,
    });
  });

  test('tracks grouping retry lifecycle for one circuit at a time', () => {
    const retryStartedState = circuitExplorerReducer(
      {
        ...initialCircuitExplorerState,
        exploredCircuits: [
          exploredCircuit({ rank: 1, pinned_ids: ['feature-a'], ready: true }),
          exploredCircuit({ rank: 2, pinned_ids: ['feature-b'], ready: true, isActiveGrouping: true }),
        ],
        exploreProgress: null,
      },
      { type: 'GROUPING_RETRY_STARTED', rank: 1, eventAt: 300 },
    );

    expect(retryStartedState.exploredCircuits[0]).toMatchObject({
      ready: false,
      isActiveGrouping: true,
      groupingStartedAt: 300,
      grouping_failed: false,
      grouping_error: null,
    });
    expect(retryStartedState.exploredCircuits[1].isActiveGrouping).toBe(false);

    const retryFinishedState = circuitExplorerReducer(retryStartedState, {
      type: 'GROUPING_RETRY_FINISHED',
      rank: 1,
      grouped: {
        supernodes: [['Retry group', 'feature-a']],
        supernode_explanations: ['Retry succeeded.'],
        supernode_member_reasons: [{ 'feature-a': 'Belongs to retry group.' }],
        grouping_model: 'sonnet',
        grouping_failed: true,
        grouping_error: 'fallback used',
      },
    });

    expect(retryFinishedState.exploredCircuits[0]).toMatchObject({
      ready: true,
      isActiveGrouping: false,
      grouping_failed: true,
      grouping_error: 'fallback used',
      supernodes: [['Retry group', 'feature-a']],
    });
  });
});
