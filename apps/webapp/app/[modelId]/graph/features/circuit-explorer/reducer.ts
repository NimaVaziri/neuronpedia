import {
  CircuitExplorerState,
  ExploreCircuitEventData,
  ExploredCircuit,
  ExploreDoneEventData,
  ExploreGroupedEventData,
  ExploreProgress,
  ExploreProgressEventData,
  ExploreStatusEventData,
  GroupingModel,
  GroupingResponsePayload,
} from './types';

export type CircuitExplorerAction =
  | { type: 'SET_MODAL_OPEN'; isOpen: boolean }
  | { type: 'START_EXPLORATION' }
  | { type: 'CLEAR_PROGRESS' }
  | { type: 'STATUS_EVENT'; data: ExploreStatusEventData; eventAt?: number }
  | { type: 'PROGRESS_EVENT'; data: ExploreProgressEventData }
  | { type: 'CIRCUIT_EVENT'; data: ExploreCircuitEventData; defaultGroupingModel: GroupingModel; eventAt: number }
  | { type: 'GROUPED_EVENT'; data: ExploreGroupedEventData; eventAt: number }
  | { type: 'DONE_EVENT'; data: ExploreDoneEventData }
  | { type: 'GROUPING_RETRY_STARTED'; rank: number; eventAt: number }
  | { type: 'GROUPING_RETRY_FINISHED'; rank: number; grouped: GroupingResponsePayload };

const defaultProgress = (): ExploreProgress => ({
  phase: 'starting',
  completedSeeds: 0,
  totalSeeds: 0,
  circuitsFound: 0,
  bestScore: 0,
});

export const initialCircuitExplorerState: CircuitExplorerState = {
  isExploreCircuitsModalOpen: false,
  exploredCircuits: [],
  exploreProgress: null,
};

function getCircuitId(circuitLike: Partial<ExploredCircuit> & { pinned_ids?: string[]; circuit_id?: string }): string {
  return circuitLike.circuit_id || [...(circuitLike.pinned_ids || [])].sort().join(',');
}

function toExploredCircuit(
  circuitLike: Partial<ExploredCircuit> & { pinned_ids?: string[]; circuit_id?: string },
  options: {
    rank: number;
    defaultGroupingModel: GroupingModel;
    ready?: boolean;
    isActiveGrouping?: boolean;
    eventAt?: number;
    existing?: ExploredCircuit;
  },
): ExploredCircuit {
  const circuitId = getCircuitId(circuitLike);
  const pinnedIds = circuitLike.pinned_ids || options.existing?.pinned_ids || [];
  const replacementScore = circuitLike.replacement_score ?? options.existing?.replacement_score ?? 0;
  const completenessScore = circuitLike.completeness_score ?? options.existing?.completeness_score ?? 0;
  const combinedScore = circuitLike.combined_score ?? replacementScore + completenessScore;
  const nodeCount = circuitLike.node_count ?? options.existing?.node_count ?? pinnedIds.length;

  return {
    rank: options.rank,
    pinned_ids: pinnedIds,
    replacement_score: replacementScore,
    completeness_score: completenessScore,
    combined_score: combinedScore,
    node_count: nodeCount,
    seed: circuitLike.seed ?? options.existing?.seed ?? '',
    circuit_id: circuitId,
    supernodes: circuitLike.supernodes ?? options.existing?.supernodes ?? [],
    supernode_explanations: circuitLike.supernode_explanations ?? options.existing?.supernode_explanations ?? [],
    supernode_member_reasons: circuitLike.supernode_member_reasons ?? options.existing?.supernode_member_reasons ?? [],
    grouping_model: circuitLike.grouping_model ?? options.existing?.grouping_model ?? options.defaultGroupingModel,
    grouping_failed: circuitLike.grouping_failed ?? options.existing?.grouping_failed ?? false,
    grouping_error: circuitLike.grouping_error ?? options.existing?.grouping_error ?? null,
    ready: circuitLike.ready ?? options.ready ?? options.existing?.ready ?? false,
    isActiveGrouping:
      options.isActiveGrouping ?? circuitLike.isActiveGrouping ?? options.existing?.isActiveGrouping ?? false,
    groupingStartedAt: circuitLike.groupingStartedAt ?? options.existing?.groupingStartedAt,
    debugLastEvent: circuitLike.debugLastEvent ?? options.existing?.debugLastEvent,
    debugLastEventAt: circuitLike.debugLastEventAt ?? options.existing?.debugLastEventAt,
    sufficiency_added_ids: circuitLike.sufficiency_added_ids ?? options.existing?.sufficiency_added_ids,
  };
}

function updateProgress(
  progress: ExploreProgress | null,
  patch: Partial<ExploreProgress> & Pick<ExploreProgress, 'phase'>,
): ExploreProgress {
  return {
    ...(progress || defaultProgress()),
    ...patch,
  };
}

export function circuitExplorerReducer(
  state: CircuitExplorerState,
  action: CircuitExplorerAction,
): CircuitExplorerState {
  switch (action.type) {
    case 'SET_MODAL_OPEN':
      return { ...state, isExploreCircuitsModalOpen: action.isOpen };

    case 'START_EXPLORATION':
      return {
        ...state,
        exploredCircuits: [],
        exploreProgress: defaultProgress(),
      };

    case 'CLEAR_PROGRESS':
      return {
        ...state,
        exploreProgress: null,
      };

    case 'STATUS_EVENT': {
      const { data, eventAt } = action;
      if (data.phase === 'grouping') {
        return {
          ...state,
          exploreProgress: updateProgress(state.exploreProgress, {
            phase: 'grouping',
            groupingCircuit: data.grouping_circuit,
            totalToGroup: data.total_to_group,
            debugLastGroupingEvent: `grouping rank=${data.grouping_circuit} id=${(data.circuit_id || 'none').slice(0, 24)} at=${eventAt}`,
          }),
          exploredCircuits: state.exploredCircuits.map((circuit) => ({
            ...circuit,
            isActiveGrouping: !!data.circuit_id && circuit.circuit_id === data.circuit_id,
            debugLastEvent:
              data.circuit_id && circuit.circuit_id === data.circuit_id
                ? `status:grouping rank=${data.grouping_circuit} at=${eventAt}`
                : circuit.debugLastEvent,
            debugLastEventAt:
              data.circuit_id && circuit.circuit_id === data.circuit_id ? eventAt : circuit.debugLastEventAt,
            groupingStartedAt:
              data.circuit_id && circuit.circuit_id === data.circuit_id && !circuit.groupingStartedAt
                ? eventAt
                : circuit.groupingStartedAt,
          })),
        };
      }

      if (data.phase === 'building') {
        return {
          ...state,
          exploreProgress: updateProgress(state.exploreProgress, {
            phase: 'building',
            completedSeeds: data.completed_seeds ?? state.exploreProgress?.completedSeeds ?? 0,
            totalSeeds: data.total_seeds ?? state.exploreProgress?.totalSeeds ?? 0,
            circuitsFound: data.circuits_found ?? state.exploreProgress?.circuitsFound ?? 0,
            bestScore: state.exploreProgress?.bestScore ?? 0,
            currentSeed: data.current_seed,
          }),
        };
      }

      return {
        ...state,
        exploreProgress: updateProgress(state.exploreProgress, {
          phase: 'exploring',
          completedSeeds: data.completed_seeds ?? state.exploreProgress?.completedSeeds ?? 0,
          totalSeeds: data.total_seeds ?? state.exploreProgress?.totalSeeds ?? 0,
          circuitsFound: data.circuits_found ?? state.exploreProgress?.circuitsFound ?? 0,
          bestScore: state.exploreProgress?.bestScore ?? 0,
        }),
      };
    }

    case 'PROGRESS_EVENT':
      return {
        ...state,
        exploreProgress: updateProgress(state.exploreProgress, {
          phase: 'building',
          currentSeed: action.data.current_seed,
          totalSeeds: action.data.total_seeds ?? state.exploreProgress?.totalSeeds ?? 0,
          buildStep: action.data.step,
          buildMaxSteps: action.data.max_steps,
          buildFeatures: action.data.features,
          buildCScore: action.data.c_score,
        }),
      };

    case 'CIRCUIT_EVENT': {
      const score =
        action.data.combined_score ?? (action.data.replacement_score ?? 0) + (action.data.completeness_score ?? 0);
      const nextCircuit = toExploredCircuit(action.data, {
        rank: state.exploredCircuits.length + 1,
        defaultGroupingModel: action.defaultGroupingModel,
        ready: false,
        isActiveGrouping: false,
        eventAt: action.eventAt,
      });
      nextCircuit.debugLastEvent = `circuit discovered rank=${nextCircuit.rank}`;
      nextCircuit.debugLastEventAt = action.eventAt;

      return {
        ...state,
        exploredCircuits: [...state.exploredCircuits, nextCircuit],
        exploreProgress: updateProgress(state.exploreProgress, {
          phase: 'exploring',
          completedSeeds: action.data.completed_seeds ?? state.exploreProgress?.completedSeeds ?? 0,
          totalSeeds: action.data.total_seeds ?? state.exploreProgress?.totalSeeds ?? 0,
          circuitsFound: action.data.circuits_found ?? state.exploreProgress?.circuitsFound ?? 0,
          bestScore: Math.max(score, state.exploreProgress?.bestScore ?? 0),
        }),
      };
    }

    case 'GROUPED_EVENT': {
      const groupedCircuitId = action.data.circuit_id ?? getCircuitId(action.data.circuit);
      const hasMatch = state.exploredCircuits.some((circuit) => circuit.circuit_id === groupedCircuitId);

      return {
        ...state,
        exploreProgress: updateProgress(state.exploreProgress, {
          phase: state.exploreProgress?.phase || 'grouping',
          debugLastGroupedEvent: `grouped id=${(groupedCircuitId || 'none').slice(0, 24)} matched=${String(hasMatch)} at=${action.eventAt}`,
        }),
        exploredCircuits: state.exploredCircuits.map((circuit) => {
          if (groupedCircuitId !== circuit.circuit_id) return circuit;
          const merged = toExploredCircuit(action.data.circuit, {
            rank: circuit.rank,
            defaultGroupingModel: circuit.grouping_model,
            ready: true,
            isActiveGrouping: false,
            existing: circuit,
          });
          merged.debugLastEvent = `grouped at=${action.eventAt}`;
          merged.debugLastEventAt = action.eventAt;
          merged.groupingStartedAt = circuit.groupingStartedAt;
          return merged;
        }),
      };
    }

    case 'DONE_EVENT': {
      const finalCircuits = action.data.circuits || [];
      return {
        ...state,
        exploreProgress: null,
        exploredCircuits: state.exploredCircuits.map((existingCircuit) => {
          const finalCircuit = finalCircuits.find((circuit) => getCircuitId(circuit) === existingCircuit.circuit_id);
          if (!finalCircuit) {
            return {
              ...existingCircuit,
              isActiveGrouping: false,
            };
          }
          const merged = toExploredCircuit(finalCircuit, {
            rank: existingCircuit.rank,
            defaultGroupingModel: existingCircuit.grouping_model,
            ready: true,
            isActiveGrouping: false,
            existing: existingCircuit,
          });
          merged.groupingStartedAt = existingCircuit.groupingStartedAt;
          merged.debugLastEvent = existingCircuit.debugLastEvent;
          merged.debugLastEventAt = existingCircuit.debugLastEventAt;
          return merged;
        }),
      };
    }

    case 'GROUPING_RETRY_STARTED':
      return {
        ...state,
        exploredCircuits: state.exploredCircuits.map((circuit) =>
          circuit.rank === action.rank
            ? {
                ...circuit,
                ready: false,
                isActiveGrouping: true,
                groupingStartedAt: action.eventAt,
                grouping_failed: false,
                grouping_error: null,
              }
            : {
                ...circuit,
                isActiveGrouping: false,
              },
        ),
      };

    case 'GROUPING_RETRY_FINISHED':
      return {
        ...state,
        exploredCircuits: state.exploredCircuits.map((circuit) => {
          if (circuit.rank !== action.rank) return circuit;
          return {
            ...circuit,
            ...action.grouped,
            ready: true,
            isActiveGrouping: false,
            grouping_failed: action.grouped.grouping_failed || false,
            grouping_error: action.grouped.grouping_error || null,
          };
        }),
      };

    default:
      return state;
  }
}
