export type GroupingModel = 'sonnet' | 'haiku';
export type ExplorePhase = 'starting' | 'exploring' | 'building' | 'grouping';
export type GraphLinkEndpointRef = string | { node_id: string };

export type ExploreProgress = {
  phase: ExplorePhase;
  completedSeeds: number;
  totalSeeds: number;
  circuitsFound: number;
  bestScore: number;
  currentSeed?: number;
  buildStep?: number;
  buildMaxSteps?: number;
  buildFeatures?: number;
  buildCScore?: number;
  groupingCircuit?: number;
  totalToGroup?: number;
  debugLastGroupingEvent?: string;
  debugLastGroupedEvent?: string;
};

export type SupernodeMemberReasons = Record<string, string>;

export type ExploredCircuit = {
  rank: number;
  pinned_ids: string[];
  replacement_score: number;
  completeness_score: number;
  combined_score: number;
  node_count: number;
  seed: string;
  circuit_id: string;
  supernodes: string[][];
  supernode_explanations: string[];
  supernode_member_reasons: SupernodeMemberReasons[];
  grouping_model: GroupingModel;
  grouping_failed: boolean;
  grouping_error: string | null;
  ready: boolean;
  isActiveGrouping: boolean;
  groupingStartedAt?: number;
  debugLastEvent?: string;
  debugLastEventAt?: number;
  sufficiency_added_ids?: string[];
};

export type GroupingResponsePayload = Pick<
  ExploredCircuit,
  | 'supernodes'
  | 'supernode_explanations'
  | 'supernode_member_reasons'
  | 'grouping_model'
  | 'grouping_failed'
  | 'grouping_error'
> & {
  ready?: boolean;
};

export type ExploreStatusEventData =
  | {
      phase: 'exploring';
      completed_seeds?: number;
      total_seeds?: number;
      circuits_found?: number;
    }
  | {
      phase: 'building';
      completed_seeds?: number;
      total_seeds?: number;
      circuits_found?: number;
      current_seed?: number;
    }
  | {
      phase: 'grouping';
      grouping_circuit?: number;
      total_to_group?: number;
      circuit_id?: string;
    };

export type ExploreProgressEventData = {
  current_seed?: number;
  total_seeds?: number;
  step?: number;
  max_steps?: number;
  features?: number;
  c_score?: number;
};

export type ExploreCircuitEventData = {
  pinned_ids?: string[];
  circuit_id?: string;
  seed?: string;
  node_count?: number;
  replacement_score?: number;
  completeness_score?: number;
  combined_score?: number;
  supernodes?: string[][];
  supernode_explanations?: string[];
  supernode_member_reasons?: SupernodeMemberReasons[];
  grouping_model?: GroupingModel;
  grouping_failed?: boolean;
  grouping_error?: string | null;
  completed_seeds?: number;
  total_seeds?: number;
  circuits_found?: number;
};

export type ExploreGroupedEventData = {
  rank?: number;
  circuit_id?: string;
  circuit: Partial<ExploredCircuit> & {
    pinned_ids?: string[];
    circuit_id?: string;
  };
};

export type ExploreDoneEventData = {
  circuits?: Array<Partial<ExploredCircuit> & { pinned_ids?: string[]; circuit_id?: string }>;
};

export type CircuitExplorerState = {
  isExploreCircuitsModalOpen: boolean;
  exploredCircuits: ExploredCircuit[];
  exploreProgress: ExploreProgress | null;
};
