import { CLTGraphNode } from '../../graph-types';
import { ExploredCircuit } from './types';

type DiffSelection = [number | null, number | null];

type SelectedGraphLike = {
  nodes: CLTGraphNode[];
} | null | undefined;

export function computeCircuitDiff(
  exploredCircuits: ExploredCircuit[],
  diffSelection: DiffSelection,
  selectedGraph: SelectedGraphLike,
) {
  const [rankA, rankB] = diffSelection;
  if (rankA === null || rankB === null) return null;

  const circuitA = exploredCircuits.find((circuit) => circuit.rank === rankA);
  const circuitB = exploredCircuits.find((circuit) => circuit.rank === rankB);
  if (!circuitA || !circuitB) return null;

  const featuresA = new Set<string>(
    (circuitA.pinned_ids || []).filter((id: string) =>
      selectedGraph?.nodes.some((node) => node.node_id === id && node.feature_type === 'cross layer transcoder'),
    ),
  );
  const featuresB = new Set<string>(
    (circuitB.pinned_ids || []).filter((id: string) =>
      selectedGraph?.nodes.some((node) => node.node_id === id && node.feature_type === 'cross layer transcoder'),
    ),
  );

  const shared = [...featuresA].filter((id) => featuresB.has(id));
  const onlyA = [...featuresA].filter((id) => !featuresB.has(id));
  const onlyB = [...featuresB].filter((id) => !featuresA.has(id));

  const supernodesA: string[][] = circuitA.supernodes || [];
  const supernodesB: string[][] = circuitB.supernodes || [];

  const supernodeDiff: Array<{
    label: string;
    inA: boolean;
    inB: boolean;
    membersA: string[];
    membersB: string[];
  }> = [];
  const matchedB = new Set<number>();

  supernodesA.forEach((supernodeA) => {
    const labelA = supernodeA[0];
    const membersA = supernodeA.slice(1);
    let bestMatch = -1;
    let bestOverlap = 0;

    supernodesB.forEach((supernodeB, index) => {
      if (matchedB.has(index) || (bestMatch >= 0 && supernodesB[bestMatch][0] === labelA)) return;
      const membersB = supernodeB.slice(1);
      const overlap = membersA.filter((member) => membersB.includes(member)).length;
      if (supernodeB[0] === labelA || overlap > bestOverlap) {
        bestMatch = index;
        bestOverlap = overlap;
      }
    });

    if (bestMatch >= 0 && bestOverlap > 0) {
      matchedB.add(bestMatch);
      supernodeDiff.push({
        label: labelA,
        inA: true,
        inB: true,
        membersA,
        membersB: supernodesB[bestMatch].slice(1),
      });
    } else {
      supernodeDiff.push({ label: labelA, inA: true, inB: false, membersA, membersB: [] });
    }
  });

  supernodesB.forEach((supernodeB, index) => {
    if (!matchedB.has(index)) {
      supernodeDiff.push({
        label: supernodeB[0],
        inA: false,
        inB: true,
        membersA: [],
        membersB: supernodeB.slice(1),
      });
    }
  });

  return { circuitA, circuitB, shared, onlyA, onlyB, supernodeDiff };
}
