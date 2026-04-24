'use client';

import { createContext, Dispatch, ReactNode, useContext, useMemo, useReducer } from 'react';
import { circuitExplorerReducer, CircuitExplorerAction, initialCircuitExplorerState } from './reducer';
import { CircuitExplorerState } from './types';

type CircuitExplorerContextType = CircuitExplorerState & {
  dispatch: Dispatch<CircuitExplorerAction>;
  setIsExploreCircuitsModalOpen: (isOpen: boolean) => void;
};

const CircuitExplorerContext = createContext<CircuitExplorerContextType | undefined>(undefined);

export function CircuitExplorerProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(circuitExplorerReducer, initialCircuitExplorerState);

  const value = useMemo(
    () => ({
      ...state,
      dispatch,
      setIsExploreCircuitsModalOpen: (isOpen: boolean) => dispatch({ type: 'SET_MODAL_OPEN', isOpen }),
    }),
    [state],
  );

  return <CircuitExplorerContext.Provider value={value}>{children}</CircuitExplorerContext.Provider>;
}

export function useCircuitExplorerContext() {
  const context = useContext(CircuitExplorerContext);
  if (!context) {
    throw new Error('useCircuitExplorerContext must be used within a CircuitExplorerProvider');
  }
  return context;
}
