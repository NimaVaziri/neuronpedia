'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { GraphModalProvider } from '@/components/provider/graph-modal-provider';
import { useGraphContext } from '@/components/provider/graph-provider';
import { LoadingSquare } from '@/components/svg/loading-square';
import { useSearchParams } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import GraphFeatureDetail from './feature-detail';
import { CircuitExplorerProvider, useCircuitExplorerContext } from './features/circuit-explorer/context';
import FidelityDashboard from './fidelity-dashboard';
import GenerateGraphModal from './generate-graph-modal';
import GraphToolbar from './graph-toolbar';
import LinkGraph from './link-graph';
import ExploreCircuitsModal, { ExploreResultsPanel } from './features/circuit-explorer';
import CopyModal from './modals/copy-modal';
import LoadSubgraphModal from './modals/load-subgraph-modal';
import SaveSubgraphModal from './modals/save-subgraph-modal';
import SteerModal from './modals/steer-modal';
import WelcomeModal from './modals/welcome-modal';
import GraphNodeConnections from './node-connections';
import Subgraph from './subgraph';

type RightPanelTab = 'circuit-explorer' | 'feature-activations';

function RightPanel() {
  const { exploredCircuits, exploreProgress, setIsExploreCircuitsModalOpen } = useCircuitExplorerContext();
  const hasCircuitExplorerContent = exploredCircuits.length > 0 || !!exploreProgress;
  const [activeTab, setActiveTab] = useState<RightPanelTab>(
    hasCircuitExplorerContent ? 'circuit-explorer' : 'feature-activations',
  );
  const hadCircuitExplorerContentRef = useRef(hasCircuitExplorerContent);

  useEffect(() => {
    if (hasCircuitExplorerContent && !hadCircuitExplorerContentRef.current) {
      setActiveTab('circuit-explorer');
    }
    hadCircuitExplorerContentRef.current = hasCircuitExplorerContent;
  }, [hasCircuitExplorerContent]);

  return (
    <div className="hidden h-full w-full flex-col overflow-hidden border-l border-slate-200 bg-white sm:flex sm:w-[47%] sm:min-w-[47%] sm:max-w-[47%]">
      <div className="flex items-center gap-x-1 border-b border-slate-200 px-2 py-2">
        <button
          type="button"
          onClick={() => setActiveTab('circuit-explorer')}
          className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
            activeTab === 'circuit-explorer'
              ? 'bg-sky-100 text-sky-700'
              : 'text-slate-500 hover:bg-slate-100 hover:text-slate-700'
          }`}
        >
          Circuit Explorer
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('feature-activations')}
          className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
            activeTab === 'feature-activations'
              ? 'bg-sky-100 text-sky-700'
              : 'text-slate-500 hover:bg-slate-100 hover:text-slate-700'
          }`}
        >
          Feature Activations
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-hidden">
        {activeTab === 'circuit-explorer' ? (
          hasCircuitExplorerContent ? (
            <ExploreResultsPanel embedded onClose={() => setActiveTab('feature-activations')} />
          ) : (
            <div className="flex h-full flex-col items-center justify-center px-6 text-center text-sm text-slate-500">
              <div className="text-base font-semibold text-slate-700">Circuit Explorer</div>
              <div className="mt-2 max-w-xs">
                Explore candidate circuits for the current graph, then review or validate them here.
              </div>
              <button
                type="button"
                onClick={() => setIsExploreCircuitsModalOpen(true)}
                className="mt-4 rounded-md bg-sky-600 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-sky-700"
              >
                Open Circuit Explorer
              </button>
            </div>
          )
        ) : (
          <GraphFeatureDetail />
        )}
      </div>
    </div>
  );
}

export default function GraphWrapper({ hasSlug, showGenerateModal }: { hasSlug: boolean; showGenerateModal: boolean }) {
  const { isLoadingGraphData, selectedMetadataGraph, loadingGraphLabel, selectedModelId, selectedSourceSetName } =
    useGraphContext();
  const { isGraphEnabledForSourceSet } = useGlobalContext();
  const [showFidelitySidebar, setShowFidelitySidebar] = useState(true);

  const searchParams = useSearchParams();
  const isEmbed = searchParams.get('embed') === 'true';
  return (
    <GraphModalProvider>
      <CircuitExplorerProvider>
        <div
          className={`${isEmbed ? 'h-[calc(100%_-_20px)] max-h-screen min-h-[calc(100%_-_20px)]' : 'h-[calc(100vh_-_75px)] max-h-[calc(100vh_-_75px)] min-h-[calc(100vh_-_75px)]'} flex w-full flex-col justify-center px-1 text-slate-700 sm:px-4`}
        >
          <div className="flex w-full flex-1 flex-col items-center justify-center overflow-hidden">
            {/* <div>{JSON.stringify(visState)}</div> */}
            <div className="flex w-full flex-col">
              <GraphToolbar />
            </div>

            <div className="w-full flex-1 overflow-hidden pt-1">
              {isLoadingGraphData ? (
                <div className="flex h-full w-full flex-col items-center justify-center gap-y-3">
                  <LoadingSquare className="h-6 w-6" />
                  <div className="text-sm text-slate-400">
                    {loadingGraphLabel.length > 0 ? loadingGraphLabel : 'Loading...'}
                  </div>
                </div>
              ) : selectedMetadataGraph ? (
                <div className="flex h-full max-h-full w-full flex-col">
                  <div className="flex h-[50%] max-h-[50%] min-h-[50%] w-full flex-row pb-2">
                    <LinkGraph />
                    <GraphNodeConnections />
                  </div>
                  <div className="relative flex h-[50%] w-full flex-row pb-1 pt-1">
                    <div className="relative w-full sm:w-[53%] sm:min-w-[53%] sm:max-w-[53%]">
                      <Subgraph
                        showFidelitySidebar={showFidelitySidebar}
                        setShowFidelitySidebar={setShowFidelitySidebar}
                      />
                      {showFidelitySidebar && <FidelityDashboard />}
                    </div>
                    <RightPanel />
                  </div>
                </div>
              ) : (
                <div className="flex h-full w-full items-center justify-center">
                  <div className="text-center text-lg text-slate-400">
                    No graph selected. Choose one from the dropdown above.
                  </div>
                </div>
              )}
            </div>
          </div>
          <LoadSubgraphModal />
          <SaveSubgraphModal />
          <WelcomeModal hasSlug={hasSlug} showGenerateModal={showGenerateModal} />
          <GenerateGraphModal showGenerateModal={showGenerateModal} />
          <CopyModal />
          <ExploreCircuitsModal />
          {isGraphEnabledForSourceSet(selectedModelId, selectedSourceSetName) && <SteerModal />}
        </div>
      </CircuitExplorerProvider>
    </GraphModalProvider>
  );
}
