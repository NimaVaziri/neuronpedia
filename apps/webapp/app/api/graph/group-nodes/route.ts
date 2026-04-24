import {
  getAuthHeaderForGraphServerRequest,
  getGraphServerRequestUrlForSourceSet,
  getIsRunpodServerlessHostForSourceSet,
} from '@/lib/db/graph-host-source';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();
  const { modelId, sourceSetName, graphData, pinnedIds, prompt, groupingModel } = body;

  if (!modelId || !graphData || !pinnedIds) {
    return NextResponse.json({ error: 'modelId, graphData, and pinnedIds are required' }, { status: 400 });
  }

  const isRunpodServerlessHost = await getIsRunpodServerlessHostForSourceSet(
    modelId,
    sourceSetName || '',
  );

  const url = await getGraphServerRequestUrlForSourceSet(
    modelId,
    sourceSetName || '',
    'group-nodes',
    isRunpodServerlessHost,
  );

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaderForGraphServerRequest(isRunpodServerlessHost),
    },
    body: JSON.stringify({
      graph_data: graphData,
      pinned_ids: pinnedIds,
      prompt: prompt || '',
      grouping_model: groupingModel || 'sonnet',
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    return NextResponse.json(
      { error: `Graph server returned ${response.status}: ${text}` },
      { status: response.status },
    );
  }

  const result = await response.json();
  return NextResponse.json(result);
});
