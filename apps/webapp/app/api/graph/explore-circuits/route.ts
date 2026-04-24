import {
  getAuthHeaderForGraphServerRequest,
  getGraphServerRequestUrlForSourceSet,
  getIsRunpodServerlessHostForSourceSet,
  LOCALHOST_GRAPH_HOST,
} from '@/lib/db/graph-host-source';
import { GRAPH_SERVER_SECRET, USE_LOCALHOST_GRAPH } from '@/lib/env';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json();
  const modelId = body.modelId || body.model_id || body.graph_data?.metadata?.scan;
  const sourceSetName =
    body.sourceSetName ||
    body.graph_data?.metadata?.feature_details?.neuronpedia_source_set ||
    body.graph_data?.metadata?.info?.neuronpedia_source_set;
  const { resolveOnly } = body;

  if (!modelId) {
    return NextResponse.json({ error: 'modelId is required' }, { status: 400 });
  }

  body.modelId = modelId;
  body.model_id = body.model_id || modelId;
  body.sourceSetName = sourceSetName;

  // Resolve the graph server URL and secret for the client to call directly
  let url: string;
  let secret: string;
  try {
    if (USE_LOCALHOST_GRAPH) {
      url = `${LOCALHOST_GRAPH_HOST}/explore-circuits`;
      secret = GRAPH_SERVER_SECRET;
    } else {
      const isRunpod = await getIsRunpodServerlessHostForSourceSet(modelId, sourceSetName || '');
      url = await getGraphServerRequestUrlForSourceSet(modelId, sourceSetName || '', 'explore-circuits', isRunpod);
      const headers = getAuthHeaderForGraphServerRequest(isRunpod);
      secret = headers['x-secret-key'] || '';
    }
  } catch (e) {
    return NextResponse.json({ error: `Failed to resolve graph server: ${e}` }, { status: 500 });
  }

  if (resolveOnly) {
    return NextResponse.json({ url, secret });
  }

  const upstream = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-secret-key': secret,
    },
    body: JSON.stringify(body),
  });

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text();
    return new NextResponse(text || 'Failed to connect to graph server', {
      status: upstream.status || 500,
      headers: { 'Content-Type': upstream.headers.get('Content-Type') || 'text/plain' },
    });
  }

  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: {
      'Content-Type': upstream.headers.get('Content-Type') || 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  });
});
