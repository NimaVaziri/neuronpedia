# Gemma-2-2B Local Runbook

This runbook covers a local Neuronpedia setup for `gemma-2-2b` with the `gemmascope-transcoder-16k` source set. It includes the web app, local Postgres, graph server, and the Neuronpedia dataset import needed for dashboards, activations, explanations, and labels. The standalone inference server is optional because the graph server handles its own model inference path.

## Current Script Coverage

The repo includes a self-contained local runner for this runbook:

```bash
scripts/gemma-2-2b-local.sh all
```

That starts the local web app/Postgres and graph server, imports Gemma-2-2B Gemmascope transcoder data, then prints verification counts.

The `all` command also reports total wall-clock time when it completes or fails.

By default the runner starts the web app with local Next.js dev mode instead of the Docker production image. This avoids blocking local bootstrap on `next build` lint failures. Use `--webapp docker` only when you specifically want to test the production webapp image.

For Apple Silicon graph via MPS:

```bash
scripts/gemma-2-2b-local.sh all --graph mps
```

The MPS path uses `apps/graph/.venv/bin/python` by default. Override with `GRAPH_PYTHON=/path/to/python` if needed.

For a smaller smoke/import pass:

```bash
scripts/gemma-2-2b-local.sh all --layers 20
```

By default, layer/source imports run sequentially to avoid overloading local Postgres. If your local Postgres has enough connection headroom, you can import multiple layers at once:

```bash
scripts/gemma-2-2b-local.sh import --layers 0-25 --import-jobs 2
```

Use higher values cautiously; each import downloads, decompresses, and inserts large feature/activation/explanation batches. If Postgres returns `53300`, reduce `--import-jobs` or use the default sequential import.

Useful subcommands:

- `scripts/gemma-2-2b-local.sh up`: start services without importing all data.
- `scripts/gemma-2-2b-local.sh all --with-inference`: also start the standalone inference server for non-graph webapp features that call it directly.
- `scripts/gemma-2-2b-local.sh all --webapp docker`: run the production webapp image path, including `next build`.
- `scripts/gemma-2-2b-local.sh import --layers 20 --explanations only`: rerun explanation import for one layer.
- `scripts/gemma-2-2b-local.sh verify`: print DB counts for sources, features, activations, and explanations.
- `scripts/gemma-2-2b-local.sh down`: stop Docker services without deleting Postgres volumes.

The script wraps the maintained Make/Docker paths and the admin import endpoint. The underlying pieces are still useful when debugging individual services.

Before this script existed, there was not one self-contained script that did all of this end to end.

The closest supported entry points are:

- `make webapp-localhost-run` or `make webapp-localhost-dev`: starts the web app with local Postgres and runs DB initialization.
- `make inference-localhost-dev MODEL_SOURCESET=gemma-2-2b.gemmascope-transcoder-16k`: optionally starts the standalone inference server for Gemma-2-2B Gemmascope transcoders. This is not needed for graph generation.
- `make graph-localhost-dev`: starts the graph server, using `apps/graph/.env` for graph-specific model/transcoder settings.
- `apps/webapp/app/api/admin/import/route.ts`: the actual webapp import path for source metadata, features, activations, and explanations.
- `apps/webapp/scripts/import-activations.js`: a narrow supplemental importer for activations only. It does not replace the admin import flow.

The shell scripts in `apps/webapp/scripts/run-local-webapp.sh` and `apps/graph/scripts/run-local-graph.sh` are not general local setup scripts right now; they contain hard-coded paths/model settings for a different environment.

## Data Responsibilities

Local Postgres stores Neuronpedia data:

- Models, releases, source sets, sources, and inference-host metadata.
- Feature rows in `Neuron`.
- Activation examples in `Activation`.
- Explanations in `Explanation`; these are the main labels the UI uses for feature labeling.
- Feature metadata such as `vectorLabel`, logits, sparsity, and correlated feature fields when present in the exported feature files.

The graph server does not load transcoder weights from Postgres. It downloads model and transcoder artifacts from Hugging Face or the configured transcoder source and caches them in the Hugging Face cache. The optional standalone inference server does the same for its own model/source configuration.

## Prerequisites

- Docker.
- Docker Compose through the `docker compose` plugin; the local runner depends on this for service startup, verification, and shutdown.
- Node.js 22 and npm if running the web app outside Docker.
- Poetry if running Python services outside Docker.
- A Hugging Face token with access to `google/gemma-2-2b`, set in root `.env` or `apps/graph/.env`.
- An Anthropic API key in `apps/graph/.env`; graph grouping/explanation endpoints use Anthropic, and the local runner requires this key before starting graph.
- Optional: `OPENAI_API_KEY` if you want semantic explanation search locally.
- Enough disk space for Postgres data plus model/transcoder caches. Importing all Gemma-2-2B transcoder sources can be large and slow.

## Environment Files

Run this once if `.env` does not exist:

```bash
make init-env
```

Before running the local runner, make sure `HF_TOKEN` exists in root `.env` or `apps/graph/.env`:

```bash
HF_TOKEN=<your-hugging-face-token>
```

For Docker Compose, root `.env.localhost` is used. It already points services at Docker hostnames such as `postgres`, `inference`, and `graph`.

For local webapp development from `apps/webapp`, `apps/webapp/.env.localhost` is used. If you run the standalone inference server, make sure it contains local service flags that match the ports below:

```bash
USE_LOCALHOST_INFERENCE=true
INFERENCE_SERVER_SECRET=localhost-secret
USE_LOCALHOST_GRAPH=true
GRAPH_SERVER_SECRET=SECRET
```

For graph, create `apps/graph/.env`:

```bash
SECRET=SECRET
HF_TOKEN=<your-hugging-face-token>
ANTHROPIC_API_KEY=<your-anthropic-api-key>
TRANSCODER_SET=gemma
```

If the web app is running in Docker Compose, the graph secret must match `GRAPH_SERVER_SECRET` from root `.env.localhost`. If the web app is running directly from `apps/webapp`, it must match `GRAPH_SERVER_SECRET` from `apps/webapp/.env.localhost`.

Similarly, if the web app is running directly from `apps/webapp` and the optional standalone inference server is started through the Make/Docker path, `INFERENCE_SERVER_SECRET` in `apps/webapp/.env.localhost` must match root `.env.localhost`.

## Start Local Postgres And Web App

Containerized path:

```bash
make webapp-localhost-build
make webapp-localhost-run
```

This starts `postgres`, runs `db-init`, and starts the web app on `http://localhost:3000`.

Local Next.js dev path:

```bash
docker compose -f docker/compose.yaml \
  --env-file .env.localhost \
  --env-file .env \
  up postgres db-init
```

Then in a separate terminal:

```bash
cd apps/webapp
npm install
npm run dev:localhost
```

Use this path when you want faster frontend/API iteration. The web app will run on `http://localhost:3000` and connect to Postgres on `localhost:5432`.

## Import Gemma-2-2B Data Into Postgres

The supported UI path is:

1. Open `http://localhost:3000/admin`.
2. Find `gemma-2-2b`.
3. Expand `gemmascope-transcoder-16k`.
4. Click `Download All`, or download individual sources such as `20-gemmascope-transcoder-16k`.
5. Leave the process running until each source completes.

The CLI equivalent uses the same import endpoint. Run this from the repo root after the web app is up:

```bash
for layer in $(seq 0 25); do
  source_id="${layer}-gemmascope-transcoder-16k"
  curl --fail --no-buffer \
    "http://localhost:3000/api/admin/import?modelId=gemma-2-2b&sourceId=${source_id}&explanations=true"
done
```

That endpoint imports:

- Global config rows such as explanation/eval types.
- Release, model, source set, source, and inference-host metadata.
- Feature rows from `features/*.jsonl.gz`.
- Activation rows from `activations/*.jsonl.gz`.
- Explanation rows from `explanations/*.jsonl.gz`.

The importer is not resumable. If a source import is interrupted, rerun that source. To retry only explanations for an already imported source:

```bash
curl --fail --no-buffer \
  "http://localhost:3000/api/admin/import?modelId=gemma-2-2b&sourceId=20-gemmascope-transcoder-16k&explanations=only"
```

To skip explanations for a faster metadata/features/activations import:

```bash
curl --fail --no-buffer \
  "http://localhost:3000/api/admin/import?modelId=gemma-2-2b&sourceId=20-gemmascope-transcoder-16k&explanations=false"
```

## Verify Imported Data

If using Docker Postgres:

```bash
docker compose -f docker/compose.yaml \
  --env-file .env.localhost \
  --env-file .env \
  exec postgres psql -U postgres -d postgres
```

Useful checks:

```sql
SELECT count(*) AS sources
FROM "Source"
WHERE "modelId" = 'gemma-2-2b'
  AND "setName" = 'gemmascope-transcoder-16k';

SELECT count(*) AS features
FROM "Neuron"
WHERE "modelId" = 'gemma-2-2b'
  AND "sourceSetName" = 'gemmascope-transcoder-16k';

SELECT count(*) AS activations
FROM "Activation"
WHERE "modelId" = 'gemma-2-2b'
  AND "layer" LIKE '%gemmascope-transcoder-16k';

SELECT count(*) AS explanations
FROM "Explanation" e
JOIN "Neuron" n
  ON n."modelId" = e."modelId"
 AND n."layer" = e."layer"
 AND n."index" = e."index"
WHERE n."modelId" = 'gemma-2-2b'
  AND n."sourceSetName" = 'gemmascope-transcoder-16k';
```

For a fully imported source set, `sources` should normally be `26` for layers `0` through `25`. If the public S3 dataset changes, treat the admin panel's listed sources as the source of truth.

## Optional: Start The Standalone Inference Server

The graph server does not need this service. Start it only when you want to use webapp features that call the standalone inference server directly, such as non-graph activation testing or steering flows.

Install and build once:

```bash
make inference-localhost-install
make inference-localhost-build USE_LOCAL_HF_CACHE=1
```

Run:

```bash
make inference-localhost-dev \
  MODEL_SOURCESET=gemma-2-2b.gemmascope-transcoder-16k \
  USE_LOCAL_HF_CACHE=1
```

Use the CUDA variants on NVIDIA hosts:

```bash
make inference-localhost-build-gpu USE_LOCAL_HF_CACHE=1
make inference-localhost-dev-gpu \
  MODEL_SOURCESET=gemma-2-2b.gemmascope-transcoder-16k \
  USE_LOCAL_HF_CACHE=1
```

The inference server listens on `http://localhost:5002`. The web app uses it when `USE_LOCALHOST_INFERENCE=true`.

## Start The Graph Server

Docker CPU/CUDA path:

```bash
make graph-localhost-install
make graph-localhost-build USE_LOCAL_HF_CACHE=1
make graph-localhost-dev USE_LOCAL_HF_CACHE=1
```

For this path, `TRANSCODER_SET=gemma` should be present in `apps/graph/.env`, and root `.env.localhost` should provide the `GRAPH_SERVER_SECRET` value used by the web app.

Use the CUDA variants on NVIDIA hosts:

```bash
make graph-localhost-build-gpu USE_LOCAL_HF_CACHE=1
make graph-localhost-dev-gpu USE_LOCAL_HF_CACHE=1
```

The graph server listens on `http://localhost:5004`. The web app uses it when `USE_LOCALHOST_GRAPH=true`.

For Apple Silicon MPS, prefer running graph directly instead of Docker because Docker cannot expose MPS:

```bash
cd apps/graph
poetry install
PYTORCH_ENABLE_MPS_FALLBACK=1 \
SECRET=SECRET \
HF_TOKEN=<your-hugging-face-token> \
ANTHROPIC_API_KEY=<your-anthropic-api-key> \
TRANSCODER_SET=gemma \
poetry run python start.py \
  --model_id google/gemma-2-2b \
  --transcoder_set gemma \
  --device mps \
  --port 5004
```

## Smoke Checks

```bash
curl --fail http://localhost:3000
curl --fail http://localhost:5002/docs
curl --fail http://localhost:5004/docs
```

Then open:

- `http://localhost:3000/gemma-2-2b/gemmascope-transcoder-16k` to browse the imported source set.
- `http://localhost:3000/gemma-2-2b/graph` to create a graph through the local graph server.

## Known Gaps

- There is no resumable all-Gemma importer. The admin import endpoint can be scripted, but interrupted sources must be rerun.
- `apps/webapp/scripts/import-activations.js` only imports activations after source metadata exists. It does not import features, explanations, source metadata, or config.
- The local graph server and optional standalone inference server download model/transcoder artifacts independently from the DB import. A complete Postgres import does not imply the Python services have warmed their Hugging Face caches.
- The Make targets are the maintained startup path; the `apps/*/scripts/run-local-*.sh` files should not be treated as portable runbooks until they are parameterized.
