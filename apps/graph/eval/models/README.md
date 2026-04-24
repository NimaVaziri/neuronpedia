Model-specific eval configuration and run notes live here.

Layout:

- `models/<model-id>/...` for configuration, parameter snapshots, run notes, and result artifacts tied to one model
- shared eval code stays in `apps/graph/eval/`

Current convention:

- `models/<model-id>/iapc-parameters.md` for the recorded IA+PC run configuration for that model
- `models/<model-id>/results/` for eval outputs produced for that model
- `models/<model-id>/graphs/` for local eval graph JSONs generated for that model
