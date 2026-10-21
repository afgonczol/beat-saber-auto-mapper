"""WS9: FastAPI app and job queue for the local web UI.

    from bsmapper.server.app import create_app      # the app (real pipeline by default)
    from bsmapper.server.runner import JobRunner    # PipelineJobRunner / MockJobRunner

The OpenAPI spec lives in ``web/openapi.json``; regenerate it with
``python -m bsmapper.server.openapi``.
"""
