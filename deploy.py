from __future__ import annotations
import os

import modal

from server.main_backend_frontend import app as server_app
from server.main_inference import app as inference_app
from server.main_stabilization import app as stabilization_app

if __name__ == '__main__':
    # Build frontend
    os.system('cd server/frontend && npm install && npm run build && cd ../..')

    # Deploy apps
    app = modal.App('windsurf-analysis').include(server_app).include(inference_app).include(stabilization_app)
    with modal.enable_output():
        app.deploy()
