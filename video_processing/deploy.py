from __future__ import annotations

import modal

from main_api import app as api_app
from main_inference import app as inference_app
from main_stabilization import app as stabilization_app
from main_tracking import app as tracking_app

if __name__ == '__main__':
    # Deploy apps
    app = (
        modal.App('windsurf-analysis')
        .include(api_app)
        .include(inference_app)
        .include(stabilization_app)
        .include(tracking_app)
    )
    with modal.enable_output():
        app.deploy()
