"""Main FastHTML application."""

import fasthtml.live_reload


# FastHTML's native live-reload has a bug that fails to handle Uvicorn websocket
# disconnects gracefully, causing loud stack traces in the terminal on every save.
# See: https://github.com/AnswerDotAI/fasthtml/issues/
# 
# We are forced to monkey-patch `fasthtml.live_reload_ws` globally before `fast_app`
# is imported because `FastHTMLWithLiveReload` binds the route in its constructor
# and Starlette caches the endpoint reference, meaning we cannot easily override
# the route endpoint after the app is initialized.
async def _patched_live_reload_ws(websocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
    except Exception:
        pass

fasthtml.live_reload.live_reload_ws = _patched_live_reload_ws

from fasthtml.common import fast_app, serve  # noqa: E402

from ui.routes import dashboard, datasets, experiments, generate, record, train  # noqa: E402

app, rt = fast_app(live=True)

# Register routes
dashboard.setup_routes(app, rt)
datasets.setup_routes(app, rt)
generate.setup_routes(app, rt)
record.setup_routes(app, rt)
train.setup_routes(app, rt)
experiments.setup_routes(app, rt)

if __name__ == "__main__":
    serve(appname="ui.main")
