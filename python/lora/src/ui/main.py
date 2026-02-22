"""Main FastHTML application."""

from fasthtml.common import fast_app, serve

from ui.routes import dashboard, datasets, generate, record, train

app, rt = fast_app(live=True)

# Register routes
dashboard.setup_routes(app, rt)
datasets.setup_routes(app, rt)
generate.setup_routes(app, rt)
record.setup_routes(app, rt)
train.setup_routes(app, rt)

if __name__ == "__main__":
    serve(appname="ui.main")
