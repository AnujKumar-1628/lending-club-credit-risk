from fastapi import FastAPI
from api.routes.predict import router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Credit Risk Prediction API",
        version="1.0.0",
    )

    @app.get("/")
    def root():
        return {
            "status": "running",
            "message": "Credit Risk API is up",
            "docs": "/docs",
        }

    app.include_router(router)
    return app


app = create_app()
