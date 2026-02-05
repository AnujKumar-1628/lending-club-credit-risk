from fastapi import FastAPI
from api.routes.predict import router

# -------------------------------------------------
# FastAPI app initialization
# -------------------------------------------------
app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0.0",
)


# -------------------------------------------------
# Simple root endpoint (sanity check)
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Credit Risk API is up",
        "docs": "/docs",
    }


# -------------------------------------------------
# Register all prediction routes
# -------------------------------------------------
app.include_router(router)
