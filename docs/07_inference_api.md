# Inference API

A FastAPI-based inference service is implemented to serve predictions.

## Endpoints
- `/predict`: returns probability of loan default
- `/health`: basic health check endpoint

## Design Choices
- Input validation using Pydantic schemas
- Model and feature builder are loaded once at startup
- Swagger UI is available for easy testing

The API is designed to simulate how a trained model would be consumed by downstream systems.

