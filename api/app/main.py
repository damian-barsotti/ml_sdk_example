from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from routers import MODELS_TO_DEPLOY


# VERSION
version = '0.1.0'
title = "ACL IMDB Review"
description = ("Movie Review Prediction, "
               "Cordoba, Argentina.")

# APP
app = FastAPI(
    title=title,
    description=description,
    version=version,
)

# # MODELS
for model in MODELS_TO_DEPLOY:
    model = model()
    app.include_router(model.router,
                       prefix=f"/{model.MODEL_NAME}",
                       tags=[model.MODEL_NAME])


# SWAGGER CUSTOMIZATION
def custom_openapi():
    # Cache
    if app.openapi_schema:
        return app.openapi_schema

    # Default openapi schema
    openapi_schema = get_openapi(
        title=title,
        description=description,
        version=version,
        routes=app.routes
    )

    # Customization: Redoc logo
    openapi_schema["info"]["x-logo"] = {
        "url": "https://www.famaf.unc.edu.ar/static/assets/logoFaMAF.svg"
    }

    # # Customization: Remove duplicated models
    # openapi_schema['components']['schemas'] = {
    #     value['title']: value
    #     for _, value in openapi_schema['components']['schemas'].items()
    # }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
