import os
from uvicorn import Server, Config
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from routers.image import license_detect

app = FastAPI()

# @app.get("/hello")
# def hello():
#     return {"message":" hellow world"}

# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")

app.include_router(license_detect.router)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    server = Server(Config(app, host="127.0.0.1", port=port, lifespan="on"))
    server.run()