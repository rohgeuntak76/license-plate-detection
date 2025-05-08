import os
from uvicorn import Server, Config
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from routers.image import objects_detection,plate_number_detection
from routers.video import video_object_detect

app = FastAPI()

# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")

app.include_router(objects_detection.router)
app.include_router(plate_number_detection.router)
app.include_router(video_object_detect.router)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    server = Server(Config(app, host="127.0.0.1", port=port, lifespan="on"))
    server.run()