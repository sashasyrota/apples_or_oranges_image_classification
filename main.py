import os

from fastapi import FastAPI, UploadFile, File, Request
import tensorflow as tf
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from classifier import classify

app = FastAPI()


STATIC_FOLDER = "static"
UPLOAD_FOLDER = os.path.join("static", "uploads")

app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")

cnn_model = tf.keras.models.load_model(
    STATIC_FOLDER + "/models/" + "apple_orange_classifier.keras"
)

templates = Jinja2Templates(directory="templates")


@app.get("/")
def main_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={
        "background_url": "static/"
                          "images/"
                          "top-view-cut-fresh-oranges-red-apples-"
                          "light-black-background-with-free-space_461922-11140.jpg"
    })


@app.post("/classify")
async def classify_image(request: Request, file: UploadFile = File(...)):
    apple_img = "apple-1702316_1280.jpg"
    orange_img = "istockphoto-477836156-612x612.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    label, prob = classify(cnn_model, file_path)
    prob = float(round((prob * 100), 2))
    background_img = apple_img if label == "apple" else orange_img
    return templates.TemplateResponse(
        request=request,
        name="model_result.html",
        context={
        "background_url": f"static/images/{background_img}",
        "label": label,
        "prob": prob
    })