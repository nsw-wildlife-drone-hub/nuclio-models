import json
import base64
from PIL import Image
import io
from model_handler import ModelHandler

def init_context(context):
    context.logger.info("Init context...  0%")

    # Read the DL model
    model = ModelHandler()
    context.user_data.model = model

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run OpenCV tracker...")
    data = event.body
    shapes = data.get("shapes")
    states = data.get("states")
    buf = io.BytesIO(base64.b64decode(data["image"]))
    states = io.BytesIO(base64.b64decode(states))
    image = Image.open(buf)

    results = {
        'shapes': [],
        'states': []
    }
    for shape in shapes:
        shape = context.user_data.model.infer(image, shape, states)
        results['shapes'].append(shape)
    results['state'] = data["image"]

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)