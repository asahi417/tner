import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src import TransformerNER

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
MODEL_CKPT = os.getenv('MODEL_CKPT', 'conll_2003_15db7244e38c1c4ab75e28a5c9419031')
CKPT_DIR = os.getenv("CKPT_DIR", './ckpt')
MODEL = TransformerNER(checkpoint=MODEL_CKPT)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index_ner.html", {"request": request})


@app.post("/process")
async def process(request: Request):
    data_json = await request.json()
    input_text = data_json['input_text']
    max_len = int(data_json['max_len'])
    ner_result = MODEL.predict([input_text], max_seq_length=max_len)[0]
    ner_result['return_probability'] = data_json["return_probability"]
    ner_result['model_ckpt'] = MODEL_CKPT
    return ner_result
