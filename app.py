from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Text Summarizer App")

# Don't load at startup — load only once on first request
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        tokenizer = T5Tokenizer.from_pretrained("karan-desai-7299/t5-text-summarizer")
        model = T5ForConditionalGeneration.from_pretrained(
            "karan-desai-7299/t5-text-summarizer",
            torch_dtype=torch.float16,  # half precision = half memory
            low_cpu_mem_usage=True
        )
        model.eval()

device = torch.device("cpu")
templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue: str) -> str:
    load_model()
    dialogue = clean_data(dialogue)
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=2,
        early_stopping=True
    )
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")