from fastapi import FastAPI, Form
from pydantic import BaseModel

app = FastAPI()


@app.post("/classify")
def process_strings(char: str = Form(...), Kategorie: str = Form(...), Unterkategorie: str = Form(...)):
    result = "firstLevel"
    return result
