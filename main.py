import re

import argostranslate.package
import argostranslate.translate
import tiktoken
from fastapi import FastAPI, Form

app = FastAPI()


def preprocess_input(char: str) -> str:
    """
    Cleans, translates and anonymizes the data before passing it for classification
    :param char:
    :return:
    """
    # clean input
    char = char.replace("\n", " ")
    char = char.strip()

    # translate
    from_code = "de"
    to_code = "en"
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code,
            available_packages,
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())
    char = argostranslate.translate.translate(char, from_code, to_code)

    # anonymize email address
    char = re.sub(
        "(([\w-]+(?:\.[\w-]+)*)@((?:[\w-]+\.)*\w[\w-]{0,66})\.([a-z]{2,6}(?:\.[a-z]{2})?))",
        "<EMAIL>",
        char,
    )
    # anonymize person names
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    doc = nlp(char)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            char = char.replace(ent.text, "<PERSON>")

    # limit length
    tiktoken.get_encoding("r50k_base")
    encoder = tiktoken.encoding_for_model("ada")
    char = encoder.decode(encoder.encode(char)[: 2048 - 9 - 1])
    char = char + "\n\n###\n\n"
    return char


def gpt_3_classification(char: str) -> str:
    """
    Classifies the data using the GPT-3 API
    :param char:
    :return:
    """
    import openai
    import os
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model = openai.FineTune.retrieve("ft-oGEtDF48FowWEF0mUhaDskGh")
    response = openai.Completion.create(
        model=model.fine_tuned_model,
        prompt=char,
        temperature=0,
        max_tokens=1,
        logprobs=2,
    )
    return response["choices"][0]["text"]

@app.post("/classify")
def process_strings(char: str = Form(...), kategorie: str = Form(...), unterkategorie: str = Form(...)):
    return gpt_3_classification(preprocess_input(char)).strip()
