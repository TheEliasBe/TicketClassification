import logging
import subprocess
import time
import pandas as pd
import os
import openai
import wandb
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

def train(training_data_finished: bool):
    subprocess.run(["openai", "tools", "fine_tunes.prepare_data", "-f", "data/05_model_input/2022.jsonl", "-q"])
    openai.api_key = os.getenv('OPENAI_API_KEY')
    response = openai.File.create(file=open("data/05_model_input/2022_prepared_train.jsonl"), purpose="fine-tune")

    params = {
        "model": "ada",
        "n_epochs": 16,
        "batch_size": 0.2,
        "learning_rate_multiplier": 0.1,
        "prompt_loss_weight": 0.05
    }

    wandb.init(project="first_level_classification", config=params)
    wandb.log({"n_epochs": params["n_epochs"]})
    wandb.log({"model": params["model"]})
    wandb.log({"batch_size": params["batch_size"]})
    wandb.log({"learning_rate_multiplier": params["learning_rate_multiplier"]})
    wandb.log({"prompt_loss_weight": params["prompt_loss_weight"]})

    fine_tune_job = openai.FineTune.create(
        training_file=response.id,
        model=params["model"],
        n_epochs=params["n_epochs"],
    )

    cost = None
    # wait for finetuning to finish
    while True:
        events = openai.FineTune.list_events(id=fine_tune_job.id)
        log.debug(f"Fine-tuning job status: {events['data'][-1]['message']}")
        print(f"Fine-tuning job status: {events['data'][-1]['message']}")

        if "succeeded" in events["data"][-1]["message"]:
            try:
                cost = float(events["data"][1]["message"][-4:])
            except:
                cost = 0
            break

        time.sleep(60)

    retrieve_response = openai.FineTune.retrieve(fine_tune_job.id)
    fine_tuned_model = retrieve_response.fine_tuned_model

    validation_df = pd.read_json(path_or_buf="data/05_model_input/2022_prepared_valid.jsonl", lines=True)

    def get_classification(prompt):
        answer = openai.Completion.create(
            model=fine_tuned_model,
            prompt=prompt,
            max_tokens=1,
            temperature=0,
            logprobs=2,
        )
        return answer['choices'][0]['text']

    validation_df['classification'] = validation_df['prompt'].apply(get_classification)
    print(classification_report(validation_df['completion'], validation_df['classification']))

    wandb.log({"accuracy": accuracy_score(validation_df['completion'], validation_df['classification'])})
    wandb.log(
        {"precision": precision_score(validation_df['completion'], validation_df['classification'], average='macro')})
    wandb.log({"recall": recall_score(validation_df['completion'], validation_df['classification'], average='macro')})
    wandb.log({"f1": f1_score(validation_df['completion'], validation_df['classification'], average='macro')})
    wandb.log({"training_examples": len(
        pd.read_json(path_or_buf="data/05_model_input/2022_prepared_train.jsonl", lines=True))})
    wandb.log({"model": params["model"]})
    wandb.log({"model_id": str(fine_tuned_model)})
    wandb.log({"cost": cost})

    wandb.finish()

    # delete file used for training and validation
    os.remove("data/05_model_input/2022_prepared_train.jsonl")
    os.remove("data/05_model_input/2022_prepared_valid.jsonl")




