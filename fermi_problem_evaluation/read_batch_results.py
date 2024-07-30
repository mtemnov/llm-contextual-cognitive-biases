from openai import OpenAI
from dotenv import load_dotenv
import json
import argparse

import pandas as pd

import sys
import os

parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_folder_path)
from utils import *

parser = argparse.ArgumentParser(prog="ReadBatch", description="Retrieve Results from batch calls")
parser.add_argument("--batch_file", help="Filename of the OpenAI batch to load the results from")

def create_row(id, answer):
    data = id.split("/")
    new_row = {
        "target_id": data[0],
        "sample": data[1],
        "model": model,
        "context": data[2],
        "context_prompt": data[3],
        "context_bias": data[4],
        "target_prompt": data[5],
        "experiment_type": data[6],
        "response": answer
    }

    return new_row

if __name__ == "__main__":
    client = OpenAI()
    args = parser.parse_args()

    batch = client.batches.retrieve(args.batch_file)

    description = batch.metadata["description"]
    metadata = description.split("/")

    print(metadata)

    #model = "gpt-3.5-turbo"
    #target_prompt = "assumptions"
    #bias = "single_turn"
    model = metadata[0]
    target_prompt = metadata[1]
    bias = metadata[2]

    bin = client.files.content(batch.output_file_id)
    results_str = bin.content.decode("ascii")

    lines = results_str.strip().splitlines()
    
    results = [json.loads(el) for el in lines]

    new_data = []

    for item in results:
        answer = item["response"]["body"]["choices"][0]["message"]["content"]
        id = item["custom_id"]

        new_data.append(create_row(id, answer))
        #print(item["response"]["body"]["choices"][0]["message"]["content"])

    result_folder = "./results/{}/{}/".format(model, target_prompt)
    #result_folder = "./results/{}/".format(model)
    os.makedirs(result_folder, exist_ok=True)
    file = "{}_target_responses.csv".format(bias) 
    #file = "simple_responses.csv"

    filepath = result_folder + file
    if os.path.exists(filepath):
        existing_data = pd.read_csv(filepath)
        new_data = pd.concat([existing_data, pd.DataFrame(new_data)], ignore_index=True)
        new_data.sort_values(["context", "context_bias", "target_id", "experiment_type"], inplace=True)
    else:
        new_data = pd.DataFrame(new_data)

    new_data.to_csv(filepath, index=False)

    