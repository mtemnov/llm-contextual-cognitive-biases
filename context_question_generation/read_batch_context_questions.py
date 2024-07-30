from openai import OpenAI
from dotenv import load_dotenv
import json

import pandas as pd

import sys
import os

import argparse

parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_folder_path)
from utils import *

parser = argparse.ArgumentParser(prog="ReadBatch", description="Retrieve Results from batch calls")
parser.add_argument("--batch_file", help="Filename of the OpenAI batch to load the results from")
parser.add_argument("--save_file", help="Filename where the data should be saved")

def create_row(id, answer):
    data = id.split("/")
    new_row = {
        "target_id": data[0],
        "source": data[1],
        "bias": data[2],
        "experiment_type": data[3],
        "question": answer
    }
    
    return new_row

if __name__ == "__main__":
    client = OpenAI()

    args = parser.parse_args()

    bin = client.files.content(args.batch_file)
    results_str = bin.content.decode("ascii")

    lines = results_str.strip().splitlines()
    
    results = [json.loads(el) for el in lines]

    new_data = []

    for item in results:
        answer = item["response"]["body"]["choices"][0]["message"]["content"]
        id = item["custom_id"]

        new_data.append(create_row(id, answer))

    result_folder = "./results/"
    os.makedirs(result_folder, exist_ok=True)

    filepath = result_folder + args.file + ".csv"
    if os.path.exists(filepath):
        existing_data = pd.read_csv(filepath)
        new_data = pd.concat([existing_data, pd.DataFrame(new_data)], ignore_index=True)
    else:
        new_data = pd.DataFrame(new_data)

    new_data.to_csv(filepath, index=False)

    