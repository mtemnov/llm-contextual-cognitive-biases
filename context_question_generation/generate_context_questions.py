import sys
import os

parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_folder_path)
from utils import *

from openai import OpenAI
from dotenv import load_dotenv
import argparse
import json
import time
import pickle
import pandas as pd
import csv

parser = argparse.ArgumentParser(prog="Thesisdatasets", description="Run API calls to ChatGPT to generate context questions")

parser.add_argument("--model", default="gpt-4-turbo", help="Model tag which is used for the API requests")

parser.add_argument("--bias", default="general", help="Select the generation style")

parser.add_argument("--experiment_type", default=None, help="Select the experiment_type style (one existing experiment_type that is available for the dataset in context_questions_prompting.csv)")

parser.add_argument("--save", default="generated_context_questions.csv", help="Filename to save the results")

parser.add_argument("--as_batch", default=False, action="store_true", help="Instead of using synchronous API calls, create a batch that can be uploaded to the OpenAI API")

parser.add_argument("--turn", default=0, help="Provide turn of conversation to generate context questions")

parser.add_argument(
    "--debug",
    dest="debug",
    default=False,
    action="store_true",
    help="Toggle if datasets should be run with mock LLM responses",
)

# Define the structure of ids for the batch calls   
def format_batch_id(row, args, target = False):
    return "/".join([str(row["id"]), args.model, args.bias, args.experiment_type])

def get_context_questions(args):
    # Load the data
    folder = "../fermi_problem_evaluation/experiment_data/"
    target_questions = pd.read_csv(folder + "target_questions.csv")
    prompts = pd.read_csv("context_questions_prompts.csv")
    context_questions = pd.read_csv(folder + "generated_context_questions.csv")

    # Get the prompt for the selected context bias
    condition = (prompts["bias"] == args.bias)
    prompts = prompts[condition]

    # Load the first turn responses if the context question is generated in two turns
    first_turn_file = "./results/{}_first_turn.csv".format(args.bias)
    if len(prompts) > 2 and args.turn == 0:
        args.save = first_turn_file
    elif args.turn == 1:
        first_turn = pd.read_csv(first_turn_file)

    if len(prompts) == 0:
        raise Exception("Prompt variant does not exist!")

    # Create the API client
    load_dotenv()
    client = APIClient(args)

    gen_questions = []
    batch_calls = []

    if not args.experiment_type:
        exp_types = ["decrease", "increase"]
    elif args.experiment_type not in ["decrease", "increase", "neutral"]:
        raise Exception("Experiment type not supported")
    else:
        exp_types = [args.experiment_type]

    # Generate the context questions
    for idx, row in target_questions.iterrows():
        for exp_type in exp_types:
            prompt = prompts.iloc[args.turn]
            prompt = prompt.str.replace("[experiment_type]", exp_type)
            print("getting", row["id"], exp_type)
            messages = None
            args.experiment_type = exp_type
            if args.turn == 1:
                context_condition = (first_turn["target_id"] == row["id"]) & (first_turn["source"] == args.model.split("/")[-1]) & (first_turn["experiment_type"] == args.experiment_type)
                context_info = first_turn[context_condition].iloc[0]["response"]
                first_turn_prompt = prompts.iloc[0].str.replace("[experiment_type]", exp_type)
                messages = client.construct_prompt(row["question"], first_turn_prompt)
                messages = client.expand_history(messages, context_info)
                messages = client.construct_prompt(None, prompt, messages)
            else:
                messages = client.construct_prompt(row["question"], prompt)

            if args.as_batch:
                id = format_batch_id(row, args, True)
                call = batch_call(args.model, messages, id)
                batch_calls.append(call)
            else:
                completion = client.api_call(messages)
                messages = client.expand_history(messages, client.get_completion_message(completion))
        
                new_row = {
                    "target_id": row["id"],
                    "source": args.model.split("/")[-1],
                    "bias": args.bias,
                    "experiment_type": args.experiment_type,
                    "question": client.get_completion_message(completion)
                }

                gen_questions.append(new_row)

    # Append new context questions to the existing data
    context_questions = pd.concat([context_questions, pd.DataFrame(gen_questions)], ignore_index=True)

    if args.debug:
        debug_str = "DEBUG"
    else:
        debug_str = ""

    context_questions.sort_values(["target_id", "source", "bias", "experiment_type"], inplace=True)

    # If the context questions are submitted as a batch, write the batch to a file and upload it to the OpenAI API
    if args.as_batch:
        batch_folder = "./batches/{}/".format(args.model)
        os.makedirs(batch_folder, exist_ok=True)
        filename = batch_folder + "{}_{}_{}_{}_batch.jsonl".format(args.model, args.bias, args.experiment_type if len(batch_calls) == 50 else "all", args.turn)
        with open(filename, "w+") as f:
            for item in batch_calls:
                f.write(json.dumps(item) + "\n")

        if not args.debug:
            batch_input = client.files.create(
                file=open(filename, "rb"),
                purpose="batch"
            )

            batch_id = batch_input.id
            client.batches.create(
                input_file_id=batch_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                "description": " ".join([args.model, "bias:", args.bias, "experiment_type:", args.experiment_type])
                }
            )

    # Save the generated context questions
    if gen_questions != []:
        context_questions.to_csv("../fermi_problem_evaluation/experiment_data/{}{}".format(debug_str, filename), index=False)


if __name__ == "__main__":
    print("START")
    args = parser.parse_args()
    args.turn = int(args.turn)
    get_context_questions(args)