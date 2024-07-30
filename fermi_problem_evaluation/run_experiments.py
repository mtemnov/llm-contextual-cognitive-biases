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
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(
    prog="ThesisExperiments",
    description="Run API calls to ChatGPT for provided questions",
)

parser.add_argument("--model", default="gpt-3.5-turbo", help="Model tag which is used for the API requests")

parser.add_argument(
    "--context_prompt",
    default="simple",
    help="Select the prompt style for the context question",
)

parser.add_argument(
    "--target_prompt",
    default="onlyanswer",
    help="Select the prompt style for the target question",
)

parser.add_argument("--experiment_type", default=None, help="Select the experiment_type style (one existing experiment_type that is available in context_questions.csv)")

parser.add_argument("--context", default=None, help="Select the source of the context question (one existing source in the cotext_questions.csv)")

parser.add_argument("--bias", default="general", help="Select the bias that the context target")

parser.add_argument("--samples", default=10, help="Number of API calls per question")

parser.add_argument("--save_target", default="target_responses.csv", help="Filename to save the target responses (if existing, appends new data)")

parser.add_argument("--save_context", default="responses.csv", help="Filename to save the context responses (if existing, appends new data)")

parser.add_argument("--as_batch", default=False, action="store_true", help="Instead of using synchronous API calls, create a batch that can be uploaded to the OpenAI API")

parser.add_argument("--context_unavailable", default=False, action="store_true", help="If context is available, it is loaded from the file instead of being generated again")

parser.add_argument(
    "--debug",
    dest="debug",
    default=False,
    action="store_true",
    help="Toggle if experiments should be run with mock LLM responses",
)

def format_response(row, idx, args, content, target = False):
    new_row = {
        "target_id": row["id"],
        "sample": idx,
        "model": args.model,
        "context": args.context if args.context else "single_turn",
        "context_prompt": args.context_prompt,
        "context_bias": args.bias if args.context else "none",
        **( {"target_prompt": args.target_prompt} if target else {} ),
        "experiment_type": args.experiment_type if args.context else "neutral",
        "response": content,
    }

    return new_row

def format_batch_id(row, idx, args, target = False):
    return "/".join([str(row["id"]), str(idx), args.context if args.context else "single_turn", args.context_prompt, args.bias if args.context else "none", args.target_prompt, args.experiment_type if args.context else "neutral"])

def run_api_requests(args):
    folder = "./experiment_data/"
    target_questions = pd.read_csv(folder + "target_questions.csv")
    prompts = pd.read_csv(folder + "prompts.csv")
    prompts = prompts.replace(np.nan, None)
    context_questions = pd.read_csv(folder + "generated_context_questions.csv")

    original_target_prompt = prompts[(prompts["turn"] == "target") & (prompts["prompt"] == args.target_prompt)].iloc[0]
    context_prompt = prompts[(prompts["turn"] == "context") & (prompts["prompt"] == args.context_prompt)].iloc[0]

    load_dotenv()

    client, args = APIClient(args).return_client_args()

    context_responses = []
    target_responses = []

    if args.as_batch:
        batch_calls = []

    if not args.context_unavailable and args.context:
        filepath = "./results/{}/{}_{}".format(args.model, args.context_prompt, args.save_context)
        context_answers = pd.read_csv(filepath)

    exp_types = []
    if not args.experiment_type and args.context:
        exp_types = ["decrease", "increase"]
    elif not args.context:
        exp_types = ["neutral"]
    elif args.experiment_type not in ["decrease", "increase", "neutral"] and exp_types == []:
        raise Exception("Experiment type not supported")
    else:
        exp_types = [args.experiment_type]

    for idx, row in target_questions.iterrows():
        for exp_type in exp_types:
            print("Sampling question", row["id"], exp_type)

            args.experiment_type = exp_type
            messages = None

            if args.context:
                context_condition = (context_questions["target_id"] == row["id"]) & (context_questions["source"] == args.context) & (context_questions["bias"] == args.bias) & (context_questions["experiment_type"] == args.experiment_type)
                test = (context_questions["target_id"] == row["id"]) & (context_questions["source"] == args.context) & (context_questions["bias"] == args.bias) & (context_questions["experiment_type"] == args.experiment_type)
                context_question = context_questions[context_condition].iloc[0]
                messages = client.construct_prompt(context_question["question"], context_prompt)
                if args.context_unavailable:
                    if args.as_batch:
                        id = format_batch_id(row, 0, args)
                        call = batch_call(args.model, messages, id)
                        batch_calls.append(call)
                        continue
                    else:
                        completion = client.api_call(messages)
                        messages = client.expand_history(messages, client.get_completion_message(completion))
                        new_row = format_response(row, 0, args, client.get_completion_message(completion))
                        context_responses.append(new_row)
                else:
                    search = (context_answers["target_id"] == row["id"]) & (context_answers["context_bias"] == args.bias) & (context_answers["context"] == args.context) & (context_answers["context_prompt"] == args.context_prompt) & (context_answers["experiment_type"] == args.experiment_type)
                    completion = context_answers[search].iloc[0]["response"]
                    messages = client.expand_history(messages, completion)

            target_prompt = original_target_prompt.str.replace("[unit]", row["unit"])
            messages = client.construct_prompt(row["question"], target_prompt, messages)

            if args.as_batch:
                for sample in range(int(args.samples)):
                    #print(sample)
                    id = format_batch_id(row, sample, args, True)
                    call = batch_call(args.model, messages, id)
                    batch_calls.append(call)
            else:
                for sample in range(int(args.samples)):
                    print(sample)    
                    completion = client.api_call(messages)

                    if args.debug and sample == 3:
                        print(messages)
                    
                    new_row = format_response(row, sample, args, client.get_completion_message(completion), True)
                    target_responses.append(new_row)

    if args.as_batch:
        batch_folder = "./batches/{}/".format(args.model)
        os.makedirs(batch_folder, exist_ok=True)
        if len(exp_types) > 1: args.experiment_type = "all"
        if args.context_unavailable:
            filename = batch_folder + "{}_{}_context_gen_{}_batch.jsonl".format(args.bias, args.context, args.experiment_type)
        else:
            filename = batch_folder + "{}_{}_{}_context_{}_batch.jsonl".format(args.target_prompt, args.bias, args.context if args.context else "single_turn", args.experiment_type if args.context else "neutral")
        with open(filename, "w+") as f:
            for item in batch_calls:
                f.write(json.dumps(item) + "\n")

        if not args.debug:
            batch_input = client.client.files.create(
                file=open(filename, "rb"),
                purpose="batch"
            )

            batch_id = batch_input.id
            client.client.batches.create(
                input_file_id=batch_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                "description": "/".join([args.model, args.target_prompt, args.bias if args.context else "neutral", args.context if args.context else "single_turn"])
                }
            )

    if args.debug:
        debug_str = "DEBUG"
    else:
        debug_str = ""

    files = [args.target_prompt + "/" + debug_str + args.bias + "_" + args.save_target, debug_str + "{}_{}".format(args.context_prompt, args.save_context)]
    collected_data = [target_responses, context_responses]

    result_folder = "./results/{}/".format(args.model)
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(result_folder + args.target_prompt, exist_ok=True)

    for idx, file in enumerate(files):
        new_data = collected_data[idx]
        if new_data != []:
            filepath = result_folder + file
            if os.path.exists(filepath):
                existing_data = pd.read_csv(filepath)
                new_data = pd.concat([existing_data, pd.DataFrame(new_data)], ignore_index=True)
                new_data.sort_values(["target_id", "context", "experiment_type"], inplace=True)
            else:
                new_data = pd.DataFrame(new_data)

            new_data.to_csv(filepath, index=False)

if __name__ == "__main__":
    print("START")
    args = parser.parse_args()
    run_api_requests(args)