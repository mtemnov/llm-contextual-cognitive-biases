from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
import json
import pickle
import time
import os
import pandas as pd
from openai import OpenAI

class APIClient:
    def __init__(self, args):
        self.args = args
        if "gpt" in args.model:
            self.client = OpenAI()
        elif "llama" in args.model:
            self.args.model = "meta-llama/" + args.model
            self.client = OpenAI(
                api_key=os.environ.get("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1",
            )
            self.args.as_batch = False
    
    def return_client_args(self):
        return self, self.args
    
    def get_client(self):
        return self.client
    
    def construct_prompt(self, question, prompt, history=None):
        if prompt.any() != None:
            items = [el for el in [prompt["prefix"], question, prompt["postfix"]] if el and pd.notna(el)]
            input = {
                "role": "user",
                "content": "\n".join(items),
            }
        else:
            input = {"role": "user", "content": question}
    
        if history:
            history.append(input)
        #elif "role" in prompt:
            #history = [{"role": "system", "content": prompt["role"]}, input]
        else:
            history = [input]
    
        return history
    
    def expand_history(self, history, response):
        new_message = {"role": "assistant", "content": response}
        history.append(new_message)
        return history

    def api_call(self, messages):
        if self.args.debug:
            mocked_response = "Mock response"
            choice = Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(content=mocked_response, role="assistant"))
            completion = ChatCompletion(id="test", model="gpt-3.5-turbo", object="chat.completion", choices=[choice], created=int(time.time()))
        else:
            completion = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=messages,
                    temperature=0.8,
                    max_tokens=512,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
            )
                
        return completion
    
    
    def get_completion_message(self, completion):
        return completion.choices[0].message.content

def batch_call(model, messages, id):
    return {
        "custom_id": id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "temperature": float(0.8),
            "max_tokens": 512,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
    }


def save_results(data, filename, as_json=True):
    directory = os.path.dirname(filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    if as_json:
        with open(filename + ".json", "w") as file:
            json.dump(data, file, indent=4)
            file.close()
    else:
        with open(filename + ".pkl", "wb") as file:
            pickle.dump(data, file)
            file.close()
