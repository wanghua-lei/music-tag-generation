import os
import openai
import argparse
import json
import argparse
import random
from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from openai import OpenAI

def api_helper(instance):

    inputs = instance['inputs']
    # completion = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #             {"role": "user", "content": inputs}
    #         ]
    # )
    # results = completion['choices'][0]['message']['content']

    # client = OpenAI(
    #     api_key = "",
    #     base_url = "https://api.moonshot.cn/v1",
    # )

    client = OpenAI(
        api_key = "",
        base_url = "https://api.deepseek.com",
    )
    try:
        response = client.chat.completions.create(
                model="deepseek-chat", #moonshot-v1-32k
                messages=[
                        {"role": "user", "content": inputs},
                ],
        temperature=0.3,
        stream=False,
        )
        results = response.choices[0].message.content
        
        # print("query: ")
        # print(inputs)
        print("---"*10)
        print(results)
        with open("temp.txt", 'a') as file:
            file.write(results)
            file.write('\n')
        return results
    except openai.RateLimitError as e:
        print(f"Rate limit reached: {e}. Retrying in {10} seconds...")
        sleep(10)
    except Exception as e:
        print(f"An error occurred: {e}")
    


class OpenAIGpt:
    def __init__(self, dataset_type):
        # load_dotenv()   
        self.dataset_type = dataset_type
        if self.dataset_type == "MTG-Jamendo":
            self.data= json.load(open("json_files/mtg_tags.json", 'r'))
        self.prompt = "I will give you the tags of musics. Extract the type of the music and generate an music caption describing the genre, moods, and instruments for music tracks. The music caption should be less than 25 words. Increased richness and granularity in music description. Do not write introductions or explanations. Make sure you are using grammatical subject-verb-object sentences. Write an one-sentence music caption to describe it." 


    def run(self):
        inputs = []
        
        if len(self.data) > 0:
            for _id, instance in enumerate(self.data):
                instance['_id'] = _id

                instance['dataset_type'] = self.dataset_type
                instance["duration"] = instance['duration']
                instance["location"] = instance['location']

                text = instance['tags']
                instance["tags"] = instance['tags']

                if len(instance["tags"]):
                    instruction = self.prompt
                else:
                    continue
                instance["inputs"] = f'{instruction} \n {text}'

                answer = api_helper(instance)
                instance['caption'] = answer
                sleep(3)
                inputs.append(instance)

            # with ThreadPoolExecutor() as pool:
            #     results = list(tqdm(pool.map(api_helper, inputs), total=len(inputs)))

            with open('tag_cap.json', 'w') as f:
                json.dump(inputs, f, indent=4)
            print("finish")
        else:
            print("already finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="MTG-Jamendo", type=str)
    # parser.add_argument("--split", default="TRAIN", type=str)
    # parser.add_argument("--prompt", default="writing", type=str)
    args = parser.parse_args()

    openai_gpt = OpenAIGpt(
        dataset_type = args.dataset_type,
        )
    openai_gpt.run()