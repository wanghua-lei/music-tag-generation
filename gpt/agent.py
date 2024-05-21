import re
import openai
import random
import numpy as np
import glv
import os


class AgentRequriement():

    def __init__(self,gpt_model=None, openai_key="",log_dir=None,iter=0) -> None:
        openai.api_key = openai_key
        self.model = gpt_model
        self.init_prompt = """"""
        self.constraints = [
        ]
        self.prefix = "###requirement###"
        self.seed_examples = [
        ]
        self.log_dir = log_dir
        self.iter = iter

    def load_seed(self,path):
        data = np.genfromtxt(path,dtype=str,delimiter='\n')
        self.seed_examples = data
        return data
    
    def load_constraints(self,path):
        data = np.genfromtxt(path,dtype=str,delimiter='\n')
        self.constraints = data
        return data
    
    def set_seed(self,seed):
        self.seed_examples = seed
    
    def set_constraints(self, req_constraints):
        self.constraints = req_constraints
    
    def construct_prompt(self):
        prompt = self.init_prompt
        for i, constraint in enumerate(self.constraints):
            prompt = f"{prompt}\n{i+1}. {constraint}"
        prompt = f"{prompt}\nHere are some examples you can refer to:"
        index = random.sample(list(range(len(self.seed_examples))),5)
        for i in index:
            prompt = f"{prompt}\n{self.prefix}{self.seed_examples[i]}"
        return prompt
    
    def parse_results(self,results):
        pattern = f"###[^#]*###[^#]*"
        matches = re.findall(pattern, results, re.DOTALL)
        res = []
        for match in matches:
            s = match
            for m in re.findall("###[^#]*###", match, re.DOTALL):
                s = s.replace(m,"")
            res.append(s.strip())
        return res
    
    def run(self, temperature=0, seed=0):
        prompt = self.construct_prompt()
        try: 
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature
            )
            if self.log_dir is not None:
                with open(os.path.join(self.log_dir,f"requirements_{self.iter}.txt"),'w') as f:
                    f.write(str(response)+'\n')
                self.iter+=1
            res = response.choices[0].message.content
            return self.parse_results(res)
        except openai.error.InvalidRequestError as e:
            print('except', e)
            raise ValueError
   
    def test_one(self, temperature=0):
        prompt = self.init_prompt
        for i in [0,1,2]:
            prompt = f"{prompt}\n{self.prefix}{self.seed_examples[i]}"
        try: 
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature
            )
            print(response)
            res = response.choices[0].message.content
            return self.parse_results(res)
        except openai.error.InvalidRequestError as e: # type: ignore
            print('except', e)
        
    



if __name__=='__main__':
    agent = AgentRequriement(gpt_model=glv.GPT_MODEL,openai_key=glv.OPENAI_KEY)
    agent.load_seed('/home/SENSETIME/yangzekang/LLMaC/LLMaC/data/requirements/seed_examples.txt')
    agent.load_constraints('/home/SENSETIME/yangzekang/LLMaC/LLMaC/data/constraints-v2.txt')
    prompt = agent.construct_prompt()
    print(prompt)