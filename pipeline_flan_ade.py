import numpy as np
import pandas as pd
import logging
import os
import glob
import regex as re
import torch
import argparse
import random
import itertools
import ast
import sys
import ast
from tqdm import tqdm
import warnings

from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline

warnings.filterwarnings("ignore")

with open("ade_prompt.txt", "r") as text_file:
    prompt = text_file.read()
    
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl", 
                                              cache_dir="/scratch/wadhwa.s/cache", 
                                              device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl", 
                                          cache_dir="/scratch/wadhwa.s/cache")


generator = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer)

input_docs = "ade_gpt3.txt"

with open(input_docs) as f:
    lines = f.readlines()
    unique_ade = [line.strip() for line in lines]
    
flan_ade = [prompt + s + "Relation List: " for s in lines]

dev_flan_ade = random.sample(flan_ade, int(0.2*len(flan_ade)))

ip = []
response = []

for ix, instance in enumerate(tqdm(dev_flan_ade)):
    try:
        res = generator(instance, 
              max_length=200)
        torch.cuda.empty_cache()
        ip.append(instance.split("\n")[-2])
        response.append(res[0]["generated_text"])
    except:
        torch.cuda.empty_cache()
        pass
   

df = pd.DataFrame({'input': ip, 'response': response}, index=None)
df.to_csv("ade_flan_fewshot_results.csv", index=False)