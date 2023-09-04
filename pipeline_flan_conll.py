import numpy as np
import pandas as pd
import logging
import os
import glob
import regex as re
import torch
import argparse
import random
import json
import itertools
import ast
import sys
import ast
from tqdm import tqdm
import warnings
import pickle as pk
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline

warnings.filterwarnings("ignore")

with open("conll_modified_prompt.txt", "r") as text_file:
    prompt = text_file.read()
   
conll = json.load(open('conll04_dev.json'))

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl", 
                                              cache_dir="/scratch/wadhwa.s/cache", 
                                              device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl", 
                                          cache_dir="/scratch/wadhwa.s/cache")

generator = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer)

dev_conll = []
for i in tqdm(conll):

    p =  prompt + "TEXT: " + ' '.join(i["tokens"]) + "\nRelation List: "
    dev_conll.append(p)
#     print (p)
#     print ("\n--------------\n")

ip = []
response = []

for ix, instance in enumerate(tqdm(dev_conll)):
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
df.to_csv("conll_flan_fewshot_results.csv", index=False)