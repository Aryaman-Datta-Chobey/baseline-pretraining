import os
import pdb
import setuptools
import torch
#base libraries for deep learning and NLP , data wrangling 
import torch 
import transformers 
import re
import tokenizers
from tokenizers import (
    normalizers, #1
    Tokenizer, 
    pre_tokenizers, #2
    models, #3
    trainers,#3
    processors, #4
    decoders, #5
)
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer
#from transformers import GPT2Tokenizer


def get_gpt2_tokenizer_func(model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return tokenizer


def get_roberta_tokenizer_func(model_name="roberta-base"):
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    return tokenizer


def get_tokenizer_func(opt_model_size='125m'):
    model_name = f"facebook/opt-{opt_model_size}"
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=50272, special_tokens=["<|endoftext|>"])
    tokenizer.train(["/home/achobey/data/babyLM/processed/train.txt"], trainer=trainer)#HARDCODED FOR NOW
    #tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.add_bos_token = False
    tokenizer.add_special_tokens(
            {
                'bos_token': '<s>', 
                'unk_token': '<unk>',
                'additional_special_tokens': [
                    '<image>', '</c>', 
                    '<PERSON>', # C-12M for person names
                    ]
            })
    return tokenizer
