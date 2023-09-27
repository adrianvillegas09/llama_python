import re
import warnings
from typing import List

import torch
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline
from langchain.schema import BaseOutputParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "tiiuae/falcon-7b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, load_in_8bit=True, device_map="cuda:0"
)
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

generation_config = model.generation_config
generation_config.temperature = 0
generation_config.num_return_sequences = 1
generation_config.max_new_tokens = 256
generation_config.use_cache = False
generation_config.repetition_penalty = 1.7
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

prompt = """
The following is a friendly conversation between a human and an AI. The AI is
talkative and provides lots of specific details from its context.
 
Current conversation:
 
Human: Who is Dwight K Schrute?
AI:
""".strip()

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
