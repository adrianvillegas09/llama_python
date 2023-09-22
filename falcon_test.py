from transformers import AutoTokenizer
import transformers
import torch
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

print(torch.cuda.is_available())

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
torch.cuda.is_available()
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda:0",
)
llm = HuggingFacePipeline(pipeline=pipeline)

template = """
You are an intelligent chatbot that can function as a brand copywriter, customer service manager,
and have the ability to insert opinion on current affairs, media, trends, and general social commentary
when prompted. You will understand specific humor based off pop culture and media, sarcasm,
and social references.
Question: {query}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["query"])

# chain
llm_chain = LLMChain(prompt=prompt, llm=llm)

query = "Hello"

print(llm_chain.run(query))
