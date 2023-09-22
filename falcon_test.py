from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import transformers
import torch

# from transformers import BitsAndBytesConfig

# create a pipeline for the Falcon model
from langchain import HuggingFacePipeline

# Langchain contains PromptTemplate - allows to alter answers from the llm
# LLMChain - chains the prompttemplate and LLM together

from langchain import PromptTemplate, LLMChain

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     llm_int8_enable_fp32_cpu_offload=True,
# )

# model_name = "tiiuae/falcon-7b-instruct"
model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    # device_map=device_map,
    max_length=200,
    use_cache=True,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

# call the huggingface pipeline() object & pass the pipeline and model parameters
# temperature of 0, makes the model not to hallucinate much (make its own answers)
llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={"temperature": 0})

# Define Template
# conversational

# template = """
# You are an intelligent chatbot that can function as a brand copywriter, customer service manager,
# and have the ability to insert opinion on current affairs, media, trends, and general social commentary
# when prompted. You will understand specific humor based off pop culture and media, sarcasm,
# and social references.
# Question: {query}
# Answer:"""
# prompt = PromptTemplate(template=template, input_variables=["query"])

# # chain
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# query = "How do i pay for a service at the market? Write me an approach for this"

# print(llm_chain.run(query))
