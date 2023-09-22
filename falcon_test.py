from transformers import AutoTokenizer
import transformers
import torch
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

print(torch.cuda.is_available())

model = "tiiuae/falcon-7b-instruct"
model_llama = "NousResearch/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    max_length=700,
    temperature=0,
    trust_remote_code=True,
    device_map="cuda:0",
)
llm = HuggingFacePipeline(pipeline=pipeline)

# tokenizer_llama = AutoTokenizer.from_pretrained(model_llama)
# pipeline_llama = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     max_length=700,
#     temperature=0,
#     trust_remote_code=True,
#     device_map="cuda:0",
# )
# llm_llama = HuggingFacePipeline(pipeline=pipeline_llama)

app = Flask(__name__)
CORS(app)


@app.route("/api/query", methods=["POST"])
def query():
    template = """
    You are an intelligent chatbot that can function as a brand copywriter, customer service manager,
    and have the ability to insert opinion on current affairs, media, trends, and general social commentary
    when prompted. You will understand specific humor based off pop culture and media, sarcasm,
    and social references.
    Question: {query}
    Answer:"""

    query_data = request.get_json()
    query = query_data["query"]
    model_type = query_data["model_type"]

    prompt = PromptTemplate(template=template, input_variables=["query"])

    # chain
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    response = llm_chain.run(query)
    return {"message": response}


if __name__ == "__main__":
    app.run("0.0.0.0", 8000)
