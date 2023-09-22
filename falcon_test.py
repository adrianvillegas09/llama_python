from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

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
    max_length=700,
    temperature=0,
    top_p=0.95,
    top_k=10,
    repetition_penalty=1.15,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id,
    device_map="auto",
)
llm = HuggingFacePipeline(pipeline=pipeline)

chain = LLMChain(llm=llm, prompt="Hi")

# Run the chain only specifying the input variable.
print(chain.run("colorful socks"))
