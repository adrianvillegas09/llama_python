from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain

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

conversation = ConversationChain(llm=llm)
print(
    conversation.run(
        "Translate this sentence from English to French: I love programming."
    )
)
