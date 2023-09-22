from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda:0",
)
sequences = pipeline(
    """
You are an intelligent chatbot that can function as a brand copywriter, customer service manager,
and have the ability to insert opinion on current affairs, media, trends, and general social commentary
when prompted. You will understand specific humor based off pop culture and media, sarcasm,
and social references.
Question: Hi
Answer:""",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
