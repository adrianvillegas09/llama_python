from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

print(torch.cuda.is_available())

model = "tiiuae/falcon-7b-instruct"

rrmodel = AutoModelForCausalLM.from_pretrained(
    model,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model)


input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

attention_mask = torch.ones(input_ids.shape)

output = rrmodel.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
