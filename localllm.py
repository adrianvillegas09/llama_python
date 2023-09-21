import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

timeStart = time.time()

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")

model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b-instruct",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

print("Load model time: ", -timeStart + time.time())

while True:
    input_str = input("Enter: ")
    input_token_length = input("Enter length: ")

    if input_str == "exit":
        break

    timeStart = time.time()

    inputs = tokenizer.encode(input_str, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_new_tokens=int(input_token_length),
    )

    output_str = tokenizer.decode(outputs[0])

    print(output_str)

    print("Time taken: ", -timeStart + time.time())
