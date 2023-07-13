from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# EOS token as per GPT style
eos_token = tokenizer.eos_token

# Input sequence
input_sequence = "Hello, how are you?"

# Encode the input sequence
input_sequence = tokenizer.encode(input_sequence, return_tensors="pt")

# Generate a sequence of tokens
generated_sequence = model.generate(
    input_sequence, max_length=100, pad_token_id=tokenizer.eos_token_id
)

# Decode the sequence
resulting_string = tokenizer.decode(generated_sequence.tolist()[0])

print(resulting_string)
