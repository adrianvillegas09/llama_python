from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model (weights)
model = AutoModelForCausalLM.from_pretrained("AlekseyKorshuk/vicuna-7b")

# Load pre-trained model tokenizer
tokenizer = AutoTokenizer.from_pretrained("AlekseyKorshuk/vicuna-7b")

# Make sure all model parameters do not require gradients
for param in model.parameters():
    param.requires_grad = False

# Conversation pipeline
conversational_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

while True:
    # Get user input
    user_input = input("User: ")

    # Generate response
    model_input = tokenizer.encode(
        user_input + tokenizer.eos_token, return_tensors="pt"
    )
    response_tokenized = model.generate(
        model_input, max_length=1000, pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(
        response_tokenized[:, model_input.shape[-1] :][0], skip_special_tokens=True
    )

    print("Bot: ", response)
