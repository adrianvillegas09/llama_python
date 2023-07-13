from llama_cpp import Llama

llm = Llama(model_path="./wizardLM-7B.ggmlv3.q8_0.bin")
output = llm(
    """Below is the chat history. 
I want you to answer the correct answer in the above content in a word. If you can't reply the answer in a word similarily, reply 'None'.
'A: What is your zip code?
 B:No, why?'""",
    max_tokens=64,
    stop=["Q:", "\n"],
    echo=True,
)
print(output)
