from langchain import LlamaCpp, PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# template = """Below is the chat history.
# If you can't reply the answer in a word similarily, reply 'None'.
# I want you to answer the correct answer in the following text in a word.
# 'A: What is your zip code?
#  B:No, why?'"""

# prompt = PromptTemplate.from_template(template)


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
local_llm = LlamaCpp(
    model_path="./wizardLM-7B.ggmlv3.q8_0.bin",
    callback_manager=callback_manager,
    verbose=True,
)

summarize_template_string = """
        Provide a summary for the following text:
        {text}
"""

summarize_prompt = PromptTemplate(
    template=summarize_template_string,
    input_variables=["text"],
)

summarize_chain = LLMChain(
    llm=local_llm,
    prompt=summarize_prompt,
)

text = input("Input : ")
summary = summarize_chain.run(text=text)
print(summary)
