from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="./wizardLM-7B.ggmlv3.q8_0.bin",
    # callback_manager=callback_manager,
    verbose=True,
)

template = """Below is the chat history. 
I want you to answer the correct answer in the above content in a word. If you can't reply the answer in a word similarily, reply 'None'.
'A: What is your zip code?
 B:No, why?'"""

prompt = PromptTemplate.from_template(template)

chatgpt_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False,
    memory=ConversationBufferWindowMemory(k=2),
)

print("GratitudeLLM loading...")
output = chatgpt_chain.predict()

while True:
    # human_input = input("\nHuman: ")
    output = chatgpt_chain.predict()
    print(output)
