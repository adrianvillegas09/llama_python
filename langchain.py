from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="./wizardLM-7B.ggmlv3.q8_0.bin",
    callback_manager=callback_manager,
    verbose=True,
)

template = """You support me in identifying gratitude in my life. 
You help me find gratitude. Your language is simple, clear, 
and you are enthusiastic, compassionate, and caring. 
Your responses are short and one or two lines.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

chatgpt_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False,
    memory=ConversationBufferWindowMemory(k=2),
)

print("GratitudeLLM loading...")
output = chatgpt_chain.predict(human_input="Hello")

while True:
    human_input = input("\nHuman: ")
    output = chatgpt_chain.predict(human_input=human_input)
