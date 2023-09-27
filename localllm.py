import re
import warnings
from typing import List

import torch
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline
from langchain.schema import BaseOutputParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)


class StopGenerationCriteria(StoppingCriteria):
    def __init__(
        self, tokens: List[List[str]], tokenizer: AutoTokenizer, device: torch.device
    ):
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        self.stop_token_ids = [
            torch.tensor(x, dtype=torch.long, device=device) for x in stop_token_ids
        ]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False


class CleanupOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        user_pattern = r"\nUser"
        text = re.sub(user_pattern, "", text)
        human_pattern = r"\nHuman:"
        text = re.sub(human_pattern, "", text)
        ai_pattern = r"\nAI:"
        return re.sub(ai_pattern, "", text).strip()

    @property
    def _type(self) -> str:
        return "output_parser"


warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "tiiuae/falcon-7b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, load_in_8bit=True, device_map="cuda:0"
)
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

generation_config = model.generation_config
generation_config.temperature = 0
generation_config.num_return_sequences = 1
generation_config.max_new_tokens = 256
generation_config.use_cache = False
generation_config.repetition_penalty = 1.7
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

template = """
The following is a friendly conversation between a human and an AI. The AI is
talkative and provides lots of specific details from its context.
 
Current conversation:
{history}
 
Human: {input}
AI:
""".strip()

# input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# input_ids = input_ids.to(model.device)

# with torch.inference_mode():
#     outputs = model.generate(
#         input_ids=input_ids,
#         generation_config=generation_config,
#     )

# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(response)

stop_tokens = [["Human", ":"], ["AI", ":"]]
stopping_criteria = StoppingCriteriaList(
    [StopGenerationCriteria(stop_tokens, tokenizer, model.device)]
)

generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task="text-generation",
    stopping_criteria=stopping_criteria,
    generation_config=generation_config,
)

llm = HuggingFacePipeline(pipeline=generation_pipeline)

prompt = PromptTemplate(input_variables=["history", "input"], template=template)
memory = ConversationBufferWindowMemory(memory_key="history", k=6, return_messages=True)
chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    output_parser=CleanupOutputParser(),
    verbose=True,
)

text = "What is the capital of Canada?"
res = chain(text)
print(res["response"])
