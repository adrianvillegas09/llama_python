from llama_cpp import Llama

llm = Llama(model_path="./model.bin")
inputdata = ""
json_fields = ["zip_code", "work_tech", "company"]
questions = [
    {
        "main": "What's your zip code?",
        "addin_prompt": "",
    },
    {
        "main": "Do you work in tech?",
        "addin_prompt": "If answer is like yes, answer 'True', and if answer is like no, answer 'False'",
    },
    {
        "main": "Which company did you last work for?",
        "addin_prompt": "",
    },
]
i = 0

while True:
    i = 0
    while i < 3:
        print(questions[i]["main"] + "?")
        answer = input("Answer : ")
        output = llm(
            "Q: 'person1: "
            + questions[i]["main"]
            + "\n person2: "
            + answer
            + "' \n "
            + "What is the person2's answer from above content in one word chunk? Don't answer with long. If you cannot find the answer correctly, answer 'NULL'. "
            + questions[i]["addin_prompt"]
            + " A: ",
            max_tokens=100,
            # stop=["Q:", "\n"],
            echo=True,
        )

        answer_point = output["choices"][0]["text"].find("A: ")
        sub_answer = output["choices"][0]["text"][answer_point:]

        print(sub_answer.find("NULL"))

        print(sub_answer)
        i = i + 1

    # What is the zip code in Person B's answer in a word?
