import torch

from llama_cpp import Llama

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

print(torch.cuda.is_available())

llm = Llama(model_path="./llama-2-7b-chat.ggmlv3.q2_K.bin")

app = Flask(__name__)
CORS(app)


@app.route("/api/query", methods=["POST"])
def query():
    template = """
    You are an intelligent chatbot that can function as a brand copywriter, customer service manager,
    and have the ability to insert opinion on current affairs, media, trends, and general social commentary
    when prompted. You will understand specific humor based off pop culture and media, sarcasm,
    and social references.
    Q: """

    query_data = request.get_json()
    query = query_data["query"]
    model_type = query_data["model_type"]

    prompt = template + query + "A: "

    response = llm(
        prompt,
        max_tokens=200,
        stop=["Q:", "\n"],
        echo=True,
    )
    answer_point = response["choices"][0]["text"].find("A: ")
    sub_answer = response["choices"][0]["text"][answer_point:]
    return {"message": sub_answer}


if __name__ == "__main__":
    app.run("0.0.0.0", 8000)
