import openai
import json


def get_summary(paragraph: str):
    with open('config.json') as json_file:
        data = json.load(json_file)
    openai.api_key = data['openai']
    paragraph = paragraph.strip("\n")
    return openai.Completion.create(
        engine="davinci",
        prompt=paragraph.__add__("\ntl;dr:"),
        temperature=0.3,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )["choices"][0]["text"]


def summary_to_sentences(summary: str):
    summary = summary.rstrip("\n")
    sentence_arr = []
    j = 0
    for i in range(len(summary)):
        if summary[i] == '.':
            sentence_arr.append(summary[j:i])
            j = i + 2
    return sentence_arr
