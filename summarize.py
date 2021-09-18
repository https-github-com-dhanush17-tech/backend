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

get_summary("A neutron star is the collapsed core of a massive supergiant star, which had a total mass of between 10 and 25 solar masses, possibly more if the star was especially metal-rich.[1] Neutron stars are the smallest and densest stellar objects, excluding black holes and hypothetical white holes, quark stars, and strange stars.[2] Neutron stars have a radius on the order of 10 kilometres (6.2 mi) and a mass of about 1.4 solar masses.[3] They result from the supernova explosion of a massive star, combined with gravitational collapse, that compresses the core past white dwarf star density to that of atomic nuclei.")