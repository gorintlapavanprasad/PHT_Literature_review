### References ###
# https://realpython.com/intro-to-python-threading/

### Import libraries ###
import time
import threading
import logging
import json
import pathlib
import re
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage
import glob
import pandas as pd

### Global variables ###
# file directory template
# list of PDFs to review
papers = ["butler19.pdf", "black11.pdf", "keate94.pdf", "jack00.pdf",
        "brashear06.pdf", "kaminer14.pdf", "mohammed06.pdf", "kuwahara06.pdf", "jiang23.pdf",
        "gadiraju19.pdf", "madeo11.pdf", "luna23.pdf", "anderson96.pdf", "india19.pdf",
        "hossain22.pdf", "illijima22.pdf", "galbraith14.pdf", "boyd17.pdf", "karpodini22.pdf",
        "gennari08.pdf", "hornof03.pdf", "li22.pdf", "gotfrid16.pdf", "husaan11.pdf",
        "aruanno18.pdf", "bondioli17.pdf", "chibaudel20.pdf", "hurd19.pdf", "langford21.pdf",
        "gadiraju21.pdf", "gay20.pdf", "lu22.pdf", "biemanns09.pdf", "gleason20.pdf",
        "baloian02.pdf", "madjaroff17.pdf", "brule16.pdf", "ahmetovic21.pdf", "jiang22.pdf",
        "kamel02.pdf", "mok22.pdf", "alankus11.pdf", "hossain23.pdf", "jayant11.pdf",
        "kwon19.pdf", "alsaleem19.pdf", "mahmud23.pdf", "collins23.pdf", "lehman98.pdf",
        "carrington17.pdf", "lobo21.pdf", "milne13.pdf", "hoffmann15.pdf", "fell03.pdf",
        "gerling13.pdf", "khurana21.pdf"]
# list of models to prompt
models = ["litellm_proxy/anvilgpt/llama3.2:latest"]
# list of model base urls
# list of model APIs
# list of PHT framework questions (not using all of them at first to test code)
pht_framework = [        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) totally fun first (i.e., Entertainment game reappropriated), 
        (2) mostly fun first (i.e., design workshop with disabled stakeholders),
        (3) both fun and utility first (i.e., design workshop with diverse stakeholders), 
        (4) mostly utility first (i.e., primary utilitarian mechanic designed first),
        (5) totally utility first (i.e., made by clinicians or specialists, fun incorporated later)""",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) unstructured play (i.e., no rules, making art),
        (2) semi-structured play (i.e., playground activities),
        (3) flexible structure with rules (i.e., improv),
        (4) flexible game (i.e., board game with house rules),
        (5) game with rigid rules (i.e., video game)""",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) entirely skill-based (i.e., trivia, sports),
        (2) mostly skill-based (i.e., mario kart),
        (3) equally skill and chance-based (i.e., catan),
        (4) mostly chance-based (i.e., Uno),
        (5) entirely chance-based (i.e., all dice)""",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) entirely solo (i.e., solitaire),
        (2) mostly solo (i.e., playing against AI in single-player StarCraft),
        (3) mix of solo and social affordances (i.e., animal crossing),
        (4) mostly social, but aspects are independent (i.e., house on hill haunt transition),
        (5) entirely social (i.e., tag)""",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) entirely turn-based (i.e., tic tac toe),
        (2) follows a set of steps (i.e., viticulture's dynamic turn ordering),
        (3) turns are taken, but some actions can be taken at any time (i.e., pandemic),
        (4) most actions can be taken at any time, but there are some phases (i.e., PvP death reset timer),
        (5) entirely simultaneous (i.e., race)""",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) entirely synchronous (i.e., real time strategy game),
        (2) mostly synchronous with a few asynchronous affordances (i.e., league of legends),
        (3) equal mix of synchronous and asynchronous affordances (i.e., helldivers 2),
        (4) mostly asynchronous with few synchronous affordances (i.e. cookie clicker),
        (5) entirely asynchronous (i.e., chess by postage mail)""",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) entirely competitive (i.e., spit card game),
        (2) mostly competitive, but sometimes collaboration is important (i.e., forbidden island),
        (3) mix of competitive and collaborative (i.e., mario party 2 vs 2 mini-games),
        (4) mostly collaborative (i.e., animal crossing),
        (5) entirely collaborative (i.e., pandemic)""",
        
        """Please answer this question with a number--which of the following best apply to the design in the study: 
        (1) entirely symmetrical (i.e., tic tac toe),
        (2) mostly symmetrical (i.e., undertale's differing NPC behaviors based on previous playthroughs),
        (3) both symmetrical and asymmetrical (i.e., pandemic with its similar turn structure but different character abilities),
        (4) mostly asymmetrical (i.e., tag),
        (5) entirely asymmetrical (i.e., keep talking, nobody explodes)"""]

## LangChain wrapper class
class WrappedEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

embedding_model = WrappedEmbeddings(model_name="all-MiniLM-L6-v2")

# DF to store all results
model_annotations_df = pd.DataFrame(columns = ["Model", "PaperID", "Framework_Question", "Response", "Confidence", "Duration"])

### Prompting Functions ###

# Function Name:
# Inputs:
# Outputs:
# Dependencies:
# Description: Chunks data into pieces of a set size.
def chunk_data(text):
    """Chunk text for processing"""
    # increasing chunk size from 500 to 1000
    split_text = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_doc = split_text.create_documents([text])
    return chunked_doc

# Function Name:
# Inputs:
# Outputs:
# Dependencies:
# Description: Uses the FAISS from_documents function to create a vector store for chunked data. This stores the data in a way that the model can access.
def create_vec(chunked_data):
    vector_store = FAISS.from_documents(chunked_data, embedding_model)
    return vector_store

# Function Name: load_context
# Inputs: file_path, the directory path for a given paper PDF
# Outputs:
# Dependencies: pymupdf4llm's to_markdown function, write_bytes, and chunk_data
# Description: Loads the PDF for a file into markdown text, then chunking the data
def load_context(file_path):
    # convert the PDF to markdown text
    markdown_text = pymupdf4llm.to_markdown(file_path)
    # encode the PDF into bytes
    pathlib.Path("output.md").write_bytes(markdown_text.encode())
    # chunk the data
    chunked_data = (markdown_text)
    return chunked_data

# Function Name: format_prompt
# Inputs: The context from the paper PDF being examined and the framework question
# Outputs: Returns a single item list containing the entire prompt
# Dependencies: None
# Description: Takes in the context (the paper being annotated) and the question from the PHT framework, formulates a prompt to give to the models
def format_prompt(context_docs, question):
    prompt_parts = []
    prompt_parts.append("Your task is to read the following paper and answer questions about it. Please answer every question to the best of your ability.")
    prompt_parts.append(f"Paper Context: {context_docs}")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("\nProvide your response in the following format:")
    prompt_parts.append("Answer: <only the answer from the list of answer options here, no additional text>")
    prompt_parts.append("Reasoning: <your explanation here>")
    prompt_parts.append("Confidence: <only include a value for how confident you are that your answer is correct, giving a score between 0.0 and 1.0 here>")
    return "\n".join(prompt_parts)

# Function Name:
# Inputs:
# Outputs:
# Dependencies:
# Description: Formats prompt and LLM variable given speicified model, sends prompt to the model
def send_prompt(context, metric, paperID, model, api_key, api_base):
    llm = ChatLiteLLM(
    model=model,
    api_key=api_key,
    api_base=api_base)

    docs_content = "\n\n".join(doc.page_content for doc in context)
    
    # Build few-shot prompt
    prompt_text = format_prompt(docs_content, metric)
    
    response = llm.invoke(prompt_text).content.strip()

    answer_match = re.search(r"(?i)^Answer:\s*(.*)", response, re.MULTILINE)
    reasoning_match = re.search(r"(?i)^Reasoning:\s*(.*)", response, re.MULTILINE)
    confidence_match = re.search(r"(?i)^Confidence:\s*([\d.]+)", response, re.MULTILINE)

    answer = answer_match.group(1).strip() if answer_match else "0"
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Not found"
    confidence = confidence_match.group(1).strip() if confidence_match else "0.0"

    return {
        "paperID": paperID,
        "question": metric,
        "answer": answer,
        "reasoning": reasoning,
        "confidence": confidence
    }

### Worker Functions ###
# The idea we are thinking here is to create a "worker" process for each framework question
# The goal is to increase speed/efficiency of the models in applying annotations to each paper
# Currently, I am just including code to do the 8 framework questions (not including classification), to make testing easier.
# Once all models have annotated for a given question, we want to compare their responses to human annotations and each other.


# Function Name: 
# Inputs:
# Outputs: 
# Dependencies:
# Description: 
def worker_prompter(model, context, framework, PaperID, delay=3):
    # start time of function run
    start_time = time.time()

    # set up model prompt
    prompt = format_prompt(context, framework)

    # Simulates a network request, not sure if this is necessary, just following python threading tutorial
    time.sleep(delay)

    # prompt the model
    model_response = send_prompt(prompt)
    response = model_response["answer"]
    confidence = model_response["confidence"]

    # calculate time duration
    end_time = time.time()
    duration = end_time - start_time

    # add to global df containing all annotatins
    model_annotations_df.loc[len(model_annotations_df)] = [model, PaperID, framework, response, confidence, duration]


### Analysis Functions ###

# Function Name:
# Inputs:
# Outputs:
# Dependencies:
# Description:
def rank_confidence():
    pass

# Function Name:
# Inputs:
# Outputs:
# Dependencies:
# Description:
def compute_IRR():
    pass

# Function Name:
# Inputs:
# Outputs:
# Dependencies:
# Description:
def compute_accuracy():
    pass


### Main Function ###
def main():
    # container for all threads (each thread represents a worker)
    threads = []

    ### I am sure there is a more efficient way to program this ###
    ### Series of for loops to set up each thread
    # for each paper to be evaluated
    for paper in papers:
        # get paperID from PDF file name
        PaperID = ""
        # format paper for processing
        context = load_context(paper)
        # for each framework question
        for framework in pht_framework:
            for model in models:
                # Using `args` to pass positional arguments and `kwargs` for keyword arguments
                thread = threading.Thread(target=worker_prompter, args=(model, context, framework, PaperID,), kwargs={"delay": 2})
                threads.append(thread)

    ### Start each thread
    for thread in threads:
        thread.start()

    ### Wait for all threads to join and finish
    for thread in threads:
        thread.join()

if __name__ == "main":
    main()