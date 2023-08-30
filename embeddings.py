# Importing the libraries (you won't need all of them right now but you will need them later)

from flask import Flask, request, jsonify
from flask import Flask, render_template, request, url_for
from llama_index import SimpleDirectoryReader, GPTListIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
import time
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas
import openai
import numpy as np
import glob
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = 'YOUR_KEY'

openai.api_key = 'YOUR_KEY'

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") # Setting your OpenAI model

gfiles = glob.glob("chatbot_docs/*") # Reading your document directory

for g1 in range(len(gfiles)): # Iterating through every document

    f = open(f"embs{g1}.csv", "w") # Creating a csv file for storing the embeddings for your ChatBot

    f.write("combined") # Creating the 'combined' collumn

    f.close()

    content = ""

    with open(f"{gfiles[g1]}", 'r') as file: # Storing the document contents
        content += file.read()
        content +=  "\n\n"


    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=250)
    texts = text_splitter.split_text(content) # Splitting the document content into chunks


    def get_embedding(text, model="text-embedding-ada-002"): # Defining the function that creates the embeddings needed for the Chatbot to function (It can't form answers from plain text)
        text = text.replace("\n", " ")
        return openai.Embedding.create(input  = [text], model=model)['data'][0]['embedding']


    df = pandas.read_csv(f"embs{g1}.csv") # Reading the empty csv file that you created earlier for storing the embeddings
    df["combined"] = texts # Filling the 'combined' collumn with the chunks you created earlier

    for i4 in range(len(df["combined"])):
        df["combined"][i4] =  '""'  + df["combined"][i4].replace("\n", "") +  '""'  # Adding triple quotes around the text chunks to prevent syntax errors caused by double quotes in the text

    df.to_csv(f"embs{g1}.csv") # Writing the data to the csv file

    df["embedding"] = df.combined.apply(lambda  x: get_embedding(x)) # Adding and filling the 'embedding' collumn which contains the embeddings created from your text chunks

    df.to_csv(f"embs{g1}.csv", index=False) # Writing the new 'embedding' collumn to the csv file
    df = pandas.read_csv(f"embs{g1}.csv") # Reading the new csv file
    embs = []

    for r1 in range(len(df.embedding)): # Making the embeddings readable to the chatbot by turning them into lists
        e1 = df.embedding[r1].split(",")
        for ei2 in range(len(e1)):
            e1[ei2] = float(e1[ei2].strip().replace("[", "").replace("]", ""))
        embs.append(e1)


    df["embedding"] = embs # Updating the 'embedding' collumn

    df.to_csv(f"embs{g1}.csv", index=False) # Writing the final version of the csv file

