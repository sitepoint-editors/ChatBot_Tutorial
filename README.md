# How to create a ChatBot Web App and train it on your own data using the OpenAI API

In this article, you'll learn how to train and test your own ChatBot as well as turn it into a web app using the OpenAI API.

## Table of Contents
- Why make a ChatBot?
- Getting started with the OpenAI API
- Preparing your data
- Training and testing a simple ChatBot on your data
- Perfecting your ChatBot
- Turning your ChatBot into a web app

## Why make a ChatBot?
With AI having revolutionised the IT landscape in 2023, many have leveraged this movement using API providers such as OpenAI to integrate AI into their data.

A particularly good way of using AI for your ideas is making your own ChatBot.

For example, if you have a dataset of thousands of company earnings reports and you would like to explore and analyse it without getting old, a good idea would be to make a ChatBot to answer any questions you may have on the documents that you would have to manually search for otherwise. For example, you may want to ask "What year did tech companies have the best earnings", a question that you would usually have to answer by manually digging around your dataset. Luckily using a ChatBot trained on your data, you can get the answer to that question in a matter of seconds.

## Getting started with the OpenAI API
To get started on your very own ChatBot, your first need access OpenAI API. To get your OpenAI API, sign up on their [website](https://openai.com/), then click your profile icon located at the top-right corner of the home page, select "View API Keys." and click "Create New Secret Key" to generate a new API key.

## Downloading a Dataset
For this tutorial I'll be using the Wikipedia page for computers to make a simple ChatBot that can answer any general question about computers and their history.
You can download the dataset in text format [here](https://github.com/MattNikonorov/ChatBot_Tutorial/blob/main/Computer.txt).

## Training your ChatBot
Once you have your API key and dataset file, you can get started with the actual code.
Go to your favourite text-editor, create a new folder where you'll be making your ChatBot, and create an empty python file inside your new project folder.
Make sure to also create a folder named "chatbot_docs" inside your project folder and paste the dataset file into that folder (the name of the folder doesn't matter but for this tutorial it's much easier to name it "chatbot_docs").

Once you've done that, download the libraries that we're going to be using by running the following in your terminal.
```Bash
pip3 install langchain flask llama_index gradio openai pandas numpy glob datetime
```

Finally, once you've installed all the necessary libraries, paste the following code into your python file.
```Python
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

  
os.environ["OPENAI_API_KEY"] =  'YOUR_KEY'

openai.api_key =  'YOUR_KEY'  # Setting your API key

  

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") # Setting your OpenAI model

  

gfiles = glob.glob("chatbot_docs/*") # Reading your document directory

  

for g1 in  range(len(gfiles)): # Iterating through every document

  

f =  open(f"embs{g1}.csv", "w") # Creating a csv file for storing the embeddings for your ChatBot

f.write("combined") # Creating the 'combined' collumn

f.close()

  

content =  ""

with  open(f"{gfiles[g1]}", 'r') as file: # Storing the document contents

content += file.read()

content +=  "\n\n"

  

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=250)

texts = text_splitter.split_text(content) # Splitting the document content into chunks

  
  

def  get_embedding(text, model="text-embedding-ada-002"): # Defining the function that creates the embeddings needed for the Chatbot to function (It can't form answers from plain text)

text = text.replace("\n", " ")

return openai.Embedding.create(input  = [text], model=model)['data'][0]['embedding']

  

df = pandas.read_csv(f"embs{g1}.csv") # Reading the empty csv file that you created earlier for storing the embeddings

  

df["combined"] = texts # Filling the 'combined' collumn with the chunks you created earlier

for i4 in  range(len(df["combined"])):

df["combined"][i4] =  '""'  + df["combined"][i4].replace("\n", "") +  '""'  # Adding triple quotes around the text chunks to prevent syntax errors caused by double quotes in the text

df.to_csv(f"embs{g1}.csv") # Writing the data to the csv file

  

df["embedding"] = df.combined.apply(lambda  x: get_embedding(x)) # Adding and filling the 'embedding' collumn which contains the embeddings created from your text chunks

df.to_csv(f"embs{g1}.csv", index=False) # Writing the new 'embedding' collumn to the csv file

  

df = pandas.read_csv(f"embs{g1}.csv") # Reading the new csv file

  

embs = []

for r1 in  range(len(df.embedding)): # Making the embeddings readable to the chatbot by turning it into a list

e1 = df.embedding[r1].split(",")

for ei2 in  range(len(e1)):

e1[ei2] =  float(e1[ei2].strip().replace("[", "").replace("]", ""))

embs.append(e1)

  

df["embedding"] = embs # Updating the 'embedding' collumn

df.to_csv(f"embs{g1}.csv", index=False) # Writing the final version of the csv file
```

For this tutorial, I'm using the "gpt-3.5-turbo" OpenAI model since it is the fastest and is the most cost efficient. As you probably noticed, I set the temperature of the ChatBot to 0. I did this to make the ChatBot as factually accurate as possible. The temperature parameter determines the creativity of the ChatBot, where a temperature of 0 means that the ChatBot is always factually accurate and a temperature of 1 means that the ChatBot has complete freedom to make up answers and details for the sake of creativity, even if they're not accurate. The higher the temperature the more creative and less factually accurate the ChatBot is. 

Throughout this code I mention the word "embeddings", this is just what the text in your Wikipedia document gets turned into in order to be understood and made sense of by the ChatBot. Each embedding is a list of numbers ranging from -1 to 1 that associate each piece of information by how closely it is related to another.

This code makes an embeddings csv file for each document in your "chatbot_docs" folder and since you only have one (for the purpose of this tutorial), it only creates one embeddings file, but if you had more documents, the code would create an embeddings file for each document. This approach makes your ChatBot more scalable.

You're also probably wondering about the part with the chunks:
```Python
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n",  "\n"], chunk_size=2000, chunk_overlap=250) 

texts = text_splitter.split_text(content)  # Splitting the document content into chunks
```
Let me explain. This code splits the Wikipedia page about computers into chunks of 2000 characters and a chunk overlap of 250 characters. The bigger the chunk size the bigger the context of the ChatBot, but this can also make it slower, so I chose 2000 as a nice middle ground between 0 and 4096(the maximum chunk size) for this tutorial. As for the chunk overlap it is recommended by ChatGPT to keep the chunk overlap between 10% to 20% of the chunk size to keep some context between the different chunks while making sure that the chunks aren't redundant by keeping them from containing too much of the previous chunks data. The smaller the chunk overlap, the smaller the context between the chunks. The bigger the chunk overlap, the bigger the context between the chunks and the more redundant the chunk data. This code also splits the document by paragraphs by splitting the text every time that there is a newline ("\n" or "\n\n") to make the chunks more cohesive by making sure the chunks aren't split mid-paragraph.

When you run your code, you would have prepared your data to be used by the chatbot, which means you can make the actual chatbot. While the Python file that you just ran created the embeddings needed for the ChatBot to function, now you're going to have to make another python file for the actual ChatBot that takes a question as an input, and outputs an answer made by the ChatBot. Once you've created a new python file, add the following code.

```Python
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



os.environ["OPENAI_API_KEY"] =  'YOUR_KEY'

  

openai.api_key =  'YOUR_KEY'

  

ips = []

ips_times = []

  

ips_ref = []

ips_times_ref = []

  

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

  
  

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

  

def  get_embedding(text, model="text-embedding-ada-002"):

text = text.replace("\n", " ")

return openai.Embedding.create(input  = [text], model=model)['data'][0]['embedding']

  
  
  

def  logic(question):

df = pandas.read_csv(f"embs0.csv")

  

embs = []

for r1 in  range(len(df.embedding)): # Changing the format of the embeddings into a list due to a parsing error

e1 = df.embedding[r1].split(",")

for ei2 in  range(len(e1)):

e1[ei2] =  float(e1[ei2].strip().replace("[", "").replace("]", ""))

embs.append(e1)

  

df["embedding"] = embs

  

bot_message =  ""

product_embedding = get_embedding( # Creating an embedding for the question that's been asked

question

)

df["similarity"] = df.embedding.apply(lambda  x: cosine_similarity(x, product_embedding)) # Finds the relevance of each piece of data in context of the question

df.to_csv("embs0.csv")

  

df2 = df.sort_values("similarity", ascending=False) # Sorts the text chunks based on how relevant they are to finding the answer to the question

df2.to_csv("embs0.csv")

df2 = pandas.read_csv("embs0.csv")

print(df2["similarity"][0])

  

from langchain.docstore.document import Document

  

comb = [df2["combined"][0]]

docs = [Document(page_content=t) for t in comb] # Gets the most relevant text chunk

  

prompt_template = question +  """

  

{text}

  

"""

  

PROMPT  = PromptTemplate(template=prompt_template, input_variables=["text"])

chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT) # Preparing the LLM

  

output = chain.run(docs) # Formulating an answer (this is where the magic happens)

  

return output

  

response = logic("when was the first computer made?") # Passing the question to the ChatBot

  

print(response)
```

