
# Employee Assistant Chatbot with Retrieval Augmented Generation

This Streamlit app functions as an Employee Assistant Bot, designed to quickly retrieve and provide information from an Employee Handbook using natural language processing. It integrates LangChain for document retrieval and OpenAI's GPT models for processing and generating responses.

![Chatbot Features](Image.jpg)

## Features

- **Document Loader**: Loads and processes PDF documents to be used as a knowledge base.
- **Text Splitter**: Splits large texts into manageable chunks to optimize the retrieval process.
- **Vector Embeddings**: Converts text chunks into vector embeddings for efficient similarity searches.
- **Chat Prompt Template**: Structures the chatbot responses to ensure relevance and clarity.
- **Chain Integration**: Combines document retrieval and chat response to generate coherent answers.
- **UI Design**: A user-friendly interface that allows employees to interact with the bot through natural language queries.

## Setup

To run this project locally, you will need Python 3.10 and the ability to install several dependencies.

### Environment Setup

Create a virtual environment and activate it:

```bash
conda create -p venv python=3.10
conda activate venv/
```

### Install Dependencies

Install the required Python libraries specified in `requirements.txt` (also referred to as `re.txt`):

```bash
pip install -r requirements.txt
```

### Create .env file

In this project, we utilize OpenAI, which requires an API key for access. You can obtain one by registering on the OpenAI website. Once you have your API key, please add it to the .env file. Additionally, I have included my Langchain API key in this file to log all responses and track costs. If you wish to do the same, please paste your Langchain API key and a project name into the .env file.

```bash
OPENAI_API_KEY=your_openai_API_key
LANGCHAIN_API_KEY=your_langchain_API_key
LANGCHAIN_PROJECT="RAG_Chatbot"
```

### Import required packages and set up envionment variables

This snippet loads environment variables from a .env file using the dotenv module to securely manage sensitive information such as API keys. It sets the OPENAI_API_KEY and LANGCHAIN_API_KEY for access within the application. Additionally, it enables a tracing feature for the Langchain API by setting LANGCHAIN_TRACING_V2 to "true".

```bash
from langchain_community.document_loaders import TextLoader
from PIL import Image
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
```

### Load and process the document into chunks

This snippet utilizes the Langchain library to load and process a PDF document. It employs the PyPDFLoader to load the "Employee Handbook copy.pdf" into the docs variable. Then, it uses a RecursiveCharacterTextSplitter to split the loaded document into smaller chunks of 100 characters with a 20-character overlap, facilitating efficient text processing and retrieval.

```bash
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("Employee Handbook copy.pdf")
docs  = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
                               chunk_size = 100,
                               chunk_overlap = 20, 
                               )

chunk_documents = text_splitter.split_documents(docs)
```
You can also get the optimal chunk size instead of choosing randomly. For getting the optimal one i started with 100 and an overlap of 20. But furthur I will use faithfullness and relevancy indicator along with response time to get optimal chunk size. 



### Vector Embeddings

This Python script is part of the Vector Embeddings setup for a text processing application using Langchain and OpenAI libraries. It imports the OpenAIEmbeddings class for generating vector embeddings of text, and FAISS from Langchain for efficient similarity search. The script creates a FAISS vector store using embeddings generated from chunk_documents, which are smaller sections of a text document, enabling faster and more efficient document retrieval.

```bash
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(chunk_documents,
                          OpenAIEmbeddings()
                          )
```

### Building the chain

This snippet sets up a retrieval-augmented chatbot using the Langchain library and GPT models from OpenAI. It defines a custom chat prompt template that instructs the model on how to answer queries without repeating the provided context. The chatbot integrates with a document retriever built on a FAISS vector store (db) containing chunked document embeddings. The script then creates a retrieval chain that combines document retrieval with the generated chat responses to answer questions effectively. Finally, it prepares to deploy the chatbot on a Streamlit web interface with a wide layout configuration.
```bash
## Design chat prompt template 

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
I will tip you $1000 if the user finds the answer helpful.
Do not give 'Based on the provided context' in the answer.                                         
< context>
{context} 
</ context>
Question : {input}                                         
"""                                   )

from langchain_community.llms import openai
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")  #gpt-3.5-turbo

## Chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm, prompt)

retriever = db.as_retriever()

## Retriver chain
from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever , document_chain)

# response = retrieval_chain.invoke({"input" : "Work hours of an employee"})
import streamlit as st
st.set_page_config(layout="wide")
```

Beyond this stage, the customization of the user interface in Streamlit can vary depending on specific requirements and objectives. Streamlit offers flexible options for tailoring the UI, allowing developers to adjust layouts, add interactive components, and incorporate aesthetic elements that enhance user engagement and functionality specific to the application's purpose. This adaptability makes Streamlit an ideal choice for developers looking to create personalized and dynamic interfaces for their projects.









