from llama_index import SimpleDirectoryReader, VectorStoreIndex, LLMPredictor, ServiceContext, PromptHelper
from llama_index import StorageContext, load_index_from_storage
from llama_index import ServiceContext
from langchain.chat_models import ChatOpenAI
from llama_index.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from config import *  
import gradio as gr
import sys
import os


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
TOKENIZERS_PARALLELISM = False

# Enable this if you enable local Chat GPT to cut costs, but will not work with gradio web interface
# service_context = ServiceContext.from_defaults(llm="local")


def init_index(directory_path):
    # model params
    # max_input_size: maximum size of input text for the model.
    # num_outputs: number of output tokens to generate.
    # max_chunk_overlap: maximum overlap allowed between text chunks.
    # chunk_size_limit: limit on the size of each text chunk.
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 0
    chunk_size_limit = 600    

    # llm predictor with langchain ChatOpenAI
    # ChatOpenAI model is a part of the LangChain library and is used to interact with the GPT-3.5-turbo model provided by OpenAI
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    
    #Costs Money, sends the query to ChatGPT API
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-4", max_tokens=num_outputs))
    
    #Local ChatGPT, but this cannot use web interface as of now so commented it out
    #llm_predictor = OpenAI(model="gpt-4", temperature=0, max_tokens=num_outputs)

    # read documents from docs folder
    documents = SimpleDirectoryReader(directory_path).load_data()

    # init index with documents data
    # This index is created using the LlamaIndex library. 
    # It processes the document content and constructs the index to facilitate efficient querying
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    # save the created index
    #index.storage_context.persist
    index.storage_context.persist(persist_dir="/Users/mk/Projects/Custom-Domain-Training-AI/chatbot/db")


    return index

def chatbot(input_text):
    # load index
    storage_context = StorageContext.from_defaults(persist_dir="/Users/mk/Projects/Custom-Domain-Training-AI/chatbot/db")
    index = load_index_from_storage(storage_context)
    # SummaryIndexRetriever
    retriever = index.as_retriever(retriever_mode='default')
    # get response for the question
    query_engine = index.as_query_engine()
    #response = query_engine.query(retriever, response_mode="compact")
    #query_engine = RetrieverQueryEngine.from_args(retriever, response_mode='compact')
    response = query_engine.query(input_text)
    #response = index.query(input_text, response_mode="compact")


    return response.response

# create index
#init_index("/Users/mk/Google Drive/Proposals")
init_index("docs")
# create ui interface to interact with gpt-3 model
iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, placeholder="Enter your question here"),
                     outputs="text",
                     title="Rhytha AI ChatBot: Your Knowledge Companion Powered-by ChatGPT",
                     description="Ask any questions",
                     allow_screenshot=True)
iface.launch(share=True)