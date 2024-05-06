
import shutup
shutup.please()

import os
import re
import glob
import time
import torch
import textwrap
import langchain
import transformers
import streamlit as st

from langchain import PromptTemplate, LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,pipeline

HF_TOKEN = os.environ.get('HF_TOKEN')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Paths:

    base = os.path.dirname(os.getcwd())
    
    pdfs = os.path.join(base, 'arquivos' + os.sep + 'br')
    pdfs_folder = pdfs.split('/')[-1]
    
    outputs = os.path.join(base, "outputs")
    output_folder = os.path.join(outputs, pdfs_folder + '-vectordb')

@st.cache(allow_output_mutation=True)
def loadMistral():

    model_repo = "mistralai/Mistral-7B-Instruct-v0.1"

    tokenizer = AutoTokenizer.from_pretrained(
        model_repo, 
        token= HF_TOKEN
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = True,
    )   

    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        token= HF_TOKEN,
        quantization_config = bnb_config,
        device_map = 'auto',
        low_cpu_mem_usage = True
    )

    return model, tokenizer

@st.cache(allow_output_mutation=True)
def loadDocuments():

    print("Loading Documents...")

    path = os.path.join(Paths.outputs, f"{Paths.pdfs_folder}-faiss_index")

    loader = DirectoryLoader(
        Paths.pdfs,
        glob= "./*.pdf",
        loader_cls= PyPDFLoader,
        show_progress= True,
        use_multithreading= True
    )

    documents = loader.load()

    print("Splitting text...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 0
    )

    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceInstructEmbeddings(
        model_name = 'all-MiniLM-L6-v2',
        model_kwargs = {
            "device": device
        }
    )

    if not os.path.exists(path):
        
        print("Creating Embeddings")

        vectordb = FAISS.from_documents(
            documents = documents, 
            embedding = embeddings
        )
        
    else:

        print("Loading Embeddings")

        ### load vector DB embeddings
        vectordb = FAISS.load_local(
            os.path.join(Paths.outputs, f"{Paths.pdfs_folder}-faiss_index"),
            embeddings,
            allow_dangerous_deserialization= True
        )
    
    return vectordb

def loadPipeline(model, tokenizer, temperature=0.0, top_p=0.95, repetition_penalty=1.15):

    print("Temperatura:", temperature)
    print("Top P:", top_p)

    pipe = pipeline(
        task = "text-generation",
        model = model,
        tokenizer = tokenizer,
        pad_token_id = tokenizer.eos_token_id,
        max_length = 4096,
        temperature = temperature,
        top_p = top_p,
        repetition_penalty = repetition_penalty
    )

    llm = HuggingFacePipeline(pipeline= pipe)

    return llm

def loadQAChain(model, tokenizer, vectordb, temperature, top_p, repetition_penalty):

    prompt_template = """
    <s>[INST] You are a helpful lawyer and a client asked you a question about a legal matter. [/INST]
    <s>[INST] Answer the question based on the context. [/INST]

    Question: {question} 

    Context: {context}

    <s>[INST] Provide practical advice or general information that helps the client understand their options or next steps in addressing their legal matter without referencing specific laws, decrees, or articles. [/INST]
    <s>[INST] Ensure the Answer is translated to Brazilian Portuguese. [/INST]

    Answer:"""

    PROMPT = PromptTemplate(
        template = prompt_template, 
        input_variables = ["context", "question"]
    )

    retriever = vectordb.as_retriever(
        search_type= "similarity_score_threshold", 
        
        search_kwargs= {
            "score_threshold": 0.9
        }
    )

    llm = loadPipeline(model, tokenizer, temperature, top_p, repetition_penalty)

    qa_chain = RetrievalQA.from_chain_type(
        llm = loadPipeline(model, tokenizer, temperature, top_p, repetition_penalty),
        chain_type = "stuff", 
        retriever = retriever, 
        chain_type_kwargs = {
            "prompt": PROMPT
        },
        verbose = False,
        return_source_documents = True
    )

    return qa_chain

def remove_after(text):
    return re.sub(r'Context:.*$', '', text, flags=re.DOTALL)

def process_answer(llm, query):

    ans = llm.invoke(query)

    newans = ans['result'].split("[/INST]")[-1].strip().replace("Answer:", "")
    newans = textwrap.fill(newans, width= 100)
    newans = remove_after(newans)

    return ans['query'], newans

if __name__ == "__main__":

    st.title("🤖 CDC 📝")
    st.sidebar.title("Configurações")

    with st.spinner('Carregando modelo...'):
        model, tokenizer = loadMistral()
    
    with st.spinner('Carregando documentos...'):
        vectordb = loadDocuments()
    
    # Sidebar
    temperature = st.sidebar.slider("Temperatura", min_value= 0.0, max_value= 1.0, value= 0.0, step= 0.05)
    top_p = st.sidebar.slider("Top P", min_value= 0.0, max_value= 1.0, value= 0.95, step= 0.01)
    repetition_penalty = st.sidebar.slider("Penalidade de Repetição", min_value= 1.0, max_value= 2.0, value= 1.10, step=0.05)

    # Main content
    qa_chain = loadQAChain(model, tokenizer, vectordb, temperature, top_p, repetition_penalty)

    query = st.text_input("Pergunta", placeholder= "O que é o código de proteção e defesa do consumidor?")

    if(query):

        with st.spinner('Gerando resposta...'):
            query, response = process_answer(qa_chain, query)
        
        finalResponse = f"**Pergunta:** {query}\n\n**Resposta:** {response}"

        st.write(finalResponse)