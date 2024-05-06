
'''
APP - 'Código de Proteção e Defesa do Consumidor'

en-US: 'Consumer Protection and Defense Code'

Required libraries and imports for the application.

Here is a brief description of the libraries used in the application:

1- The `shutup` library is used to suppress the warnings from the transformers library.
2- The `langchain` library is used to create the pipeline for the question-answering model.
3- The `transformers` library is used to load the model and tokenizer from the Hugging Face model repository.
4- The `streamlit` library is used to create the web application.
'''

import shutup
shutup.please()

import os
import re
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

'''
The Hugging Face access token is stored in an environment variable
Also the device is set to GPU if available, otherwise it is set to CPU.
'''

HF_TOKEN = os.environ.get('HF_TOKEN')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Paths:

    '''Class with the purpose of storing the paths of files and folders.

    To run with different files, change the path of the PDFs folder in the variable `pdfs`.
    '''

    base = os.path.dirname(os.getcwd())
    
    pdfs = os.path.join(base, 'arquivos' + os.sep + 'br')
    pdfs_folder = pdfs.split('/')[-1]
    
    outputs = os.path.join(base, "outputs")
    output_folder = os.path.join(outputs, pdfs_folder + '-vectordb')

@st.cache(allow_output_mutation=True)
def loadMistral():

    '''Function to load the Mistral model and tokenizer from the Hugging Face model repository.

    A few notes about the model and the configuration used:
        1- The version used is the 7B-Instruct-v0.1, which is a instruct fine-tuned version of the main model.
        2- The BitsAndBytes quantization is used to reduce the model size and memory usage since the model is large and the application is running on my local machine.
        3- The cache is set to True to allow the model to be cached and not reloaded every time the application is run.
    '''

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

    '''Function to load the documents and create the embeddings using the FAISS library.

    A few steps are performed in this function:
        1- Load the documents from the PDFs folder.
        2- Split the text into chunks.
        3- Create the embeddings using the Hugging Face model.
        4- If the FAISS index is not saved in the outputs folder, the embeddings are created and saved.
        5- If the FAISS index is saved, it is loaded from the outputs folder.
    '''

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

    '''Function to load the pipeline for the language model.

    The reason this function is created is to allow the user to change the temperature, top_p, and repetition_penalty values in the sidebar using the pre-loaded model and tokenizer.
    '''

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

    '''Function to load the QA chain for the language model.

    The QA chain is created using the Mistral model and the FAISS index created from the documents.

    The prompt template is created to provide the context and the question to the model.
    Notice that the prompt, as stated in the mistral documentation, requires the [INST] tags to be present in the prompt.
    '''

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
    '''Function to remove the text after the 'Context' tag in the answer.' This is to handle the text that should not be part of the answer.'''
    return re.sub(r'Context:.*$', '', text, flags=re.DOTALL)

def process_answer(llm, query):
    
    '''Function to process the answer from the language model.

    It is also possible to cite the sources from the documents. However, in this example 
    i'm using documents that usually are related with laws and from my previous experience, those models have issues citing laws and articles.
    '''

    ans = llm.invoke(query)

    newans = ans['result'].split("[/INST]")[-1].strip().replace("Answer:", "")
    newans = textwrap.fill(newans, width= 100)
    newans = remove_after(newans)

    return ans['query'], newans

if __name__ == "__main__":

    '''Main function to run the application.

    streamlit run app.py
    '''

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