import streamlit as st 
from streamlit_chat import message
import tempfile
import chromadb
import PyPDF2
from langchain_community.vectorstores import Chroma
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


# Initialize Chroma DB client
client = chromadb.PersistentClient(path="./chroma_db")


#Methods Section
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "mistral-7b-instruct-v0.2.Q8_0.gguf",
        model_type="mistral",
        config={'max_new_tokens': 512,
                        'temperature': 0.01,'context_length' : 2048}
    )
    return llm

# Function to convert PDF to text
def pdf_to_text(file_path):
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range( len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    pdf_file.close()
    return text

def conversational_chat(query):
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever(),return_source_documents=True)
    query = "Who is Mohit Kumar Raj Badi?"
    result = chain({'question': query, 'chat_history': st.session_state['history']})
    st.session_state['history'].append((query,result["answer"]))
    return result["answer"]

st.title("Private GPT🦙🦜")
uploaded_file = st.sidebar.file_uploader("Upload your Data", type="pdf")

# We will use HuggingFace embeddings
embeddings = HuggingFaceEmbeddings()

# If any file is uploaded
if uploaded_file:
    #use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    collection = client.get_or_create_collection(name="my_collection")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

     # Convert PDF to text
    text = pdf_to_text(tmp_file_path)
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)

    # Convert chunks to vector representations and store in Chroma DB
    documents_list = []
    embeddings_list = []
    ids_list = []
        
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk)
            
        documents_list.append(chunk)
        embeddings_list.append(vector)
        ids_list.append(f"{tmp_file.name}_{i}")
        
        
    collection.add(
        embeddings=embeddings_list,
        documents=documents_list,
        ids=ids_list
    )
    

# Load the LLM Model from the Local
llm = load_llm()

# create the open-source embedding function
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
collection = client.get_or_create_collection("my_collection")
query_vector = embeddings.embed_query("Who is Mohit ?")

# Query Chroma DB with the vector representation
results = collection.query(query_embeddings=query_vector, n_results=2 , include=["documents"])

db = Chroma(
    client=client,
    collection_name="my_collection",
    embedding_function= embeddings
)


if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hi ! I am Private GPT. "]

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hey ! 👋']

#container for the chat history
response_container = st.container()

#container for the user's text input
container = st.container()

with container:
    with st.form(key="my_form",clear_on_submit = True):
        user_input = st.text_input("Query : ", placeholder = "Message Private GPT here :",key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i],is_user=True,key=str(i) + '_user',avatar_style="fun-emoji")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")