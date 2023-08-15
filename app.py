import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size = 1000, 
        chunk_overlap = 200, 
        length_function = len
    
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory = "./chroma_db")
    return vector_store

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm, 
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

if "conversation" not in st.session_state:
    st.session_state.conversation = None


if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

def main():
    load_dotenv()
    st.set_page_config(page_title="AI", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("AI")
    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        handle_user_input(user_question)

    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

    if (vector_store != None):
        vector_store.get() 

    #vector_store = FAISS.load_local("./dados")
    #create conversation chain
    st.session_state.conversation = get_conversation_chain(vector_store)

    #st.write(user_template.replace("{{MSG}}", "hello robot"), unsafe_allow_html=True)
    #st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Seus documentos")
        pdf_docs = st.file_uploader("Envie os PDFs e clique em 'Processar'", accept_multiple_files=True)
        if st.button("Processar"):
            with st.spinner("Processando..."):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)

                #get text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                #create vector store
                vector_store = get_vector_store(text_chunks)

                #vector_store = FAISS.load_local("./dados")
                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)
if __name__ == '__main__':
    main()