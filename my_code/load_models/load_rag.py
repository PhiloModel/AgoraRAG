import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
from my_code.load_data.import_pdfs import load_pdfs_from_dir

# Potrzebne funkcje langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def load_rag_based_on_pdfs(dir_path, persist_directory, chunk_size = 1000, chunk_overlap = 150, model_name='gpt-3.5-turbo-instruct'):

    _ = load_dotenv(find_dotenv()) # read local .env file
    
    openai.api_key = os.environ['OPENAI_API_KEY']

    documents = load_pdfs_from_dir(dir_path)

    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    splits = text_splitter.split_documents(documents)

    # Embedding
    embedding = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )

    # Opis meta danych
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="Źródło wiadomosci",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="Strona",
            type="integer",
        ),
    ]

    document_content_description = "Streszczenie poglądów filozofa."

    # Retriver
    llm = OpenAI(model=model_name, temperature=0)
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectordb,
        document_content_description,
        metadata_field_info,
        verbose=True
    )

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "~ PhiloBot!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:
    """

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    # Pamięć
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )

    return qa


def load_saved_rag_model(persist_directory, model_name='gpt-3.5-turbo-instruct'):
    
    _ = load_dotenv(find_dotenv()) # read local .env file
    
    openai.api_key = os.environ['OPENAI_API_KEY']
    
    # Inicjalizacja embeddingów
    embedding = OpenAIEmbeddings()

    # Załaduj istniejącą bazę wektorów
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    # Opis meta danych
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="Źródło wiadomości",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="Strona",
            type="integer",
        ),
    ]

    document_content_description = "Streszczenie poglądów filozofa."

    # Inicjalizacja LLM
    llm = OpenAI(model=model_name, temperature=0)

    # Tworzenie retrievera
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectordb,
        document_content_description,
        metadata_field_info,
        verbose=True
    )

    # Szablon promptu
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "~ PhiloBot!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:
    """

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    # Pamięć konwersacji
    memory = ConversationBufferMemory(
        memory_key="chat_history",  
        return_messages=True
    )

    # Tworzenie łańcucha QA
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )

    return qa