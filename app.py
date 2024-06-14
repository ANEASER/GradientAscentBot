import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import customer
import requests

load_dotenv()
API_TOKEN = os.getenv('OPENAI_API_KEY')

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.getvalue().decode("utf-8")
    return text

def get_csv_text(csv_docs):
    text = ""
    for csv in csv_docs:
        df = pd.read_csv(csv)
        text += df.to_string()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=API_TOKEN, temperature=0.5, max_tokens=100)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def sanitize_response(response):
    response = response.lower()
    response = response.replace("customer_code_100", "you")
    filtering_phrases = ["cluster", "strategy", "customer", "customer_code", "customer_code_100", "favorite"] 
    for phrase in filtering_phrases:
        response = response.replace(phrase, " ")

    return response

def handle_userinput(user_question, username=None):

    st.session_state.chat_history.append({'content': user_question, 'role': 'user'})

    cluster = customer.identify_customer(username)
    favourite_items = customer.customer_transactions(username)

    user_specific_phrases = ["me", "my", "mine", "i"]
    promotional_phrases = ["offer", "promotions", "discount", "hotdeals", "sale"]

    if any(word in user_question.lower() for word in user_specific_phrases):
        prompt = (
            f"As a cluster {cluster} customer, {user_question}. "
            f"Analyze favorite items: {favourite_items}. "
            "If the requested department or category items are not in favorite items, do not consider them. "
            "Also, do not mention strategies or the cluster in the response."
        )
    elif any(word in user_question.lower() for word in promotional_phrases):
        prompt = (
            f"As a cluster {cluster} customer, {user_question}. "
            "Suggest promotions by creating custom promotions according to strategy. "
            "Do not expose the strategy or mention the cluster, just provide the promoted items in the response. no need to justify why the items are promoted."
        )
    else:
        prompt = user_question + "don not consider the cluster,user or strategy to the response.answer genarally"
        
    response = st.session_state.conversation({'question': prompt})
    bot_response = sanitize_response(response['chat_history'][-1].content)
    st.session_state.chat_history.append({'content': bot_response, 'role': 'bot'})
    st.markdown(f"**Bot:** {bot_response}")

def recommend_some(username):
    url = "http://localhost:4000/recommend/"

    items = customer.customer_transactions(username)

    json_data = {
         "items" : items
    }

    response = requests.post(url, json=json_data)
    data = response.json()
    words = data['recommendation']
    
    return words

def main():
    load_dotenv()
    st.set_page_config(page_title="Welcome To GradientAscentBot", page_icon="🤖:")


    # initialize login 
    username = st.text_input("Enter your username:")

    if username:
        st.markdown("---")
        st.sidebar.text("") 
        st.success(f"Username '{username}' entered successfully!")

        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "processed_docs" not in st.session_state:
            st.session_state.processed_docs = False

        chat_history = st.session_state.chat_history

        for message in chat_history:
            if message['role'] == 'user':
                st.markdown(f"{username}: {message['content']}")
            elif message['role'] == 'bot':
                st.markdown(f"**🤖:** {message['content']}")

        st.markdown("---")
        st.sidebar.text("") 
        st.header("Welcome To GradientAscentBot 🤖:")

        # User input box at the bottom
        user_question = st.text_input("Ask a question for your preferences:")

        if user_question and st.session_state.conversation:
            handle_userinput(user_question, username=username)
            st.empty()

        with st.sidebar:
            st.subheader("Process documents")

            file_docs = [filename for filename in os.listdir(".") if filename.endswith('.pdf') or filename.endswith('.txt') or filename.endswith('.csv')]

            if not st.session_state.processed_docs:
                with st.spinner("Processing"):
                    try:
                        raw_text = ""
                        pdf_docs = [filename for filename in file_docs if filename.endswith('.pdf')]
                        txt_docs = [filename for filename in file_docs if filename.endswith('.txt')]
                        csv_docs = [filename for filename in file_docs if filename.endswith('.csv')]

                        for pdf in pdf_docs:
                            raw_text += get_pdf_text([pdf])

                        for txt in txt_docs:
                            with open(txt, 'r', encoding='utf-8') as file:
                                raw_text += file.read()

                        for csv in csv_docs:
                            df = pd.read_csv(csv)
                            raw_text += df.to_string()

                        # Split text into chunks
                        text_chunks = get_text_chunks(raw_text)

                        # Create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.session_state.processed_docs = True
                        st.success("Documents processed successfully.")
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")

            st.subheader("Recommended For You")

            words = recommend_some(username)

            for word in words:
                st.markdown(f"- {word}")

if __name__ == '__main__':
    main()