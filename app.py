import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.google_genai import GoogleGenAI
import streamlit as st
from llama_index.core import ServiceContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter


load_dotenv()
llm=GoogleGenAI(
    model="gemini-3.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

@st.cache_data(show_spinner=False)
def f():
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=20)
    storage_context=StorageContext.from_defaults(persist_dir="./storage")
    index=load_index_from_storage(storage_context)
    return index.as_query_engine()


st.title('Financial Stock Analysis using LlamaIndex')

st.header("Reports:")

report_type = st.selectbox(
    'What type of report do you want?',
    ('Single Stock Outlook', 'Competitor Analysis'))


if report_type == 'Single Stock Outlook':
    symbol = st.text_input("Stock Symbol")

    if symbol:
        with st.spinner(f'Generating report for {symbol}...'):
            query_engine=f()
            response = query_engine.query(f"Write a report on the outlook for {symbol} stock from the years 2023-2027. Be sure to include potential risks and headwinds.")
            print(type(response))

            st.write(str(response))
            

if report_type == 'Competitor Analysis':
    symbol1 = st.text_input("Stock Symbol 1")
    symbol2 = st.text_input("Stock Symbol 2")

    if symbol1 and symbol2:
        query_engine=f()
        with st.spinner(f'Generating report for {symbol1} vs. {symbol2}...'):
            response = query_engine.query(f"Write a report on the competition between {symbol1} stock and {symbol2} stock.")

            st.write(str(response))




