import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.qa_with_sources.retrieval import \
    RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID"]
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL")

st.set_page_config(page_title="Interweb Explorer", page_icon="🌐")

def settings():

    # Vectorstore
    import faiss
    from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy
    from langchain_community.docstore import InMemoryDocstore  
    embeddings_model = AzureOpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    )

    embedding_size = 1536  
    index = faiss.IndexFlatL2(embedding_size)  
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # LLM
    llm = AzureChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        model=os.getenv("OPENAI_API_MODEL"),
        temperature=0.6,
        max_tokens=2048,
        streaming=True
    )
    # Search
    from langchain_community.utilities import GoogleSearchAPIWrapper
    search = GoogleSearchAPIWrapper()   

    # Initialize 
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=llm, 
        search=search, 
        num_search_results=3
    )

    return web_retriever, llm

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)


st.sidebar.image("img/ai.png")
st.header("`Interweb Explorer`")
st.info("`I am an AI that can answer questions by exploring, reading, and summarizing web pages."
    "I can be configured to use different modes: public API or private (no data sharing).`")

# Make retriever and llm
if 'retriever' not in st.session_state:
    st.session_state['retriever'], st.session_state['llm'] = settings()
web_retriever = st.session_state.retriever
llm = st.session_state.llm

# User input 
question = st.text_input("`Ask a question:`")

if question:

    # Generate answer (w/ citations)
    import logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)    
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)

    # Write answer and sources
    retrieval_streamer_cb = PrintRetrievalHandler(st.container())
    answer = st.empty()
    stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
    result = qa_chain({"question": question},callbacks=[retrieval_streamer_cb, stream_handler])
    answer.info('`Answer:`\n\n' + result['answer'])
    st.info('`Sources:`\n\n' + result['sources'])
