import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.qa_with_sources.retrieval import \
    RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os

# Access secrets 
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_CSE_ID= st.secrets["GOOGLE_CSE_ID"]
api_key = st.secrets["api_key"]
api_version = st.secrets["api_version"]
azure_endpoint = st.secrets["azure_endpoint"]
deployment_chat = st.secrets["deployment_chat"]
deployment_embed = st.secrets["deployment_embed"]
st.set_page_config(page_title="Crawler for Jarina", page_icon="🌐")

def settings():

    # Vectorstore
    import faiss
    from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy
    from langchain_community.docstore import InMemoryDocstore  
    embeddings_model = AzureOpenAIEmbeddings(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        model=deployment_embed,
    )

    embedding_size = 1536  
    index = faiss.IndexFlatL2(embedding_size)  
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # LLM
    llm = AzureChatOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        model=deployment_chat,
        temperature=0.6,
        max_tokens=2048,
        streaming=True,
    )
    # Search
    from langchain_google_community import GoogleSearchAPIWrapper
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
st.header("`Crawler for Jarina`")
st.info("`Ez a tool válaszokat talál és aggregál több lépésben egy adott kérdésre úgy, hogy megtalálja a forrásokat, elolvassa és összefoglalja`")

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
    try:
        result = qa_chain({"question": question}, callbacks=[retrieval_streamer_cb, stream_handler])
        answer.info('`Answer:`\n\n' + result['answer'])
        st.info('`Sources:`\n\n' + result['sources'])
    except Exception as e:
        st.error(f"An error occurred: {e}")
