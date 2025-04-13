import os
import streamlit as st
import pickle
import time
import re
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from typing import List, Optional, Dict, Any, ClassVar
from langchain.schema import LLMResult, Generation, Document
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Streamlit UI
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 50px; margin-bottom: 0;'>NewsLens🔍</h1>
        <h3 style='font-size: 20px; margin-top: -30px; color: #EC6A5E;'>News Research Tool</h3>
    </div>
""", unsafe_allow_html=True)

st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"
main_placeholder = st.empty()

class GeminiLLM(LLM):
    model_name: ClassVar[str] = "gemini-1.5-flash-latest"

    def __init__(self, model_name: str = "gemini-1.5-flash-latest", **kwargs):
        super().__init__(**kwargs)
        self._model = genai.GenerativeModel(model_name)

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = self._model.generate_content(prompt)
            if response and hasattr(response, "text"):
                return response.text
            return "No response"
        except Exception as e:
            return f"Error: {e}"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

# Initialize Gemini model
model = GeminiLLM()
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def clean_text(text):
    """Removes unnecessary content such as navigation menus, advertisements, and extra spaces."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'AdvertisementRemove Ad', '', text, flags=re.IGNORECASE)
    navigation_terms = ["Home", "News", "Markets", "Stocks", "Economy", "IPO", "Trending Topics", "Follow Us On:"]
    for term in navigation_terms:
        text = text.replace(term, "")
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

if process_url_clicked:
    loader = WebBaseLoader(urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    
    try:
        data = loader.load()
        if not data:
            st.error("Error: No data was extracted. The URLs may be blocking scraping.")
            st.stop()
        
        # Ensure each document has proper metadata with source
        for doc in data:
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            doc.metadata['source'] = doc.metadata.get('source', doc.metadata.get('source_url', urls[0]))
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','], 
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    
    try:
        # Split documents while preserving metadata
        docs = text_splitter.split_documents(data)
        if not docs:
            st.error("Error: Text splitting failed. No valid documents were found.")
            st.stop()
    except Exception as e:
        st.error(f"Text splitting failed: {e}")
        st.stop()

    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    
    try:
        # Create vectorstore from documents (not from texts) to preserve metadata
        vectorstore = FAISS.from_documents(docs, embedding_function)
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)
        st.success(f"URLs successfully processed!✅")
    except Exception as e:
        st.error(f"FAISS storage failed: {e}")
        st.stop()

    time.sleep(2)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                retriever = vectorstore.as_retriever()

                # Test the retriever
                sample_results = retriever.get_relevant_documents("test")
                if not sample_results:
                    st.error("FAISS retriever returned no results. The index might be empty.")
                    st.stop()
            
            #st.success("FAISS Vectorstore loaded successfully! ✅")
        except Exception as e:
            st.error(f"Failed to load FAISS vectorstore: {e}")
            st.stop()

        chain = RetrievalQAWithSourcesChain.from_llm(llm=model, retriever=retriever)
        
        try:
            result = chain({"question": query}, return_only_outputs=True)
            
            if isinstance(result, dict):
                answer = result.get("answer", "No answer found.")
                sources = result.get("sources", "")

                st.header("Answer")
                st.write(answer)

                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n") if isinstance(sources, str) else sources
                    for source in sources_list:
                        if source.strip():  # Skip empty strings
                            st.write(source)
            else:
                st.error("Unexpected result format.")
        except Exception as e:
            st.error(f"Query processing failed: {e}")