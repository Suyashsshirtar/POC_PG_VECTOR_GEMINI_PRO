import os
import logging
import streamlit as st
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import docx
import pandas as pd
from dotenv import load_dotenv
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("document_qa_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentQAApp:
    def __init__(self):
        self.config = {
            'max_file_size_mb': 50,
            'supported_extensions': ['pdf', 'docx', 'txt'],
            'chunk_size': 1000,
            'chunk_overlap': 100,
            'embedding_model': 'all-MiniLM-L6-v2',
            'top_k_similar_chunks': 3
        }

        self._validate_env_config()
        
        self._init_genai()
        self._init_database()
        self._init_embedding_model()
        self._init_text_splitter()

    def _validate_env_config(self):
        """Validate and log environment configuration"""
        required_env_vars = [
            'GEMINI_API_KEY', 'DB_HOST', 'DB_NAME', 
            'DB_USER', 'DB_PASSWORD'
        ]
        
        for var in required_env_vars:
            if not os.getenv(var):
                st.error(f"Missing environment variable: {var}")
                logger.error(f"Missing environment variable: {var}")
                st.stop()

    def _init_genai(self):
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            logger.info("Gemini AI configured successfully")
        except Exception as e:
            st.error("Failed to configure Gemini AI")
            logger.error(f"Gemini AI configuration error: {e}")

    def _init_database(self):
        try:
            self.conn = psycopg2.connect(
                host=os.getenv("DB_HOST"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD")
            )
            self.cursor = self.conn.cursor()
            self._create_embeddings_table()
            logger.info("Database connection established")
        except Exception as e:
            st.error("Database connection failed")
            logger.error(f"Database connection error: {e}")
            st.stop()

    def _create_embeddings_table(self):
        try:
            self.cursor.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT,    
                    embedding VECTOR(384),
                    file_name TEXT,
                    file_hash TEXT
                )
            """)
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error creating embeddings table: {e}")

    def _init_embedding_model(self):
        try:
            self.embedding_model = SentenceTransformer(
                self.config['embedding_model']
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            st.error("Failed to load embedding model")
            logger.error(f"Embedding model error: {e}")

    def _init_text_splitter(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap'],
            length_function=len
        )
        logger.info("Text splitter initialized")

    def _validate_file(self, uploaded_file):
        """Validate uploaded file"""
        if not uploaded_file:
            st.error("No file uploaded")
            return False

        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > self.config['max_file_size_mb']:
            st.error(f"File size exceeds {self.config['max_file_size_mb']} MB")
            return False

        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext not in self.config['supported_extensions']:
            st.error(f"Unsupported file type. Supported types: {', '.join(self.config['supported_extensions'])}")
            return False

        return True

    def process_document(self, uploaded_file):
        """Enhanced document processing with progress tracking"""
        if not self._validate_file(uploaded_file):
            return 0

        with st.spinner('Processing document...'):
            try:
                text = self._extract_text(uploaded_file)
                if not text:
                    st.error("Could not extract text from the document")
                    return 0

                chunks = self.text_splitter.split_text(text)
                st.info(f"Document split into {len(chunks)} chunks")

                embeddings = self.embedding_model.encode(chunks)

                self.cursor.execute(
                    "DELETE FROM document_embeddings WHERE file_name = %s", 
                    (uploaded_file.name,)
                )

                execute_values(self.cursor, """
                    INSERT INTO document_embeddings 
                    (chunk_text, embedding, file_name) 
                    VALUES %s
                """, [
                    (chunk, np.array(embedding).tolist(), uploaded_file.name) 
                    for chunk, embedding in zip(chunks, embeddings)
                ])
                self.conn.commit()

                st.success(f"Document processed: {len(chunks)} chunks stored")
                return len(chunks)

            except Exception as e:
                st.error(f"Document processing error: {e}")
                logger.error(f"Document processing error: {e}")
                return 0

    def _extract_text(self, uploaded_file):
        """Extract text from different file types with better error handling"""
        file_extension = uploaded_file.name.split('.')[-1].lower()

        try:
            if file_extension == 'pdf':
                reader = PyPDF2.PdfReader(uploaded_file)
                text = " ".join([page.extract_text() for page in reader.pages])
            elif file_extension == 'docx':
                doc = docx.Document(uploaded_file)
                text = " ".join([para.text for para in doc.paragraphs])
            elif file_extension == 'txt':
                text = uploaded_file.getvalue().decode('utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            return text
        except Exception as e:
            st.error(f"Text extraction error: {e}")
            logger.error(f"Text extraction error: {e}")
            return ""

    def similarity_search(self, query_embedding) -> List[Tuple[str, float]]:
        """Perform similarity search with better error handling"""
        top_k = self.config['top_k_similar_chunks']
        
        try:
            self.cursor.execute("""
                SELECT chunk_text, 
                       (embedding <=> %s::vector) AS distance 
                FROM document_embeddings 
                ORDER BY distance 
                LIMIT %s
            """, (np.array(query_embedding).tolist(), top_k))
            
            results = self.cursor.fetchall()
            print(results)
            return results
        except Exception as e:
            st.error("Similarity search failed")
            logger.error(f"Similarity search error: {e}")
            return []

    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer with improved prompt and error handling"""
        try:
            model = genai.GenerativeModel('gemini-pro')
            print(context)
            
            prompt = f"""
            Context: {context}
            Question: {query}
            
            Please follow these instructions to create an effective summary:
 
            1. Read the entire document carefully to understand its main ideas and key points.
 
            2. Identify the most important information, focusing on main concepts, crucial details, and significant conclusions.
 
            3. Condense this information into clear, concise sentences. Each sentence should capture a single main idea or key piece of information.
 
            4. Limit your summary to a maximum of 20 sentences. If the document is short or simple, you may use fewer sentences, but never exceed 10.
 
            5. Ensure that your sentences are:
                - Concise: Keep each point brief and to the point.
                - Self-contained: Each sentence should make sense on its own.
                - Informative: Provide substantive information, not vague statements.
                - Ordered logically: Present the information in a sequence that makes sense.
 
            6. Use your judgment to determine the appropriate level of detail. For longer or more complex documents, focus on higher-level concepts. For shorter documents, you may include more specific details.
 
            7. Avoid repetition. Each sentence should contribute unique information to the summary.
 
            8. Use clear, straightforward language. Avoid jargon unless it's essential to understanding the document's content.
 
            9. If the document contains numerical data or statistics, include the most significant figures in your summary.
 
            10. After creating your summary, review it to ensure it accurately represents the main points of the original document without any misinterpretations.

            """
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error("Answer generation failed")
            logger.error(f"Answer generation error: {e}")
            return "Unable to generate an answer. Please try again."

    def run(self):
        """Main Streamlit application with improved UI"""
        st.title("📄 Document Q&A Assistant")

        uploaded_file = st.file_uploader(
            "Choose a document", 
            type=self.config['supported_extensions']
        )

        if uploaded_file is not None:
            if st.button("Process Document"):
                chunk_count = self.process_document(uploaded_file)
                
                if chunk_count == 0:
                    st.warning("Document processing failed. Please try again.")

        query = st.text_input("🤔 Ask a question about the document")

        if query:
            try:
                query_embedding = self.embedding_model.encode([query])[0]
                similar_chunks = self.similarity_search(query_embedding)

                if not similar_chunks:
                    st.warning("No relevant information found.")
                    return

                context = "\n".join([chunk[0] for chunk in similar_chunks])
                answer = self.generate_answer(query, context)

                st.markdown("### 🎯 Answer")
                st.write(answer)

                with st.expander("📝 Retrieved Context"):
                    for i, (chunk, distance) in enumerate(similar_chunks, 1):
                        st.markdown(f"**Chunk {i} (Relevance: {1/distance:.2f})**")
                        st.text(chunk)

            except Exception as e:
                st.error("An error occurred while processing your question")
                logger.error(f"Query processing error: {e}")

    def __del__(self):
        """Safely close database connection"""
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if hasattr(self, 'conn'):
            self.conn.close()
        logger.info("Database connection closed")

def main():
    load_dotenv() 
    app = DocumentQAApp()
    app.run()

if __name__ == "__main__":
    main()