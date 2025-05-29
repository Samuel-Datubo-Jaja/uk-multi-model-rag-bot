# SQLite compatibility fix for Chromadb
import sqlite3
print(f"SQLite version: {sqlite3.sqlite_version}")

# Try alternative vector store approach if SQLite version is too old
import os
os.environ["LANGCHAIN_CHROMA_ALLOW_DEPRECATED_BACKEND"] = "true"

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from cloud_storage import download_vectorstore
import time
import numpy as np
from gradio_client import Client

# Page config
st.set_page_config(
    page_title="StructureGPT - UK Building Regulations Assistant",
    page_icon="🏗️",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Hugging Face Space URL
HF_SPACE_URL = "https://samueljaja-llama3-1-8b-merged-uk-building-regulations.hf.space"


# Create a cached client instance for Hugging Face
@st.cache_resource
def get_gradio_client():
    """Get a cached Gradio client instance"""
    try:
        client = Client(HF_SPACE_URL)
        return client
    except Exception as e:
        st.error(f"Error connecting to model: {str(e)}")
        return None

# Hybrid search function for better document retrieval
def hybrid_search(vectorstore, query, k=4):
    """Combine vector search with BM25 keyword search for better results"""
    from rank_bm25 import BM25Okapi
    
    # Get documents from vector search
    vector_docs = vectorstore.similarity_search(query, k=k*2)
    
    # Extract content and metadata
    contents = [doc.page_content for doc in vector_docs]
    
    # Prepare corpus for BM25
    tokenized_corpus = [doc.split() for doc in contents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Get BM25 scores
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Normalize scores to 0-1 range
    if np.max(bm25_scores) > 0:
        bm25_scores = bm25_scores / np.max(bm25_scores)
    
    # Combine with vector search position (proxy for vector similarity)
    # Weight: 70% vector similarity, 30% BM25
    vector_weights = np.linspace(1, 0, len(contents))
    combined_scores = (0.7 * vector_weights) + (0.3 * bm25_scores)
    
    # Get top k results based on combined score
    top_indices = np.argsort(combined_scores)[::-1][:k]
    top_docs = [vector_docs[i] for i in top_indices]
    
    return top_docs

# Response cleaning functions
def clean_model_response(response: str) -> str:
    """Clean up model response by removing instruction tags and truncating repetition"""
    # First, ensure response is a string
    if not isinstance(response, str):
        # Handle non-string response
        if hasattr(response, '__str__'):
            response = str(response)
        else:
            return "Error: Received non-string response from model"
    
    # Continue with normal cleaning
    cleaned = response.replace("[INST]", "").replace("[/INST]", "")
    
    # Remove any repeated paragraphs (a simple approach)
    paragraphs = []
    seen = set()
    for paragraph in cleaned.split("\n\n"):
        # Only keep paragraph if we haven't seen it before
        if paragraph.strip() and paragraph.strip() not in seen:
            paragraphs.append(paragraph)
            seen.add(paragraph.strip())
    
    return "\n\n".join(paragraphs)   

def clean_response_for_display(response):
    """Comprehensive cleaning of model responses for display"""
    # Step 1: Remove all instruction tags and sections
    if "Instructions:" in response:
        response = response.split("Instructions:")[0]
    
    # Step 2: Remove question and context repetition
    lines = response.split('\n')
    filtered_lines = []
    skip_until_empty_line = False
    
    for line in lines:
        if "Question:" in line or "Context:" in line:
            skip_until_empty_line = True
            continue
        if skip_until_empty_line and line.strip() == "":
            skip_until_empty_line = False
            continue
        if not skip_until_empty_line:
            filtered_lines.append(line)
    
    response = '\n'.join(filtered_lines)
    
    # Step 3: Remove "Based on the following context..." intro text
    response = response.replace("Based on the following context from UK Building Regulations, provide a detailed answer to the question.", "")
    
    # Step 4: Handle table content - detect and format any tables in the response
    if ("<table>" in response.lower() or 
        ("<th>" in response.lower() and "<td>" in response.lower()) or
        ("| " in response and " |" in response)):
        
        # Extract and format table data
        try:
            lines = response.split('\n')
            table_lines = []
            in_table = False
            header_added = False
            
            for line in lines:
                # Detect table start
                if ("<table>" in line.lower() or 
                    "<th>" in line.lower() or 
                    ("|" in line and any(x in line for x in ["type", "volume", "width", "depth", "requirement"]))):
                    in_table = True
                
                if in_table:
                    # Clean the line of HTML tags
                    clean_line = line
                    for tag in ["<table>", "</table>", "<tr>", "</tr>", "<td>", "</td>", "<th>", "</th>", "<thead>", "</thead>", "<tbody>", "</tbody>"]:
                        clean_line = clean_line.replace(tag, " ")
                    
                    # Format as markdown table row if not already
                    if "|" not in clean_line and len(clean_line.strip()) > 0:
                        # Split by multiple spaces to find columns
                        parts = [p for p in clean_line.split("  ") if p.strip()]
                        if len(parts) >= 2:  # Only proceed if we have at least 2 columns
                            clean_line = "| " + " | ".join(parts) + " |"
                    
                    # Add the line if it looks like a table row
                    if "|" in clean_line and clean_line.strip():
                        # Count pipes to ensure it's a proper table row
                        if clean_line.count("|") >= 3:  # At least 2 columns (3 pipes)
                            table_lines.append(clean_line.strip())
                            
                            # After first row, add separator if needed
                            if len(table_lines) == 1 and not header_added:
                                # Count columns
                                cols = clean_line.count("|") - 1
                                table_lines.append("| " + " | ".join(["---"] * cols) + " |")
                                header_added = True
                
                # End of table detection (blank line after table content)
                elif in_table and not line.strip():
                    in_table = False
            
            # If we found table lines, reconstruct the response with the formatted table
            if table_lines:
                # Find a good place to insert the table
                intro_text = ""
                remaining_text = ""
                
                # Check if there's text before the table
                for i, line in enumerate(lines):
                    if ("<table>" in line.lower() or 
                        "<th>" in line.lower() or 
                        ("|" in line and any(x in line for x in ["type", "volume", "width", "depth", "requirement"]))):
                        intro_text = "\n".join(lines[:i]).strip()
                        remaining_text = "\n".join(lines[i+len(table_lines):]).strip()
                        break
                
                # If we couldn't find a clear division, use a generic approach
                if not intro_text and not remaining_text:
                    # Look for a natural introduction sentence
                    for phrase in ["is provided below", "is as follows", "is shown below"]:
                        if phrase in response:
                            parts = response.split(phrase, 1)
                            intro_text = parts[0] + phrase
                            remaining_text = parts[1].split("\n\n", 1)[-1] if "\n\n" in parts[1] else ""
                            break
                    
                    # If still no intro found, create a generic one based on content
                    if not intro_text:
                        # Try to extract what the table is about from the content
                        topic = ""
                        if "clay type" in response.lower():
                            topic = "Clay type and Volume change potential"
                        elif "strip foundation" in response.lower():
                            topic = "Strip foundation requirements"
                        elif "minimum width" in response.lower():
                            topic = "Minimum width requirements"
                        else:
                            # Generic fallback
                            topic = "requested information"
                        
                        intro_text = f"The table of {topic} is provided below:"
                
                # Combine intro, table and any remaining text
                formatted_response = intro_text + "\n\n" + "\n".join(table_lines)
                if remaining_text:
                    formatted_response += "\n\n" + remaining_text
                
                response = formatted_response
        except Exception as e:
            # If table formatting fails, log the error but continue with original response
            print(f"Table formatting error: {str(e)}")
    
    # Step 5: Remove repeated paragraphs
    paragraphs = []
    seen_paragraphs = set()
    
    for paragraph in response.split("\n\n"):
        cleaned = paragraph.strip()
        if cleaned and cleaned not in seen_paragraphs:
            paragraphs.append(paragraph)
            seen_paragraphs.add(cleaned)
    
    response = "\n\n".join(paragraphs)
    
    # Step 6: Ensure response ends with punctuation
    if response and not response.rstrip().endswith((".", "!", "?")):
        response = response.rstrip() + "."
    
    return response.strip()

# Query the fine-tuned Llama 3.1-8B model on Hugging Face
def query_hf_model(prompt, temperature=0.1, max_tokens=256, top_p=0.9):
    """Send a query to the Hugging Face-hosted Llama model using gradio_client."""
    client = get_gradio_client()
    if client is None:
        return "Error: Could not connect to model service."
    
    # Try up to 3 times with increasing delays
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Show status if retrying
            if attempt > 0:
                st.info(f"Retrying connection (attempt {attempt+1}/{max_retries})...")
                
            # Call the model
            result = client.predict(
                prompt,                # prompt
                temperature,           # temperature
                max_tokens,            # max_tokens
                top_p,                 # top_p
                api_name="/timed_predict"
            )
            return clean_model_response(result)
            
        except Exception as e:
            if attempt < max_retries - 1:
                # Wait with exponential backoff before retrying
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                st.error(f"Failed to connect after {max_retries} attempts: {str(e)}")
                return f"Sorry, the model is currently unavailable. Technical details: {str(e)}"

# Initialize RAG components
@st.cache_resource
def init_rag():
    """Initialize RAG components with caching"""
    try:
        # Check if main_chroma_data exists
        if not os.path.exists("./main_chroma_data"):
            download_vectorstore()

        # Initialize embeddings
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                encode_kwargs={'normalize_embeddings': True}  # Added for stability
            )
        except Exception as e:
            st.error(f"Error initializing embeddings: {str(e)}")
            return None, None, None

        # Initialize vector store
        try:
            vectorstore = Chroma(
                collection_name="main_construction_rag",
                embedding_function=embeddings,
                persist_directory="./main_chroma_data"
            )
        except Exception as e:
            st.warning("Using deprecated backend due to SQLite version constraints")
            # Use alternative initialization if needed
            from langchain_community.vectorstores import Chroma as ChromaDeprecated
            vectorstore = ChromaDeprecated(
                collection_name="main_construction_rag",
                embedding_function=embeddings,
                persist_directory="./main_chroma_data"
            )

        # Check if GROQ API key is set
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("Error: GROQ_API_KEY not found in environment variables")
            return vectorstore, None, None

        # Initialize LLMs - both GROQ models
        try:
            llm_70b = ChatGroq(
                api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0.1
            )
            
            llm_8b = ChatGroq(
                api_key=groq_api_key,
                model_name="llama3-8b-8192",
                temperature=0.1
            )
        except Exception as e:
            st.error(f"Error initializing LLMs: {str(e)}")
            return vectorstore, None, None

        return vectorstore, llm_70b, llm_8b
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None, None, None

# Initialize systems
vectorstore, llm_70b, llm_8b = init_rag()
hf_client = get_gradio_client()

# Check if initialization successful
if vectorstore is None:
    st.error("Failed to initialize vector store. Please check logs.")
if llm_70b is None or llm_8b is None:
    st.warning("GROQ models not initialized. Please check GROQ API key. You can still use the fine-tuned model.")
if hf_client is None:
    st.warning("Fine-tuned model not initialized. Please check connection to Hugging Face. You can still use GROQ models if available.")

# Sidebar for model selection and feedback
with st.sidebar:
    st.title("🔧 Model Settings")
    
    # Model selection toggle
    model_option = st.radio(
        "Select Model:",
        [
            "Llama-3.3-70B (GROQ, Most accurate)",
            "Llama3-8B (GROQ, Balanced)",
            "Fine-tuned Llama-3.1-8B (HF, Domain-specific)"
        ],
        index=0,  # Default to 70B model
        help="Choose between different models based on your needs"
    )
    
    # Display selected model details
    if model_option == "Llama-3.3-70B (GROQ, Most accurate)":
        st.info("Using Llama-3.3-70B: Highest accuracy but slower responses")
        if llm_70b is None:
            st.error("This model is currently unavailable. Please check GROQ API key.")
    elif model_option == "Llama3-8B (GROQ, Balanced)":
        st.info("Using Llama3-8B: Faster responses with good accuracy")
        if llm_8b is None:
            st.error("This model is currently unavailable. Please check GROQ API key.")
    else:  # Fine-tuned model
        st.info("Using fine-tuned Llama-3.1-8B: Optimized for building regulations")
        if hf_client is None:
            st.error("This model is currently unavailable. Please check HF connection.")
    
    # Advanced options expander
    with st.expander("Advanced Options"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, 
                              help="Higher values make output more random, lower values more deterministic")
        search_k = st.slider("Number of documents to retrieve", 2, 8, 4, 1,
                            help="More documents provide broader context but might dilute relevance")
    
    st.divider()
    
    # Feedback section
    st.title("📝 Feedback")
    feedback = st.text_area("Share your feedback on the answers:", height=100)
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

# Main interface
st.title("🏗️ StructureGPT - UK Building Regulations AI Assistant")
st.markdown("""
This AI assistant helps answer questions about UK building regulations using:
- Official GOV.UK Building Regulations Documents
- Technical documentation and guidance
""")

# Add testing phase notice with warning styling
st.warning("""
⚠️ **TESTING PHASE** - StructureGPT is currently in beta testing.
""")

# User input
question = st.text_input("Enter your question about UK building regulations:")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    elif vectorstore is None:
        st.error("RAG system not properly initialized. Please check the errors above.")
    else:
        # Determine which model to use
        if model_option == "Llama-3.3-70B (GROQ, Most accurate)" and llm_70b is not None:
            selected_model = "70B"
            model_name = "Llama-3.3-70B"
        elif model_option == "Llama3-8B (GROQ, Balanced)" and llm_8b is not None:
            selected_model = "8B" 
            model_name = "Llama3-8B"
        elif model_option == "Fine-tuned Llama-3.1-8B (HF, Domain-specific)" and hf_client is not None:
            selected_model = "FT"
            model_name = "Fine-tuned Llama-3.1-8B"
        else:
            # Fallback to any available model
            if llm_70b is not None:
                selected_model = "70B"
                model_name = "Llama-3.3-70B (fallback)"
                st.info("Selected model unavailable. Using Llama-3.3-70B as fallback.")
            elif llm_8b is not None:
                selected_model = "8B"
                model_name = "Llama3-8B (fallback)"
                st.info("Selected model unavailable. Using Llama3-8B as fallback.")
            elif hf_client is not None:
                selected_model = "FT"
                model_name = "Fine-tuned Llama-3.1-8B (fallback)"
                st.info("Selected model unavailable. Using fine-tuned model as fallback.")
            else:
                st.error("No models are available. Please check configurations.")
            
        
        with st.spinner(f"Searching regulations and generating answer using {model_name}..."):
            try:
                # Get relevant documents using hybrid search if available, otherwise regular search
                try:
                    docs = hybrid_search(vectorstore, question, k=search_k)
                except:
                    docs = vectorstore.similarity_search(question, k=search_k)
                
                contexts = [doc.page_content for doc in docs]
                
                # Generate answer
                context_text = "\n\n".join(contexts)
                
                # Construct prompt based on model
                if selected_model in ["70B", "8B"]:
                    # GROQ models prompt
                    prompt = f"""Based on the following context from UK Building Regulations, provide a clear and detailed answer to the question.
                    Include specific references to regulations where available.
                    
                    Question: {question}
                    
                    Context: {context_text}
                    
                    Answer:"""
                    
                    # Process with selected GROQ model
                    if selected_model == "70B":
                        response = llm_70b.invoke(prompt)
                        content = response.content
                    else:  # 8B model
                        response = llm_8b.invoke(prompt)
                        content = response.content
                        
                else:  # Fine-tuned model
                    # Enhanced prompt for fine-tuned model
                    prompt = f"""Based on the following context from UK Building Regulations, provide a detailed answer to the question.

                    Question: {question}

                    Context: {context_text}

                    Instructions:
                    1. Answer directly and concisely
                    2. Cite specific regulations and document sections when applicable
                    3. Format measurements and requirements clearly
                    4. If the exact information is not in the context, state this clearly
                    5. For technical specifications, use proper formatting
                    6. If you need to show a table, format it as markdown
                    7. Avoid repetition - state information only once

                    Answer:"""
                    
                    # Query status indicator
                    query_status = st.empty()
                    query_status.info("Connecting to custom fine-tuned Llama 3.1-8B model...")
                    
                    # Call the fine-tuned model
                    content = query_hf_model(prompt, temperature=temperature)
                    
                    # Clear the status message
                    query_status.empty()
                
                # Process response based on model
                if selected_model == "FT":
                    processed_response = clean_response_for_display(content)
                else:
                    processed_response = content
                
                # Display answer
                st.markdown("### Answer")
                st.markdown(processed_response)
                
                # Display model used
                st.caption(f"Answer generated using {model_name}")
                
                # Display sources
                with st.expander("View Source Documents"):
                    for i, context in enumerate(contexts, 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(context)
                        st.divider()
                
                # Add feedback section
                st.subheader("Was this answer helpful?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful"):
                        st.success("Thank you for your feedback!")
                with col2:
                    if st.button("👎 Not Helpful"):
                        st.info("Thank you for your feedback. Please let us know how we can improve in the sidebar.")
                        
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*StructureGPT is a research project in testing phase. Always verify information with official sources.*")