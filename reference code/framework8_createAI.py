## Approach 1: Minimal Changes - Keep FAISS, Replace LLM with CreateAI
## This keeps your existing RAG pipeline and only changes the LLM call

import json
import pathlib
import re
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import glob
import requests
import time

## Global variables
framework_question = """Please answer this question with a number--which of the following best apply to the design in the study: 
    (1) entirely symmetrical (i.e., tic tac toe),
    (2) mostly symmetrical (i.e., undertale's differing NPC behaviors based on previous playthroughs),
    (3) both symmetrical and asymmetrical (i.e., pandemic with its similar turn structure but different character abilities),
    (4) mostly asymmetrical (i.e., tag),
    (5) entirely asymmetrical (i.e., keep talking, nobody explodes)"""


## CreateAI Configuration - ONLY NEED QUERY TOKEN
CREATEAI_API_URL = "https://api-main.aiml.asu.edu/query"
CREATEAI_TOKEN = "ASU_TOKEN"

## LangChain wrapper class for embeddings
class WrappedEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

embedding_model = WrappedEmbeddings(model_name="all-MiniLM-L6-v2")

def is_paper_annotated(pdf_filepath):
    """
    Check if a paper has already been annotated by looking for its output JSON file
    
    Args:
        pdf_filepath: Path to the PDF file
    
    Returns:
        Boolean indicating if annotation exists
    """
    base_name = os.path.splitext(os.path.basename(pdf_filepath))[0]
    output_path = f"outputs/framework8/{base_name}.json"
    return os.path.exists(output_path)

def call_createai_llm(prompt_text, model_provider="aws", model_name="claude3_opus"):
    """
    Call CreateAI API to generate response
    
    Args:
        prompt_text: The prompt to send to the LLM
        model_provider: Provider name (e.g., "openai", "aws")
        model_name: Model name (e.g., "gpt4o", "claude3_5_sonnet")
    
    Returns:
        String response from the LLM
    """
    headers = {
        "Authorization": f"Bearer {CREATEAI_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "action": "query",
        "request_source": "script",
        "query": prompt_text,
        "model_provider": model_provider,
        "model_name": model_name,
        "session_id": "test-session-1",
        "model_params": {
            "temperature": 0.7,
            "system_prompt": "You are an expert research assistant.",
        }
    }
    
    try:
        response = requests.post(CREATEAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extract the response text from the API response
        if isinstance(result, dict):
            return result.get('response', result.get('content', str(result)))
        return str(result)
    
    except requests.exceptions.RequestException as e:
        print(f"Error calling CreateAI API: {e}")
        return None

def build_few_shot_prompt(question, context_docs):
    prompt_parts = []
    prompt_parts.append("Your task is to read the following paper and answer questions about it. Please answer every question to the best of your ability.")
    prompt_parts.append(f"Paper Context: {context_docs}")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("\nProvide your response in the following format:")
    prompt_parts.append("Answer: <only the answer from the list of answer options here, no additional text>")
    prompt_parts.append("Reasoning: <your explanation here>")
    prompt_parts.append("Confidence: <only include a value for how confident you are that your answer is correct, giving a score between 0.0 and 1.0 here>")
    return "\n".join(prompt_parts)

def load_pdf(file_path):
    """Load PDF and convert to chunked documents"""
    markdown_text = pymupdf4llm.to_markdown(file_path)
    pathlib.Path("output.md").write_bytes(markdown_text.encode())
    chunked_data = chunk_data(markdown_text)
    return chunked_data

def chunk_data(text):
    """Chunk text for processing"""
    split_text = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_doc = split_text.create_documents([text])
    return chunked_doc

def create_vec(chunked_data):
    """Create vector store from chunked documents"""
    vector_store = FAISS.from_documents(chunked_data, embedding_model)
    return vector_store

def prompt(vector_store, paper_filename):
    """Apply annotations to paper using LLM"""
    annotations = []
    context = retrieve(vector_store, framework_question)
    response = generate(context, framework_question, paper_filename)
    annotations.append(response)
    return annotations

def retrieve(vector_store, metric):
    """Retrieve relevant context for a metric"""
    retrieved_context = vector_store.similarity_search(metric)
    return retrieved_context

def generate(context, metric, paper_filename):
    """Generate annotation using CreateAI LLM with few-shot prompting"""
    docs_content = "\n\n".join(doc.page_content for doc in context)
    
    # Build few-shot prompt
    prompt_text = build_few_shot_prompt(metric, docs_content)
    
    # Call CreateAI API instead of local LLM
    response = call_createai_llm(prompt_text)
    
    if response is None:
        return {
            "paperID": paper_filename,
            "question": metric,
            "answer": "ERROR",
            "reasoning": "API call failed",
            "confidence": "0.0"
        }
    
    # Parse response
    answer_match = re.search(r"(?i)^Answer:\s*(.*)", response, re.MULTILINE)
    reasoning_match = re.search(r"(?i)^Reasoning:\s*(.*)", response, re.MULTILINE)
    confidence_match = re.search(r"(?i)^Confidence:\s*([\d.]+)", response, re.MULTILINE)
    
    answer = answer_match.group(1).strip() if answer_match else "0"
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "Not found"
    confidence = confidence_match.group(1).strip() if confidence_match else "0.0"
    
    return {
        "paperID": paper_filename,
        "question": metric,
        "answer": answer,
        "reasoning": reasoning,
        "confidence": confidence
    }

def generate_json(annotation_list, original_filename):
    """Generate JSON file with annotations matching the input filename"""
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    output_path = f"outputs/framework8/{base_name}.json"
    os.makedirs("outputs/framework8", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotation_list, f, indent=2, ensure_ascii=False)
    
    return output_path

def main():
    # Check for API token
    if not CREATEAI_TOKEN:
        print("ERROR: CREATEAI_TOKEN environment variable not set!")
        print("Please set it with: export CREATEAI_TOKEN='your_token_here'")
        return
    
    print("Starting up....")
    start_time = time.perf_counter()

    headers = {
    "Authorization": f"Bearer {CREATEAI_TOKEN}",
    "Content-Type": "application/json"
    }
    payload = {"action": "query", "query": "test", "model_provider": "aws", "model_name": "claude3_opus"}

    response = requests.post(CREATEAI_API_URL, headers=headers, json=payload)
    print(response.status_code)
    print(response.text)

    directory_path = f"../../papers"
    file_pattern = os.path.join(directory_path, '*.pdf')
    
    pdf_files = glob.glob(file_pattern)
    print(f"Found {len(pdf_files)} PDF files to process.\n")

    #pdf_files = ["../../papers/butler11.pdf", "../../papers/keate94.pdf"]
    
    print(f"Using token: {CREATEAI_TOKEN[:8]}... (length {len(CREATEAI_TOKEN)})")
    
    # Filter out already annotated papers
    papers_to_process = []
    skipped_papers = []
    
    for file_path in pdf_files:
        if is_paper_annotated(file_path):
            skipped_papers.append(os.path.basename(file_path))
        else:
            papers_to_process.append(file_path)
    
    if skipped_papers:
        print(f"\n⏭️  Skipping {len(skipped_papers)} already annotated papers:")
        for paper in skipped_papers:
            print(f"   - {paper}")
        print()
    
    print(f"📝 Processing {len(papers_to_process)} new papers...\n")

    for file_path in papers_to_process:
        try:
            filename = os.path.basename(file_path)
            print(f"Processing {filename}...")
            
            chunked_data = load_pdf(file_path)
            vector_store = create_vec(chunked_data)
            annotations = prompt(vector_store, filename)
            output_path = generate_json(annotations, file_path)
            
            print(f"✓ Completed: {filename} -> {os.path.basename(output_path)}\n")
            
        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}\n")
            continue
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\n{'='*60}")
    print(f"Script execution time: {elapsed_time:.4f} seconds")
    print(f"Total papers found: {len(pdf_files)}")
    print(f"Papers skipped (already annotated): {len(skipped_papers)}")
    print(f"Papers processed: {len(papers_to_process)}")
    print(f"{'='*60}")
    print("Program execution complete.")

if __name__ == "__main__":
    main()