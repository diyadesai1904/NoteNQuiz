import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os
import time
import io
from pypdf import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
import random
import re

# Download NLTK resources if not present
try:
    nltk.data.find('tokenizers/punkt')
except (nltk.downloader.DownloadError, LookupError):
    nltk.download('punkt')

# --- Configuration ---
MODEL_PATH = "./final_notes_quiz_model"
MAX_INPUT_LENGTH = 384
MAX_OUTPUT_LENGTH = 150 
QA_MAX_LENGTH = 25
QUESTION_WORDS = ["who", "what", "where", "when", "why", "how", "which"] 

# --- Helper Functions ---

@st.cache_resource
def load_model():
    """Loads the fine-tuned T5 model and tokenizer."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model directory not found at: {MODEL_PATH}")
        st.stop()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with st.spinner(f"Loading T5 model onto {device}..."):
        tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
    
    return tokenizer, model, device

def extract_text_from_file(uploaded_file):
    """Extracts text content from a PDF or TXT file."""
    # (Extraction logic remains the same)
    if uploaded_file.type == "text/plain":
        string_io = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        text = string_io.read()
        return text.strip()
    
    elif uploaded_file.type == "application/pdf":
        try:
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}. Ensure the PDF is not encrypted.")
            return None
    else:
        st.warning("Unsupported file type. Please upload a PDF or TXT file.")
        return None

def generate_output(input_text, tokenizer, model, device, max_length, temperature=0.7):
    """Generates the output using the T5 model, with tunable temperature."""
    try:
        input_ids = tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=MAX_INPUT_LENGTH, 
            truncation=True
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                temperature=temperature, 
                top_k=50
            )
        
        return tokenizer.decode(output[0], skip_special_tokens=True)

    except Exception as e:
        return f"An error occurred during generation: {e}"

def clean_answer(answer):
    """Cleans up T5-generated answers."""
    answer = answer.strip()
    # Remove any stray starting punctuation or sentence endings
    answer = re.sub(r'^\W+', '', answer)
    answer = re.sub(r'\.$', '', answer)
    # Capitalize the first letter for display
    return answer.capitalize()

def generate_single_quiz_item(context_chunk, question_type, tokenizer, model, device):
    """Generates a QG item, forces a factual answer, and filters poor output."""
    if len(context_chunk.split()) < 40:
        return None, None, None
        
    sentences = sent_tokenize(context_chunk)
    if not sentences:
        return None, None, None
    
    base_sentence = random.choice(sentences)
    
    try:
        if question_type == "QG":
            # --- Step 1: Generate Question (QG Task) ---
            dummy_answer = "key concepts" # Slightly abstract answer to get broad question
            qg_prefix = f"generate question: context: {context_chunk} answer: {dummy_answer}"
            generated_question = generate_output(qg_prefix, tokenizer, model, device, max_length=64)

            # --- HEURISTIC 1: Filter out non-questions ---
            if "?" not in generated_question or len(generated_question.split()) < 3 or generated_question.split()[0].lower() not in QUESTION_WORDS:
                 return None, None, None 

            # --- Step 2: Generate Factual Answer (QA Task) ---
            qa_prefix = f"question and context: {generated_question} context: {context_chunk}"
            generated_answer = generate_output(qa_prefix, tokenizer, model, device, max_length=QA_MAX_LENGTH)
            
            # --- HEURISTIC 2: Filter out poor answers ---
            cleaned_answer = clean_answer(generated_answer)
            if not cleaned_answer or len(cleaned_answer.split()) > 10: # Answer too long/empty
                 return None, None, None

            return "QG_QA", generated_question, cleaned_answer

        elif question_type == "MCQ":
            # --- MCQ Generation ---
            dummy_question = f"What is a key fact regarding {base_sentence.split()[:3]}?"
            input_prefix = f"generate mcq: question: {dummy_question} context: {context_chunk}"
            result_raw = generate_output(input_prefix, tokenizer, model, device, max_length=MAX_OUTPUT_LENGTH)
            
            # --- HEURISTIC 3: Ensure MCQ format is parsable ---
            if "|" not in result_raw:
                 return None, None, None

            return "MCQ", dummy_question, result_raw
    
    except Exception as e:
        return None, None, None
        
    return None, None, None

# --- Main App Execution ---

# Load model first (cached)
tokenizer, model, device = load_model()

# Set Streamlit Page Config (Design remains the same)
st.set_page_config(
    page_title="Smart Study Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- Sidebar Content (Design remains the same) ---
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("This AI-powered tool helps you:")
    st.markdown("- **Generate concise and detailed notes**")
    st.markdown("- **Create interactive quizzes**")
    st.markdown("- **Test your understanding**")
    
    st.markdown("---")
    
    st.markdown("**Supported formats:**")
    st.markdown("- **PDF** (`.pdf`)")
    st.markdown("- **Text** (`.txt`)")
    
    st.markdown("---")
    
    st.markdown("**How to use:**")
    st.markdown("1. Upload your study material")
    st.markdown("2. Choose Notes or Quiz")
    st.markdown("3. Review and download results")
    
# --- Main Content Section ---

st.title("🧠 Smart Study Assistant")
st.markdown("### Upload your study material and generate Notes or Quizzes")

# 1. File Uploader Section
uploaded_file = st.file_uploader(
    "Choose your file (PDF or TXT)",
    type=["txt", "pdf"],
    key="file_uploader_main"
)
st.caption("Limit 200MB per file · PDF, TXT")
st.markdown("---")

# 2. Configuration
st.subheader("⚙️ Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    output_choice = st.radio(
        "Select Output Type:",
        ('Generate Both (Notes & Quiz)', 'Notes Only', 'Quiz Only'),
        index=0 
    )

with col2:
    difficulty = st.select_slider(
        "Quiz Difficulty:",
        options=['Easy', 'Medium', 'Hard'],
        value='Medium'
    )

with col3:
    num_questions = st.slider(
        "Number of Questions:",
        min_value=1,
        max_value=10,
        value=5
    )

generate_button = st.button("Generate Smart Study Material", type="primary")
st.markdown("---")

# --- Generation Logic ---

if generate_button:
    
    if not uploaded_file:
        st.warning("Please upload a file to begin generating material.")
        st.stop()
    
    # Set parameters based on difficulty
    if difficulty == 'Easy':
        chunk_words = 150
        question_type = "QG" 
    elif difficulty == 'Medium':
        chunk_words = 100
        question_type = "QG"
    else: # Hard
        chunk_words = 60
        question_type = "MCQ"

    with st.status("Processing and Generating...", expanded=True) as status:
        
        status.update(label="Extracting text from file...", state="running", expanded=True)
        full_context = extract_text_from_file(uploaded_file)
        
        if not full_context or len(full_context.split()) < 40:
            status.update(label="Text extraction failed or document is too short (< 40 words).", state="error")
            st.stop()
        
        sentences = sent_tokenize(full_context)
        total_words = len(full_context.split())
        st.success(f"File processed! Extracted {total_words} words.")
        
        start_time = time.time()

        # ----------------------------------------------------
        # PHASE 1: GENERATE SUMMARY NOTES (CLEANED)
        # ----------------------------------------------------
        if output_choice in ['Generate Both (Notes & Quiz)', 'Notes Only']:
            status.update(label="Generating Summary Notes (Phase 1/2): Applying Cleaning Heuristics...", state="running")
            
            summary_max_chunk_words = 250 
            summary_chunks = []
            current_chunk = ""
            
            for sent in sentences:
                if len((current_chunk + " " + sent).split()) < summary_max_chunk_words:
                    current_chunk += " " + sent
                else:
                    summary_chunks.append(current_chunk.strip())
                    current_chunk = sent
            if current_chunk:
                summary_chunks.append(current_chunk.strip())

            full_summary_sentences = []
            
            # Process each chunk
            for chunk in summary_chunks:
                input_prefix = f"summarize: {chunk}"
                
                # Use slightly higher temp (0.9) to push for less extractive output
                generated_summary_raw = generate_output(input_prefix, tokenizer, model, device, max_length=MAX_OUTPUT_LENGTH, temperature=0.9)
                
                if "| long:" in generated_summary_raw:
                    long_summary = generated_summary_raw.split("| long:")[-1].strip()
                else:
                    long_summary = generated_summary_raw
                
                # Post-process: Break into sentences and add unique ones
                for sent in sent_tokenize(long_summary):
                    if sent not in full_summary_sentences and len(sent.split()) > 5:
                         full_summary_sentences.append(sent)

            final_summary = " ".join(full_summary_sentences)
            
            st.subheader("📖 Your Concise Notes")
            st.markdown(final_summary)
            st.download_button(
                "Download Notes as TXT",
                data=final_summary.strip(),
                file_name="smart_notes.txt",
                mime="text/plain"
            )

        # ----------------------------------------------------
        # PHASE 2: GENERATE QUIZ (QG -> QA CHAINED & FILTERED)
        # ----------------------------------------------------
        if output_choice in ['Generate Both (Notes & Quiz)', 'Quiz Only']:
            status.update(label=f"Generating {num_questions} Quiz Questions (Phase 2/2): Applying Filters...", state="running")
            
            # Chunking and Sampling logic
            quiz_chunks = []
            current_quiz_chunk = ""
            for sent in sentences:
                if len((current_quiz_chunk + " " + sent).split()) < chunk_words:
                    current_quiz_chunk += " " + sent
                else:
                    quiz_chunks.append(current_quiz_chunk.strip())
                    current_quiz_chunk = sent
            if current_quiz_chunk:
                quiz_chunks.append(current_quiz_chunk.strip())
            
            if len(quiz_chunks) > num_questions * 2:
                selected_chunks = random.sample(quiz_chunks, num_questions * 2)
            else:
                selected_chunks = quiz_chunks

            quiz_items = []
            
            # Generation Loop with Error Handling/Filtering
            for i, chunk in enumerate(selected_chunks):
                if len(quiz_items) >= num_questions:
                    break
                
                item_type, q_text_or_dummy, result_raw = generate_single_quiz_item(chunk, question_type, tokenizer, model, device)
                
                if item_type == "QG_QA":
                    quiz_items.append({
                        "type": "QG",
                        "question": f"Question {len(quiz_items) + 1}: {q_text_or_dummy.strip()} (Difficulty: {difficulty})",
                        "answer": result_raw
                    })
                
                elif item_type == "MCQ" and "|" in result_raw:
                    try:
                        options_part = result_raw.split("|")[0].replace("options:", "").strip()
                        answer_part = result_raw.split("|")[-1].replace("answer:", "").strip()
                        
                        quiz_items.append({
                            "type": "MCQ",
                            "question": f"Question {len(quiz_items) + 1}: What is a key detail from this chunk? (Difficulty: {difficulty})",
                            "options": [opt.strip() for opt in options_part.split('|')],
                            "answer": clean_answer(answer_part)
                        })
                    except:
                        # Log error internally, skip item
                        pass

            # Display Quiz and Final Feedback
            generated_count = len(quiz_items)
            
            if generated_count < num_questions:
                st.warning(f"Could only generate {generated_count} out of {num_questions} requested items due to model instability or strict output filtering.")
                
            st.subheader("📚 Generated Quiz")
            st.markdown(f"**Difficulty:** {difficulty} | **Type:** {question_type} | **Questions Generated:** {generated_count}")

            quiz_text_for_download = ""

            for i, item in enumerate(quiz_items):
                st.markdown(f"**{item.get('question', f'Question {i+1}')}**")
                quiz_text_for_download += f"Q{i+1}: {item.get('question', f'Question {i+1}')}\n"
                
                if item['type'] == 'MCQ':
                    options_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
                    for idx, option in enumerate(item['options']):
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**{options_map.get(idx, '?')}.** {option}")
                        quiz_text_for_download += f"  {options_map.get(idx, '?')}. {option}\n"
                    st.markdown(f"**Correct Answer:** `{item['answer']}`")
                    quiz_text_for_download += f"Answer: {item['answer']}\n\n"
                    
                elif item['type'] == 'QG':
                    st.markdown(f"**Factual Answer:** `{item['answer']}`")
                    quiz_text_for_download += f"Answer: {item['answer']}\n\n"
                
                st.markdown("---")

            st.download_button(
                "Download Quiz with Answers (TXT)",
                data=quiz_text_for_download,
                file_name=f"smart_quiz_{difficulty}_{len(quiz_items)}q.txt",
                mime="text/plain"
            )

        # Final Status Update
        status.update(label=f"Generation Complete! Total Time: {time.time() - start_time:.2f} seconds.", state="complete")