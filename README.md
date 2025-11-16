#  Smart Study Assistant: The AI Content Accelerator

 **Accelerate Learning: From Document to Knowledge in Seconds**

The **Smart Study Assistant** is a production-ready web application built to solve the challenge of rapid information processing and knowledge extraction from unstructured academic or enterprise documents.  
It transforms lengthy PDF and TXT files into actionable learning modules using a **fine-tuned Text-to-Text Transfer Transformer (T5)** model, deployed seamlessly on **Streamlit**.

This project showcases **multi-task NLP deployment**, **inference chaining**, and **robust error handling** — all essential for real-world document processing systems.

---

##  Core Product Features

| **Feature** | **User Benefit** | **Technical Driver** |
|--------------|------------------|----------------------|
| **Comprehensive Note Generation** | Converts 50,000+ word documents into concise, flowing study notes. | Multi-chunk summarization pipeline and post-processing heuristics. |
| **Factual Quiz Generation** | Generates verified, fact-checked short-answer questions. | QG → QA Inference Chaining (Sequential T5 calls for Question → Answer). |
| **Customized Learning Paths** | Adjusts question complexity and type (QG or MCQ) based on user-selected difficulty (Easy, Medium, Hard). | Dynamic input chunking based on difficulty level (e.g., smaller chunks for 'Hard'). |
| **Broad Compatibility** | Accepts common document formats (.pdf, .txt). | PyPDF and robust text extraction logic. |

---

##  Deployment & Architecture

The application uses a **unified architecture** where a single **T5-small** model performs three distinct NLP tasks — summarization, question generation, and question answering.

###  Technology Stack

| **Layer** | **Technology** | **Purpose** |
|------------|----------------|--------------|
| **Interface** | Streamlit | Fast, responsive Python-native application layer. |
| **Model** | T5-small (Hugging Face) | Encoder-Decoder core fine-tuned for sequence-to-sequence tasks. |
| **Backend** | PyTorch, Transformers | Model loading and efficient CPU/GPU inference. |
| **Utilities** | pypdf, nltk | Document parsing, text cleaning, and sentence tokenization. |

---

##  The QG → QA Chaining Solution (Production Fix)

To ensure **factual accuracy** in the generated quizzes, the app performs **two sequential inference calls** to the same T5 model:

1. **Question Generation (QG):**  
   Prompt:  
→ Produces a question relevant to the provided context.

2. **Question Answering (QA):**  
Prompt: → Forces the model to produce an accurate, context-grounded answer.

This **Inference Chaining** technique converts the T5-small into a **reliable Q&A engine** while maintaining lightweight CPU-friendly performance.

---

##  Quick Start & Setup

###  Prerequisites
- Python **3.11+**
- Internet connection (for first-time model loading if not cached)

###  Install Dependencies
```bash
pip install streamlit torch transformers datasets pandas sentencepiece pypdf nltk
 
