🧠 Smart Study Assistant: The AI Content Accelerator
🚀 Accelerate Learning: From Document to Knowledge in Seconds
The Smart Study Assistant is a production-ready web application built to solve the challenge of rapid information processing and knowledge extraction from unstructured academic or enterprise documents. It transforms lengthy PDF and TXT files into actionable learning modules using a fine-tuned Text-to-Text Transfer Transformer (T5) model, deployed on Streamlit.

This project showcases robust multi-task NLP deployment, advanced inference chaining, and critical error handling necessary for real-world document processing.

✨ Core Product Features
Feature	User Benefit	Technical Driver
Comprehensive Note Generation	Converts 50,000+ word documents into concise, flowing study notes.	Multi-chunk summarization pipeline and post-processing heuristics.
Factual Quiz Generation	Generates verified, fact-checked short-answer questions.	QG → QA Inference Chaining (Sequential T5 calls for Question → Answer).
Customized Learning Paths	Adjusts question complexity and type (QG or MCQ) based on user-selected difficulty (Easy, Medium, Hard).	Dynamic input chunking based on difficulty level (e.g., smaller chunks for 'Hard').
Broad Compatibility	Accepts common document formats (.pdf, .txt).	PyPDF and robust text extraction logic.
🛠️ Deployment & Architecture
This application utilizes a unified architecture where a single T5-small model serves three distinct NLP tasks.

Technology Stack
Layer	Technology	Purpose
Interface	Streamlit	Fast, responsive Python-native application layer.
Model	T5-small (Hugging Face)	Encoder-Decoder core fine-tuned for sequence-to-sequence tasks.
Backend	PyTorch, transformers	Model loading and efficient CPU/GPU inference.
Utilities	pypdf, nltk	Document parsing, text cleaning, and sentence tokenization.
The QG → QA Chaining Solution (Production Fix)
The key innovation to overcome the limitations of the quick-trained T5-small model is the Inference Chain deployed in the Streamlit backend.

To ensure factual accuracy in the generated quizzes, the app does not rely on a single question-generation call. Instead, it performs two sequential model calls:

Question Generation (QG): The model is prompted (generate question: context: [Chunk] answer: key concepts) to create a question.

Question Answering (QA): The generated Question and the original [Context] are immediately fed back into the same T5 model (using a question and context: prefix) to force the model to output the precise, factual Answer.

This technique transforms the brittle T5-small into a reliable Q&A engine for specific contexts.

💻 Quick Start and Setup
Prerequisites
Ensure you have Python 3.11+ installed.

Install Dependencies:

Bash

pip install streamlit torch transformers datasets pandas sentencepiece pypdf nltk
Model Availability:
The application requires the fine-tuned T5 model and tokenizer files. Ensure the directory ./final_notes_quiz_model is located in the project root alongside notenquiz_app.py.

Running the Application
Navigate to the project directory in your terminal and run:

Bash

streamlit run notenquiz_app.py
The application will automatically load in your browser at http://localhost:8501.

Note: Since the model is running on the CPU, the first generation might take 20–30 seconds for full document processing. Subsequent generations will be faster.
