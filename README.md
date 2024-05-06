# PDF Chat Bot

This project is a Question and Answer Chat Bot with a Large Language Model (LLM). The current implementation is designed around the Brazilian 'Código de Proteção e Defesa do Consumidor'. However, the Retrieval-Augmented generation (RAG) can be easily customized and adapted to suit a variety of different scenarios.

## Technologies Used

- Python
- Langchain
- Streamlit
- HuggingFace

## Installation

Follow these steps to install and run this project:

1. Clone the repository to your local machine using Git.
2. Navigate to the project directory.
3. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

4. Export your HuggingFace token. This token is necessary to access certain features of the HuggingFace API. Replace `your-token` with your actual token:

```bash
export HF_TOKEN=your-token
```

5. Run the Streamlit app:

```bash
streamlit run app.py
```

## Usage

After running the Streamlit app, a user interface will open in your default web browser. Here, you can interact with the chat bot. Simply type your question into the input field and press enter to receive an answer.