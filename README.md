# generativeai-dev

This repository contains a collection of examples for building conversational AI and LLM-powered applications using [Chainlit](https://docs.chainlit.io).

## 🚀 Overview

This project provides several sample scripts demonstrating how to integrate Chainlit with various powerful tools from the LLM ecosystem. The examples include:

*   A basic chat interface with an LLM.
*   An advanced agent-like application using **[LangGraph](https://langchain-ai.github.io/langgraph/)** to create stateful, multi-step chains.
*   An integration with **[Ollama](https://ollama.com/)** to run and interact with open-source LLMs locally on your machine.

## ✨ Features

*   Interactive chat interface powered by Chainlit.
*   Examples of building stateful agents with **LangGraph**.
*   Demonstration of using local LLMs via **Ollama**.

## 📋 Requirements

*   Python 3.8+
*   Chainlit
*   LangChain (`langchain`, `langchain-openai`, etc.)
*   LangGraph (`langgraph`)
*   Ollama (`ollama`)

For local model usage, you also need to have the Ollama application installed and the desired models pulled (e.g., `ollama pull llama3`).

## 📦 Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management. If you don't have it installed, follow the instructions on their website.

1.  Clone the repository:
    ```bash
    git clone https://github.com/tanaka-r/generativeai-dev.git
    cd generativeai-dev
    ```

2.  Create a virtual environment and install the dependencies from `pyproject.toml` using `uv`:
    ```bash
    # Create and activate a virtual environment
    uv venv
    source .venv/bin/activate

    # Install dependencies defined in pyproject.toml
    uv pip install -e .
    ```

## ▶️ Running the Examples

To run a specific example, use the `chainlit run` command followed by the path to the script. The `-w` flag enables auto-reloading when you make changes to the code.

```bash
# Example for a generic app
chainlit run app/chainlit_app.py -w

```

*(Replace `app.py` with the name of your main Python script if it's different.)*

Then, open your web browser and navigate to `http://localhost:8000`.
