## Ollama-copilot-enterprise

This project is a codebase chatbot that allows users to interact with their codebase using [Ollama](https://ollama.com/). It utilizes a Streamlit app to provide a user-friendly chat interface.

### Features
Users can enter the name of their GitHub repository.
The repository is then cloned, chunked and embedded. Langchain is used to build a QA retriever so users can chat with their code. 
The chat interface allows users to ask questions and interact with the codebase.
Usage

To use this codebase chatbot, follow these steps:

1. Clone the repository:

```git clone https://github.com/example/repository.git```

2. Install the required dependencies:

```poetry install ```

3. Run the Streamlit app:

```	streamlit run ollama_copilot_enterprise/chatbot.py```

or 
```make streamlit```


Access the chat interface by opening your web browser and navigating to http://localhost:8501.

Enter the name of your GitHub repository in the provided input fields.

The codebase will be chunked and embedded, and the chat interface will be displayed.

Ask questions or provide instructions using natural language, and the chatbot will respond accordingly.

### Limitations
* The codebase chatbot relies on Ollama Language Model and its capabilities.
* Large codebases or repositories with complex structures may take longer to chunk and embed.
* The accuracy and quality of responses depend on the accuracy of the language model and the code embeddings.

### Future Improvements
* Integrate with external tools and services to provide more advanced codebase analysis and insights.
* Allow to select the model from the UI.

### Contributing
Contributions to this codebase chatbot project are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.
