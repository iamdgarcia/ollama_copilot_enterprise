.PHONY: streamlit

# Default target executed when no arguments are given to make.
all: help

streamlit:
	streamlit run ollama_copilot_enterprise/chatbot.py

######################
# HELP
######################

help:
	@echo '----'
	@echo 'make streamlit                        - start streamlit'