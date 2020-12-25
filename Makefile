export CWD := $(shell pwd)

all: setup_project install_nlp_tools

setup_project:
	pip install -e .

install_nlp_tools:
	echo "Installing Greek Language support for spacy"
	mkdir -p nlp_tools
	wget https://github.com/eellak/gsoc2018-spacy/raw/6212c56f94ca3926b0959ddf9cee39df28e1c5a8/spacy/lang/el/models/el_core_web_sm-1.0.0.tar.gz -P $(CWD)/nlp_tools/
	pip install nlp_tools/el_core_web_sm-1.0.0.tar.gz
	rm -rf nlp_tools
