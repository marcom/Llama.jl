# This is not a true makefile, just a collection of convenient scripts

# Variables
model ?= <provide-model-path>

default: help

format:
	# assumes you have JuliaFormatter installed in your global env / somewhere on LOAD_PATH
	julia -e 'using JuliaFormatter; format(".")'

server:
	# starts the server with the specified model
	julia --project=@. -t auto -e 'using Llama; Llama.run_server(; model = "$(model)")'

help:
	echo "make help - show this help"
	echo "make format - format the code"
	echo "make server model=models/some-model.gguf - run the server with the specified model"