# Jane Street neural network puzzles: reproduce both solves with `make all`.

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.PHONY: help setup torch fetch all figures test clean

help:
	@echo "make setup    create $(VENV) and install the package"
	@echo "make torch    add CPU PyTorch, only needed for the cross-checks"
	@echo "make fetch    download both puzzles from Hugging Face into ./data (1.2 GB)"
	@echo "make all      run every step and write build/results.json"
	@echo "make figures  redraw docs/img"
	@echo "make test     run the test suite"

$(BIN)/nnre:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -e '.[dev]'

setup: $(BIN)/nnre

torch: setup
	$(BIN)/pip install -q torch --index-url https://download.pytorch.org/whl/cpu

fetch: setup
	$(BIN)/nnre fetch

all: setup
	$(BIN)/nnre all

figures: setup
	$(BIN)/nnre figures

test: setup
	$(BIN)/pytest -q

clean:
	rm -rf build .pytest_cache
