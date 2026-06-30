.PHONY: all ingest run eval clean help

PYTHON := .venv/bin/python
N ?= 10        # number of contracts for subset runs

help:
	@echo "Targets:"
	@echo "  make ingest [N=10]   Pull CUAD subset and chunk contracts"
	@echo "  make run             Run agent over all ingested contracts"
	@echo "  make eval            Score results and write report to reports/"
	@echo "  make all [N=10]      Full pipeline end-to-end"
	@echo "  make clean           Remove processed data and reports"

ingest:
	$(PYTHON) cli.py ingest --subset $(N)

run:
	$(PYTHON) cli.py run

eval:
	$(PYTHON) cli.py eval

all:
	$(PYTHON) cli.py ingest --subset $(N)
	$(PYTHON) cli.py run
	$(PYTHON) cli.py eval

clean:
	rm -rf data/processed/* data/raw/* reports/*
	rm -rf .chroma/

install:
	.venv/bin/pip install -r requirements.txt