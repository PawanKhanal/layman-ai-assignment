# Makefile for Padel Shot Classifier

.PHONY: install clean run test setup-env

install:
	pip install -r requirements.txt

run:
	python src/main.py --config configs/default.yaml

test:
	pytest tests/ -v

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf outputs/shots/*
	rm -rf outputs/reports/*

setup-env:
	python -m venv .venv
	@echo "Run: .venv\Scripts\activate on Windows"
