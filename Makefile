# Variables
PYTHON = python
PIP = pip
APP = app.py
PORT = 3000
REQUIREMENTS = requirements.txt

# Default target
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make install       Install dependencies"
	@echo "  make run           Run the Flask app"
	@echo "  make clean         Clean temporary files"

# Install dependencies
.PHONY: install
install:
	$(PIP) install -r $(REQUIREMENTS)

# Run the Flask app
.PHONY: run
run:
	$(PYTHON) $(APP) --port=$(PORT)

# Clean up temporary files
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

