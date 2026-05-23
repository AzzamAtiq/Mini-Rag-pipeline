.PHONY: run validate api clean all install

# Run the full pipeline and regenerate all artifacts
run:
	python pipeline.py

# Validate all generated artifacts
validate:
	python validate.py

# Run both pipeline and validation in sequence
all: run validate

# Start the REST API server (requires uvicorn)
api:
	uvicorn api:app --reload --port 8000

# Install dependencies
install:
	pip install -r requirements.txt

# Remove generated artifacts (evaluator may do this before re-running)
clean:
	rm -rf artifacts/
	rm -f llm_calls.jsonl

# Convenience: clean then run then validate
fresh: clean run validate
