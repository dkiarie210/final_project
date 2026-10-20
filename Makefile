PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-final.txt

train:
	$(PYTHON) run_model_pipeline.py

validate:
	$(PYTHON) run_pre_presentation_validations.py

test:
	pytest -q

run:
	docker compose up --build -d

retrain:
	docker compose --profile training run --rm pipeline

stop:
	docker compose down

logs:
	docker compose logs -f
