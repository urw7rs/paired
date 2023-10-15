data: $(DATA_ROOT)/aistpp

$(DATA_ROOT)/aistpp: scripts/build_datasets.py
	python scripts/build_datasets.py --root $(DATA_ROOT)

format:
	black .

lint:
	ruff check .
	ruff check --fix --select I .

test:
	pytest -v -s tests --data_root $(DATA_ROOT)
