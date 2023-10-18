data: $(DATA_ROOT)/aistpp

$(DATA_ROOT)/aistpp: scripts/build_datasets/aistpp.py
	python scripts/build_datasets/aistpp.py download --root $(DATA_ROOT)

format:
	black .

lint:
	ruff check --fix .

test:
	pytest -n auto tests --data_root $(DATA_ROOT)
