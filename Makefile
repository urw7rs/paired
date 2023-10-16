data: $(DATA_ROOT)/aistpp

$(DATA_ROOT)/aistpp: scripts/build_datasets/aistpp.py
	python scripts/build_datasets/aistpp.py download --root $(DATA_ROOT)
	python scripts/build_datasets/aistpp.py build --root $(DATA_ROOT)/aistpp

format:
	black .

lint:
	ruff check --fix .

test:
	pytest tests --data_root $(DATA_ROOT)
