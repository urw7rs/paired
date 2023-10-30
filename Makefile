DATA_ROOT=datasets

$(DATA_ROOT)/aistpp: scripts/makeshards.py
	python scripts/makeshards.py download --root $(DATA_ROOT)

	python scripts/makeshards.py raw --root $(DATA_ROOT)/aistpp --split train
	python scripts/makeshards.py raw --root $(DATA_ROOT)/aistpp --split val
	python scripts/makeshards.py raw --root $(DATA_ROOT)/aistpp --split test

$(DATA_ROOT)/aistpp_sliced: scripts/makeshards.py $(DATA_ROOT)/aistpp
	python scripts/makeshards.py slice --pattern $(DATA_ROOT)"/aistpp/train/shard-{000000..000019}.tar" --output $(DATA_ROOT)/aistpp_sliced/train/shard-%06d.tar
	python scripts/makeshards.py slice --pattern $(DATA_ROOT)/aistpp/val/shard-000000.tar --output $(DATA_ROOT)/aistpp_sliced/val/shard-%06d.tar
	python scripts/makeshards.py slice --pattern $(DATA_ROOT)/aistpp/test/shard-000000.tar --output $(DATA_ROOT)/aistpp_sliced/test/shard-%06d.tar

data: $(DATA_ROOT)/aistpp $(DATA_ROOT)/aistpp_sliced

format:
	black .

lint:
	ruff check --fix .

test:
	pytest -n auto tests --data_root $(DATA_ROOT)

clean: $(DATA_ROOT)
	rm -r $(DATA_ROOT)/aistpp*
