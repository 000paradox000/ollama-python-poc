requirements.install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

run:
	python main.py

make-dummy-question:
	ollama run gemma3:1b "is the creator of dragon ball z alive"

pre-commit.run:
	pre-commit run --all-files
	pre-commit run --all-files

pre-commit.update:
	pre-commit autoupdate
