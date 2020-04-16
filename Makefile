init:
    pip install -r src/requirements.txt

test:
    py.test tests

.PHONY: init test