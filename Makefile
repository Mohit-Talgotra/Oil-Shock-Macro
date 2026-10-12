# make data | var | forecast | stress | timesfm | app | test | all   (plan Section 6)
PY ?= python

.PHONY: data var forecast stress timesfm report app test all lint fmt demo app-demo clean-demo

data:
	$(PY) -m oilshock.pipeline data

var:
	$(PY) -m oilshock.pipeline var

forecast:
	$(PY) -m oilshock.pipeline forecast

stress:
	$(PY) -m oilshock.pipeline stress

timesfm:
	$(PY) -m oilshock.pipeline timesfm

report:
	$(PY) -m oilshock.pipeline report

all:
	$(PY) -m oilshock.pipeline all

app:
	streamlit run app/Home.py

test:
	$(PY) -m pytest -q

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff check --fix .
	ruff format .

# Synthetic demo (writes to ./demo, never touches data/ or artifacts/). NOT real results.
demo:
	OILSHOCK_DEMO=1 $(PY) -m oilshock.pipeline all --synthetic --fast

app-demo:
	OILSHOCK_DEMO=1 streamlit run app/Home.py

clean-demo:
	rm -rf demo
