# Reproduce the headline results. `make repro` is the <15-minute path.
PY ?= python
export PYTHONPATH := src
export PYTHONIOENCODING := utf-8

.PHONY: help setup data prep taxonomy prelabel golden eval agreement failures bias repro status resume all clean-cache

help:
	@echo "make setup      install python deps"
	@echo "make data       download the Kaggle corpus (~169MB, needs ~/.kaggle/kaggle.json)"
	@echo "make prep       rebuild brand cases + train/eval splits"
	@echo "make repro      HEADLINE: eval + judge agreement + failure analysis (uses shipped cache)"
	@echo "make all        full rebuild from raw data, including LLM pre-labelling"
	@echo "make resume     continue the pipeline across daily free-tier quota windows"
	@echo "make status     show pipeline progress + live per-model quota"
	@echo "make watch      run to completion, retrying across free-tier quota windows"

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip

prep:
	$(PY) scripts/prepare_brand.py British_Airways
	$(PY) -m support_agent.data.dataset

taxonomy:
	$(PY) scripts/cluster_messages.py

prelabel:
	$(PY) scripts/prelabel_golden.py

golden:
	$(PY) scripts/build_golden.py

eval:
	$(PY) scripts/run_eval.py --ablation --judge-n 60

agreement:
	-$(PY) scripts/judge_agreement.py

failures:
	$(PY) scripts/failure_analysis.py

bias:
	-$(PY) scripts/judge_bias.py --n 80

repro: eval failures bias agreement
	@echo "--- headline table ---"
	@cat reports/results/RESULTS.md

all: prep taxonomy prelabel golden repro

clean-cache:
	rm -rf llm_cache

status:
	@$(PY) scripts/run_status.py

resume:
	@$(PY) scripts/resume.py

watch:
	@$(PY) scripts/resume.py --watch --every 30
