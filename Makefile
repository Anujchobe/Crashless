PY ?= .venv/bin/python
test:        ; $(PY) -m pytest -q
test-fast:   ; $(PY) -m pytest -q -m "not slow"
benchmark:   ; $(PY) scripts/benchmark.py main --seeds 0 1 2 3 4 && $(PY) scripts/report.py main
ablations:   ; $(PY) scripts/benchmark.py smoothness && $(PY) scripts/report.py smoothness && \
               $(PY) scripts/benchmark.py shaping && $(PY) scripts/report.py shaping
demo:        ; $(PY) scripts/demo.py --ppo runs/main/final/ppo_s0 --sac runs/main/final/sac_s0
.PHONY: test test-fast benchmark ablations demo
