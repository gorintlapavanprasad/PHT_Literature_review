.PHONY: test report ui docker
test:    ; python -m pytest -q
report:  ; cd paper_annotation && python -m evaluation.report --out ../reports $(if $(HUMAN),--human $(HUMAN),)
ui:      ; cd pht_ui && python -m uvicorn backend.server:app --port 8765
docker:  ; docker build -t pht-review .
