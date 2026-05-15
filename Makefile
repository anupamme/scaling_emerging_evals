.PHONY: test lint format figures writeup

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

figures:
	uv run python scripts/generate_figures.py

writeup:
	@if command -v pandoc >/dev/null 2>&1; then \
		pandoc WRITEUP.md -o WRITEUP.pdf; \
		echo "Generated WRITEUP.pdf"; \
	else \
		echo "pandoc not found — opening WRITEUP.md directly"; \
		open WRITEUP.md 2>/dev/null || xdg-open WRITEUP.md 2>/dev/null || echo "Open WRITEUP.md manually"; \
	fi
