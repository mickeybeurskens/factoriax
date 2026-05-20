.PHONY: install-hooks test test-fast-fail

install-hooks:
	ln -sf ../../scripts/hooks/pre-commit .git/hooks/pre-commit
	@echo "Hooks installed."

# Full suite, no coverage, run through every failure. The default
# pytest behaviour after Task 2.7 dropped ``-x`` from addopts.
test:
	uv run pytest --no-cov

# Stop at the first failing test — handy when iterating on a known-
# broken test that runs deep in the suite. Mirrors the pre-commit
# hook's ``-x`` behaviour without forcing every ``pytest`` invocation
# to share it.
test-fast-fail:
	uv run pytest -x --no-cov
