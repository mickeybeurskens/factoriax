.PHONY: install-hooks

install-hooks:
	ln -sf ../../scripts/hooks/pre-commit .git/hooks/pre-commit
	ln -sf ../../scripts/hooks/post-commit .git/hooks/post-commit
	@echo "Hooks installed."
