CYAN := $$(tput setaf 6)
NC := $$(tput sgr0)

worktree: ## Create a worktree for a given branch. Usage: make worktree <branch>
	@echo
	@echo ${CYAN} ---- Worktree ---- ${NC}
	$(eval BRANCH := $(word 2,$(MAKECMDGOALS)))
	@if [ -z "$(BRANCH)" ]; then \
	    echo "Usage: make worktree <branch>"; exit 1; \
	fi
	git worktree add -b $(BRANCH) ../$(BRANCH)
	@echo "Worktree created successfully. Need launching command..."

worktree-remove: ## Remove a worktree for a given branch. Usage: make worktree-remove <branch>
	@echo
	@echo ${CYAN} ---- Worktree ---- ${NC}
	$(eval BRANCH := $(word 2,$(MAKECMDGOALS)))
	@if [ -z "$(BRANCH)" ]; then \
	    echo "Usage: make worktree-remove <branch>"; exit 1; \
	fi
	git worktree remove ../$(BRANCH)
	git branch -D $(BRANCH)

llms-txt: ## Generate llms.txt for any GitHub repository. Usage: make llms-txt <github-repo-url>
	@echo
	@echo ${CYAN} ---- Generating llms.txt ---- ${NC}
	@mkdir -p .docs
	$(eval REPO_URL := $(word 2,$(MAKECMDGOALS)))
	@if [ -z "$(REPO_URL)" ]; then \
		echo "Usage: make llms-txt <github-repo-url>"; \
		echo "Example: make llms-txt https://github.com/user/repo"; \
		exit 1; \
	fi
	$(eval REPO_NAME := $(shell echo "$(REPO_URL)" | sed 's|.*/||' | sed 's|\.git$$||' | sed 's|-|_|g'))
	$(eval OUTPUT_FILE := .docs/$(REPO_NAME)_llms.txt)
	@echo "Repository URL: $(REPO_URL)"
	@echo "Output file: $(OUTPUT_FILE)"
	uv run python interconnection/scripts/generate_llms_txt.py "$(REPO_URL)" -o "$(OUTPUT_FILE)"
	@echo "Generated $(OUTPUT_FILE)"

# dummy rule so "foo" (the branch) isn't treated as an unknown target
# This is a workaround for Makefile not really supporting CLI arguments
%:
	@:

# To enable help on all commands, make sure to add a ## to the end of the command
help:              ## Get help on all commands. 
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## ' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
