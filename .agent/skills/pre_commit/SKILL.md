---
name: Pre-Commit Documenter
description: Runs the linter, compares workspace state to the last pushed commit, and updates README.md and architecture diagrams.
---

# Pre-Commit Documenter

This skill automates the workflow for wrapping up a set of changes before making a git commit. It ensures that the code is clean and that the project documentation reflects the latest architectural and performance modifications.

## How to use this skill

When the user asks you to run the pre-commit workflow or prepare a commit, follow these precise steps:

1. **Run the Linter**:
   Execute the codebase linter to ensure formatting is consistent and anti-patterns are resolved:
   ```bash
   bash .agent/skills/linting/scripts/run_lint.sh
   ```

2. **Analyze the Delta**:
   Compare the current workspace against the last pushed commit (the upstream tracking branch) to figure out what was recently built:
   ```bash
   git diff @{u}
   ```
   *(If `@{u}` fails, fall back to `git log -p -1` or `git diff origin/main` to understand what was changed recently).*

3. **Update Documentation (`README.md` & `architecture.mmd`)**:
   - Analyze the diff and map the changes to the `README.md`. Update any KPIs (tokens/sec), add new feature bullets, and describe any new fused operators or architectural bypasses.
   - Check if the system architecture changed. If it did, edit `architecture.mmd` to keep the visual flow accurate.

4. **Regenerate Architecture Diagram**:
   If `architecture.mmd` was modified (or if the user explicitly requested it), you **must** re-render the PNG diagram. Use `mermaid-cli` with a safe Puppeteer configuration to avoid sandbox crashes natively on Linux:
   ```bash
   echo '{"args": ["--no-sandbox"]}' > puppeteer-config.json
   npx -y @mermaid-js/mermaid-cli -i architecture.mmd -o tiny_llm_architecture.png -b transparent -p puppeteer-config.json
   rm puppeteer-config.json
   ```

5. **Stage and Notify**:
   - Inform the user of what documentation changes you extrapolated from the codebase.
   - Ask for confirmation before natively committing!
