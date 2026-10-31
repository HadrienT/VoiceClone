# Instructions for Claude Code

- **Continuous merge allowed**: the repository owner runs `git pull` on `main`. Once work is
  tested (`pytest` green), push the working branch, open a PR to `main` and merge it without
  waiting for confirmation.
- **Language: English by default.** Everything you produce is in English: code, identifiers, comments, commit messages, PRs, issues, documentation and interface text. The only exception is the conversation: reply in the language the maintainer writes in, so in French when they write in French. The existing interface is in French.
- Before pushing: `pytest`, `node --check web/*.js`, and a check in Chromium for interface
  changes (fake engine: `tests/fake_engine.py`).
- Improvements are tracked in `WORKPLAN.md` (☑ tested, ⚠ to validate on the GPU server).
