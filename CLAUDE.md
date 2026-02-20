# Campsite

AI-powered analysis assistant for Jupyter notebooks. Provides an interactive chat interface for hypothesis generation and analysis code generation.

## Quick Reference

```bash
# Install
pip install -e .
npm install

# Build frontend
node esbuild.config.js    # Bundles src/campsite.ts → static/campsite.js

# Test
pytest tests/test_campsite_lib.py -v

# Format
black campsite/            # Line length 88, Python 3.10 target
```

## Architecture

- **campsite/campsite.py** — Main AnyWidget Jupyter widget
- **campsite/campsite_server.py** — Flask API server (thread-safe singleton)
- **campsite/campsite_lib/** — Core analysis library:
  - `ir_ast.py` — Hypothesis IR AST dataclasses
  - `ir_parser.py` — Lark grammar parser with partial recovery
  - `si_checkers.py` — Semantic invariant validation framework
  - `nl_extractors.py` — Natural language field extractors
  - `hypothesis_node.py` — Hypothesis node handling
  - `.analysis_graph.py` — LangGraph analysis state machine
  - `grammar_specs/` — Text-based grammar files (v1–v3)
- **campsite/src/campsite.ts** — TypeScript frontend (D3, anime.js, marked)
- **campsite/static/** — Compiled JS bundle

## Conventions

- Full type annotations in Python and TypeScript
- `@dataclass` for data structures, Pydantic for API/state models
- ABC pattern for checker classes (Strategy pattern)
- Private methods: `_underscore_prefix`
- Tests: class-based organization with descriptive `test_*` method names
