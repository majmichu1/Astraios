# Contributing to Astraios

Thank you for your interest in contributing! Astraios is a community-driven project to build professional astrophotography software that's **free and open source**.

## Getting Started

### 1. Set up the development environment

```bash
# Clone the repository
git clone https://github.com/majmichu1/astraios.git
cd astraios

# Install dependencies
poetry install

# Install dev dependencies (tests, linters)
poetry install --with dev
```

### 2. Run the application

```bash
poetry run astraios
# or
poetry run python -m astraios
```

### 3. Run tests

```bash
poetry run pytest  # 729+ tests
poetry run ruff check .  # Lint
poetry run mypy astraios  # Type check
```

## How to Contribute

### Reporting Bugs
- Open an issue at [GitHub Issues](https://github.com/majmichu1/astraios/issues)
- Include: OS, Python version, Astraios version, steps to reproduce
- Attach screenshots if possible

### Suggesting Features
- Open an issue with the `enhancement` label
- Describe the use case and why it's useful
- Bonus: propose a rough implementation approach

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Run the full test suite: `poetry run pytest`
5. Run the linter: `poetry run ruff check .`
6. Commit with a clear message
7. Push and open a Pull Request

### Code Style
- **Line length**: 100 characters
- **Type hints**: Expected on all public functions
- **Imports**: Sorted by Ruff's `I` rule (isort-compatible)
- **Docstrings**: Google-style for public APIs
- **Tests**: Mirror the source structure in `tests/`

### GPU Acceleration
Astraios uses PyTorch for GPU acceleration. All GPU code MUST go through `get_device_manager()` in `astraios/core/device_manager.py`. Never call `torch.cuda.*` directly elsewhere.

## AI Model Training

Astraios includes self-supervised AI models (Noise2Self for denoising). To train models:

```bash
# Place FITS files in astro_data/
mkdir -p astro_data
cp /path/to/your/*.fits astro_data/

# Run training
poetry run python scripts/train_denoise_model.py --input astro_data --epochs 30
```

## License
By contributing, you agree that your contributions will be licensed under the GPL v3.

## Buy Me a Coffee
If you'd like to support the project financially:
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-donate-yellow)](https://buymeacoffee.com/majmichu)
