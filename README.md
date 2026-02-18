# Yonemura_Research

## Docs

- [`docs/interim.pdf`](docs/thesis.pdf): 論文
- [`docs/abstract.pdf`](docs/abstract.pdf): 概要資料
- [`docs/interim.pdf`](docs/interim.pdf): 中間発表資料
- [`docs/slides.pdf`](docs/interim.pdf): 最終発表資料

## Setup

```bash
# Clone the repository
git clone https://github.com/59GauthierLab/Yonemura_Research.git
cd Yonemura_Research

# Install dependencies
uv sync

# Install the project in editable mode (for development/experiments)
uv pip install -e .

# Set up the reproducible environment
task set-repro-env

# Define environment variables
cp .env.example .env # Edit .env as needed
```
