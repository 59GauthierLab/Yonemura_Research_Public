# Yonemura_Research

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
