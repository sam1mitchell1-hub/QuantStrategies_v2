## Installation

### Prerequisites

- Python 3.8+ (dev uses 3.11)
- pip (or `uv` if preferred)

### Install with pip

```bash
pip install --upgrade pip
pip install .
```

Optional extras:

- Notebook/plotting tools:

```bash
pip install .[notebook]
```

- Test dependencies:

```bash
pip install .[test]
```

### Install with uv (optional)

```bash
uv pip install .[notebook]
```

### Environment Variables

- `MARKETSTACK_API_KEY`: required only if using `MarketstackClient`

### Docker

Build image:

```bash
docker build -t quant-strategies:latest .
```

Run with a mounted data directory and logs:

```bash
docker run --rm -it \
  -v "$PWD/data":/app/data \
  -v "$PWD/logs":/app/logs \
  -e MARKETSTACK_API_KEY="$MARKETSTACK_API_KEY" \
  quant-strategies:latest
```

The default container command runs `supercronic` with the provided `crontab` to periodically fetch options chains.

