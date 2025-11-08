# Development Guide

This guide covers the development workflow for the Machine Vision Backend.

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Start development server
make dev

# 3. Open API documentation
# Browser: http://localhost:8000/docs
```

## Project Structure

```
backend/
├── python-backend/          # Python FastAPI application
│   ├── .venv/              # Virtual environment (created by make install)
│   ├── main.py             # FastAPI app entry point
│   ├── config.py           # Configuration management
│   ├── api/                # REST API layer
│   ├── core/               # Core infrastructure (managers, utils)
│   ├── services/           # Business logic
│   ├── vision/             # Computer vision algorithms
│   ├── schemas/            # Pydantic models
│   ├── tests/              # Test suite
│   ├── requirements.txt    # Production dependencies
│   └── requirements-dev.txt # Development dependencies
├── Makefile                # Development commands
└── DEVELOPMENT.md          # This file
```

## Available Commands

### Setup Commands

#### `make install`
Creates a virtual environment at `python-backend/.venv` and installs all dependencies.

```bash
make install
```

What it does:
- Creates `.venv` directory
- Upgrades pip
- Installs all dependencies from `requirements-dev.txt` (includes production deps)

#### `make setup-hooks`
Installs pre-commit hooks for automatic code quality checks.

```bash
make setup-hooks
```

Pre-commit hooks run automatically on `git commit`:
- **black**: Code formatting (line-length=100)
- **isort**: Import sorting (black-compatible)
- **flake8**: Linting (ignore E203, W503)
- **trailing-whitespace**, **end-of-file-fixer**: Cleanup
- **check-yaml**, **check-merge-conflict**: Safety checks

### Development Commands

#### `make dev`
Starts the development server with auto-reload.

```bash
make dev
```

Features:
- Auto-reloads on `.py` file changes
- Uses `config.dev.yaml` (debug mode, verbose logging)
- Accessible at `http://localhost:8000`
- API docs at `http://localhost:8000/docs`
- Alternative docs at `http://localhost:8000/redoc`

Press `Ctrl+C` to stop.

### Testing Commands

#### `make test`
Runs all tests with verbose output.

```bash
make test
```

#### `make test-fast`
Runs only fast tests, skipping tests marked as `@pytest.mark.slow`.

```bash
make test-fast
```

Use this for quick feedback during development.

#### `make test-watch`
Starts test watch mode - automatically reruns tests when files change.

```bash
make test-watch
```

Requires `pytest-watch` (included in `requirements-dev.txt`). Perfect for Test-Driven Development (TDD).

#### `make cov`
Runs tests with coverage report.

```bash
make cov
```

Generates:
- Terminal coverage summary
- HTML coverage report at `python-backend/htmlcov/index.html`
- Automatically opens the HTML report in your browser (if `xdg-open` is available)

### Code Quality Commands

#### `make format`
Formats all Python code using black and isort.

```bash
make format
```

Run this before committing if you haven't set up pre-commit hooks.

#### `make lint`
Lints Python code using flake8.

```bash
make lint
```

Configuration:
- Max line length: 100
- Ignored rules: E203 (whitespace before ':'), W503 (line break before binary operator)

#### `make check`
Runs lint + all tests - a complete pre-push check.

```bash
make check
```

Use this before pushing to ensure everything passes CI.

### Cleanup Commands

#### `make clean`
Removes build artifacts and caches.

```bash
make clean
```

Removes:
- `__pycache__` directories
- `.pyc` files
- `.pytest_cache`
- `htmlcov` (coverage reports)
- `.coverage` (coverage data)
- Runtime logs (`var/log`, `var/run`)

#### `make clean-all`
Deep cleanup - also removes the virtual environment.

```bash
make clean-all
```

Use this to start completely fresh. You'll need to run `make install` again.

### Packaging Commands

#### `make deb`
Builds a Debian package.

```bash
make deb
```

See `packaging/README.md` for details on the DEB package structure.

#### `make deb-clean`
Cleans DEB build artifacts.

```bash
make deb-clean
```

## Development Workflow

### Initial Setup

```bash
# Clone the repository
cd /home/cnc/machine-vision/backend

# Install dependencies
make install

# Set up pre-commit hooks (optional but recommended)
make setup-hooks

# Start development server
make dev
```

### Daily Development

```bash
# Start dev server in one terminal
make dev

# In another terminal, run tests in watch mode
make test-watch

# Make changes to code...
# Tests auto-run, server auto-reloads
```

### Before Committing

```bash
# Format code
make format

# Run checks
make check

# Or just commit - pre-commit hooks will run automatically
git commit -m "Your message"
```

### Before Pushing

```bash
# Run full check
make check

# If all passes, push
git push
```

## Testing

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── api/                     # API endpoint tests
│   ├── conftest.py         # API-specific fixtures
│   ├── test_vision_api.py
│   ├── test_camera_api.py
│   └── ...
├── services/                # Service layer tests
│   ├── test_vision_service.py
│   └── ...
├── core/                    # Core infrastructure tests
│   └── test_image_utils.py
└── vision/                  # Vision algorithm tests
    ├── test_edge_detection.py
    ├── test_color_detection.py
    └── ...
```

### Writing Tests

Tests use pytest with async support and fixtures from `conftest.py`:

```python
# Example test
def test_edge_detection(vision_service, test_image, image_manager):
    # Store test image
    image_id = image_manager.store(test_image)

    # Run detection
    result = await vision_service.edge_detect(image_id, method="canny")

    # Assertions
    assert len(result.objects) > 0
    assert result.processing_time_ms > 0
```

Available fixtures:
- `test_image`, `test_template`: Synthetic test images
- `image_manager`, `camera_manager`, `template_manager`: Real manager instances
- `vision_service`, `camera_service`, `image_service`: Service instances
- `mock_*`: Mock managers for unit tests

### Test Markers

Use markers to categorize tests:

```python
@pytest.mark.slow
def test_slow_operation():
    ...

@pytest.mark.integration
def test_integration():
    ...
```

Available markers:
- `slow`: Tests that take > 1 second
- `integration`: Integration tests requiring multiple components

Run specific markers:
```bash
# Skip slow tests
pytest -m "not slow" tests/

# Run only integration tests
pytest -m integration tests/
```

## Configuration

### Development Configuration

Development uses `python-backend/config.dev.yaml`:

```yaml
# Debug mode
debug: true
log_level: DEBUG

# Larger thumbnails for inspection
image:
  thumbnail_width: 400

# Verbose camera logging
camera:
  default_camera: "test"  # Use test camera by default
```

### Production Configuration

Production uses environment variables or `python-backend/config.yaml`.

Configuration precedence:
1. Environment variables (e.g., `MV_IMAGE__MAX_IMAGES=100`)
2. YAML file specified by `MV_CONFIG_FILE` env var
3. Code defaults

### Environment Variables

All config values can be overridden via environment variables with `MV_` prefix:

```bash
# Set max images
export MV_IMAGE__MAX_IMAGES=200

# Set log level
export MV_LOG_LEVEL=INFO

# Nested config uses double underscores
export MV_CAMERA__DEFAULT_CAMERA=usb
```

## Debugging

### VS Code Debugging

Launch configurations are available in `.vscode/launch.json`:

1. **Python: FastAPI** - Debug the development server
2. **Python: Current File** - Debug the current Python file
3. **Python: Pytest** - Debug tests

Set breakpoints and press F5 to start debugging.

### API Debugging

Use the interactive API documentation:

1. Start dev server: `make dev`
2. Open http://localhost:8000/docs
3. Try out endpoints directly in the browser
4. See request/response schemas and examples

### Logging

Development mode logs to console with DEBUG level. Look for:
- Request/response details
- Image processing steps
- Camera connection status
- Configuration values

### Common Issues

**"Error: venv not found"**
- Run `make install` first

**"Module not found" errors**
- Make sure you're running commands through make (which activates venv)
- Or manually activate: `source python-backend/.venv/bin/activate`

**Tests failing with image not found**
- Images are stored in shared memory with LRU eviction
- Increase `max_images` in config if needed
- Use fresh test fixtures for each test

**Port 8000 already in use**
- Stop any running instances
- Or change port: `PORT=8001 make dev`

**Pre-commit hooks failing**
- Run `make format` to fix formatting issues
- Run `make lint` to see linting errors
- Fix errors manually, then commit again

## Production Deployment

For production deployment via systemd and DEB package, see:
- `packaging/README.md` - Package building
- `packaging/INSTALL.md` - Installation guide
- `debian/machinevision.service` - Systemd service unit

The development workflow (Makefile, scripts) is **only for local development**. Production uses systemd directly.

## Additional Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **OpenCV Python**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- **Pytest Docs**: https://docs.pytest.org/
- **Black Formatter**: https://black.readthedocs.io/
- **Pre-commit**: https://pre-commit.com/

## Getting Help

- Check API docs: http://localhost:8000/docs
- Read `CLAUDE.md` for architecture details
- Read `backend/CLAUDE.md` for detailed component documentation
- Check test files for usage examples
- Run `make help` to see all available commands
