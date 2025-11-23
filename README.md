# Machine Vision Backend

**Industrial-grade computer vision REST API powered by FastAPI and OpenCV**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)](https://opencv.org/)

## Overview

This is the **REST API backend** for the Machine Vision system - a comprehensive solution for industrial computer vision applications inspired by platforms like Keyence and Cognex.

### Two-Repository Architecture

The Machine Vision system consists of **two required components**:

- **This Repository**: FastAPI REST API Backend
  - Camera management (USB, IP, Test mode)
  - Computer vision algorithms
  - Image processing and template management
  - RESTful API for all vision operations
  - **Designed to be controlled via Node-RED**

- **Frontend Repository** (Required): [node-red-contrib-machine-vision](https://github.com/mat100/node-red-contrib-machine-vision)
  - **Primary user interface** - Node-RED custom nodes
  - Visual flow-based programming interface
  - Pre-built nodes for camera, vision processing, and analysis
  - Integration with industrial automation systems (PLCs, SCADA, etc.)

**The backend is designed to work with Node-RED** as its primary interface. Together they provide a complete, visual programming environment for machine vision workflows.

### Node-RED as Primary Interface

This backend is **specifically designed** to be controlled through the Node-RED custom nodes. The workflow is:

1. Install and start this backend (provides vision processing engine)
2. Install [node-red-contrib-machine-vision](https://github.com/mat100/node-red-contrib-machine-vision) in Node-RED
3. Use Node-RED flows to build your machine vision application

**Node-RED is the user interface** for this backend.

## Key Features

### Camera Management
- **Multi-camera support**: USB webcams, IP cameras (RTSP/HTTP), Test mode
- **Live streaming**: MJPEG stream for preview and monitoring
- **Flexible capture**: Single frame capture with configurable parameters

### Vision Algorithms

#### Edge Detection
- **5 methods**: Canny, Sobel, Laplacian, Prewitt, Scharr
- Configurable thresholds and parameters
- ROI (Region of Interest) support

#### Color Detection
- **11 predefined colors**: Red, green, blue, yellow, orange, purple, pink, cyan, brown, black, white
- **Custom HSV ranges**: Define your own color detection criteria
- Multi-object detection with confidence scores

#### ArUco Marker Detection
- Multiple dictionary support (4x4, 5x5, 6x6, 7x7)
- Pose estimation
- ID-based tracking

#### Template Matching
- **Basic** (`/template-match`):
  - Normalized cross-correlation
  - Learn templates from captured images
  - Upload custom template images
  - Multi-scale matching
- **Advanced** (`/advanced-template-match`):
  - **Rotation-invariant matching**: Detect objects at any angle (±180°)
  - **Multi-instance detection**: Find all occurrences in a single pass
  - **Non-Maximum Suppression (NMS)**: Intelligent overlap filtering
  - **Configurable search**: Custom rotation ranges and step sizes
  - Industrial-grade performance for batch inspection

#### Rotation Detection
- **3 methods**: PCA (Principal Component Analysis), MinAreaRect, Image Moments
- Angle measurement and object orientation
- Centroid calculation

### Advanced Architecture
- **Shared memory**: Zero-copy image storage with LRU cache
- **Async/await**: Non-blocking I/O throughout
- **Type safety**: Pydantic schemas for all data
- **Auto-documentation**: Interactive API docs at `/docs`

## Quick Start

### Complete System Setup (Recommended)

To use the Machine Vision system, you need **both components**:

1. **This backend** (vision processing engine)
2. **Node-RED with custom nodes** (user interface)

#### Step 1: Install Backend

**Prerequisites:**
- **Python 3.9+** (Python 3.11 recommended for development)
- **pip** and **virtualenv** (or system package manager for production)
- **Optional**: Camera hardware (USB webcam or IP camera)

**Development Installation:**

```bash
# Clone the repository
git clone <your-repo-url>
cd backend

# Install dependencies and setup development environment
make install

# Start development server (with auto-reload)
make dev
```

The backend API will be available at `http://localhost:8000`

**Verify backend is running:**
```bash
curl http://localhost:8000/api/system/health
# Should return: {"status":"healthy"}
```

#### Step 2: Install Node-RED Frontend

Once the backend is running, install the Node-RED custom nodes:

```bash
# In your Node-RED user directory (usually ~/.node-red)
cd ~/.node-red
npm install node-red-contrib-machine-vision

# Restart Node-RED
node-red-restart  # or restart Node-RED service
```

**Access Node-RED:**
- Open Node-RED in browser: http://localhost:1880
- Find "Machine Vision" nodes in the palette
- Drag and drop nodes to create your vision workflow

See the [Node-RED package documentation](https://github.com/mat100/node-red-contrib-machine-vision) for detailed setup and usage.

### Production Installation

For production deployment on Debian/Ubuntu systems:

**Backend:**
```bash
# Build DEB package
cd packaging
./build.sh

# Install package
sudo dpkg -i machinevision_1.0.0_all.deb

# Manage service
machinevision start
machinevision status
machinevision logs
```

**Node-RED nodes:**
```bash
cd ~/.node-red  # or your Node-RED directory
npm install node-red-contrib-machine-vision
```

See [packaging/INSTALL.md](packaging/INSTALL.md) for detailed production deployment instructions.

### Using the System with Node-RED

**Recommended workflow:**

1. **Start backend**: Backend must be running before using Node-RED nodes
2. **Open Node-RED**: Access the Node-RED editor
3. **Create flow**: Drag Machine Vision nodes into your flow
4. **Configure nodes**: Point nodes to backend URL (default: http://localhost:8000)
5. **Deploy and run**: Deploy your flow and start processing

**Example Node-RED flow:**
```
[Camera Connect] → [Capture Image] → [Edge Detection] → [Debug]
```

See [Node-RED documentation](https://github.com/mat100/node-red-contrib-machine-vision) for examples and tutorials.

## System Architecture

### High-Level Architecture

```
┌──────┐      ┌─────────────┐      ┌─────────────┐      ┌──────────┐
│ User │─────▶│  Node-RED   │─────▶│   Backend   │─────▶│  Vision  │
└──────┘      │   Visual    │ REST │   Engine    │      │Processing│
              │   Editor    │ API  │ (This Repo) │      └──────────┘
              └─────────────┘      └─────────────┘
                                          │
                                          ▼
                                    ┌──────────┐
                                    │ Cameras  │
                                    └──────────┘
```

### Backend Layers

```
        ┌─────────────────────┐
        │   Node-RED Nodes    │
        │  (User Interface)   │
        └──────────┬──────────┘
                   │ HTTP
        ┌──────────▼──────────┐
        │   REST API Layer    │
        │  (src/routers/)     │
        │   FastAPI Routes    │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   Managers Layer    │
        │  (src/managers/)    │
        │ • CameraManager     │
        │ • ImageManager      │
        │ • TemplateManager   │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Algorithms Layer   │
        │  (src/algorithms/)  │
        │ • Edge Detection    │
        │ • Color Detection   │
        │ • ArUco • Template  │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   Types & Models    │
        │ • domain_types.py   │
        │ • models.py         │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ Hardware & Storage  │
        │  Cameras • Files    │
        │ Shared Memory       │
        └─────────────────────┘
```

### Request Flow

```
Node-RED                API Router              ImageManager       Algorithm
    │                       │                        │                 │
    │  POST /edge-detect    │                        │                 │
    ├──────────────────────▶│                        │                 │
    │                       │                        │                 │
    │                       │  get(image_id)         │                 │
    │                       │  (via DI)              │                 │
    │                       ├───────────────────────▶│                 │
    │                       │    numpy array         │                 │
    │                       │◀───────────────────────┤                 │
    │                       │                        │                 │
    │                       │  EdgeDetector.detect() │                 │
    │                       ├────────────────────────┼────────────────▶│
    │                       │     VisionObjects      │                 │
    │                       │◀───────────────────────┼─────────────────┤
    │                       │                        │                 │
    │                       │  create_thumbnail()    │                 │
    │                       ├───────────────────────▶│                 │
    │                       │    thumbnail base64    │                 │
    │                       │◀───────────────────────┤                 │
    │                       │                        │                 │
    │    VisionResponse     │                        │                 │
    │◀──────────────────────┤                        │                 │
    │                       │                        │                 │
```

**Note:** Router uses dependency injection (DI) to access ImageManager directly. No intermediate service layer exists.

## API Overview

### Camera Operations (`/api/camera`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/connect` | POST | Connect to camera (USB, IP, or Test mode) |
| `/capture` | POST | Capture single frame |
| `/status` | GET | Check camera connection status |
| `/disconnect` | POST | Disconnect camera |
| `/stream` | GET | Live MJPEG stream |

### Vision Processing (`/api/vision`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/edge-detect` | POST | Edge detection (Canny, Sobel, Laplacian, Prewitt, Scharr) |
| `/color-detect` | POST | Color detection (HSV-based) |
| `/aruco-detect` | POST | ArUco marker detection |
| `/template-match` | POST | Basic template matching (single instance, no rotation) |
| `/advanced-template-match` | POST | Advanced template matching (multi-instance, rotation-invariant) |
| `/rotation-detect` | POST | Rotation analysis (PCA, MinAreaRect, Moments) |

### Template Management (`/api/template`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/learn` | POST | Learn template from image |
| `/upload` | POST | Upload template file |
| `/list` | GET | List available templates |
| `/{template_id}` | GET | Get template info |
| `/{template_id}` | DELETE | Delete template |

### Image Operations (`/api/image`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/extract-roi` | POST | Extract ROI thumbnail |
| `/import` | POST | Import image from filesystem |

### System (`/api/system`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | System information |
| `/metrics` | GET | Performance metrics |

For detailed API documentation with request/response schemas and examples, visit the interactive documentation at `/docs` when the server is running.

## Configuration

### Environment Variables

Configuration can be provided via environment variables with the `MV_` prefix:

```bash
# Server configuration
export MV_SERVER__HOST=0.0.0.0
export MV_SERVER__PORT=8000
export MV_SERVER__DEBUG=false

# Logging
export MV_LOGGING__LEVEL=INFO

# Image processing
export MV_IMAGE__MAX_IMAGES=100
export MV_IMAGE__MEMORY_LIMIT_MB=1024

# Camera
export MV_CAMERA__DEFAULT_WIDTH=1920
export MV_CAMERA__DEFAULT_HEIGHT=1080

# Paths
export MV_PATHS__TEMPLATES=/path/to/templates
```

For nested configuration, use double underscores: `MV_SECTION__SUBSECTION__KEY`

### YAML Configuration

Alternatively, use YAML configuration files:

**Development**: `config.dev.yaml` (automatically loaded in dev mode)
```yaml
server:
  host: 0.0.0.0
  port: 8000
  debug: true

logging:
  level: DEBUG
  verbose: true

camera:
  default_type: test
```

**Production**: `/etc/machinevision/config.yaml` (DEB package installation)

Configuration precedence: **Environment variables > YAML file > Code defaults**

See [config/config.yaml](config/config.yaml) for all available options.

## Related Projects

### Required Frontend

- **Node-RED Custom Nodes**: [node-red-contrib-machine-vision](https://github.com/mat100/node-red-contrib-machine-vision)
  - **Primary user interface** for this backend
  - Visual flow-based programming
  - Pre-built nodes for all vision operations
  - Integration with industrial automation systems (PLCs, SCADA, MQTT, etc.)
  - **This is the recommended way to use the backend**

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Comprehensive development guide
  - Design patterns and principles
  - Code structure and organization
  - Development workflow and tools
  - Testing and debugging
  - Code quality tools (black, flake8, isort)
  - Implementation details and best practices

- **[packaging/INSTALL.md](packaging/INSTALL.md)** - Production deployment guide
  - DEB package installation
  - Systemd service management
  - Configuration for production
  - Backup and rollback procedures

- **[packaging/UPDATE.md](packaging/UPDATE.md)** - Update and maintenance guide
  - Upgrading to new versions
  - Rollback procedures
  - Backup strategies

- **Interactive API Docs**: Available at `/docs` and `/redoc` when server is running

## Development

### Common Commands

```bash
# Setup
make install          # Install dependencies
make dev              # Start development server with auto-reload
make test             # Run test suite
make test-watch       # Run tests in watch mode (TDD)

# Code Quality
make format           # Format code with black and isort
make lint             # Run flake8 linter
make check            # Run all quality checks

# Cleanup
make clean            # Remove cache files and temporary data
```

For detailed development instructions, see [CLAUDE.md](CLAUDE.md).

## Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/managers/test_camera_manager.py

# Run with coverage
pytest --cov=src --cov-report=html

# Watch mode for TDD
make test-watch
```

## License

[Your License Here - e.g., GPL-3.0]

## Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: See docs linked above
- **Node-RED Integration**: See [node-red-contrib-machine-vision](https://github.com/mat100/node-red-contrib-machine-vision) documentation

## Contributing

Contributions are welcome! Please ensure:
- Code follows black formatting (line length 100)
- Tests pass and maintain coverage
- Pre-commit hooks pass (installed automatically with `make install`)
- Documentation is updated

For major changes, please open an issue first to discuss the proposed changes.

---

**Built with**: FastAPI, OpenCV, Python 3.9+
**Part of**: Machine Vision System
**Related**: [node-red-contrib-machine-vision](https://github.com/mat100/node-red-contrib-machine-vision)
