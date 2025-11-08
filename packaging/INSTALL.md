# Machine Vision Backend - Installation Guide

## Overview

Machine Vision Backend is a FastAPI-based application for image processing and computer vision. This documentation describes installation from a DEB package.

## System Requirements

- **Operating System**: Debian 10+, Ubuntu 20.04+, or RISC-V Linux
- **Python**: 3.9 or newer
- **RAM**: Minimum 512 MB, recommended 2 GB+
- **Disk Space**: Minimum 200 MB

## Prerequisites

**IMPORTANT**: This package does NOT bundle Python dependencies. You must install them separately before installing the DEB package.

### Option 1: Install via pip (Recommended)

```bash
sudo pip3 install fastapi==0.104.1 uvicorn==0.24.0 pydantic==1.10.13 \
    numpy>=2.0.0 opencv-python-headless>=4.8.0 python-dotenv==1.0.0 \
    pyyaml>=6.0.0
```

Or install all from requirements.txt:
```bash
# Download requirements.txt first, or use from source
wget https://example.com/requirements.txt
sudo pip3 install -r requirements.txt
```

### Option 2: Install via apt (if available)

```bash
sudo apt install python3-fastapi python3-uvicorn python3-pydantic \
    python3-numpy python3-opencv python3-dotenv python3-yaml
```

**Note**: Not all packages may be available in apt repositories, especially on older Debian/Ubuntu versions. Use pip if packages are missing.

### Verify Installation

Before installing the DEB package, verify dependencies:
```bash
python3 -c "import fastapi, uvicorn, pydantic, numpy, cv2, dotenv, yaml; print('✓ All dependencies available')"
```

## Installation from DEB Package

### 1. Download Package

Download the latest DEB package version:
```bash
# Example - adjust path according to your location
wget https://example.com/machinevision_1.0.0_all.deb
```

Or if you have the package locally (e.g., after build):
```bash
cd /path/to/machine-vision/backend
make deb
# Package will be in: build/machinevision_X.Y.Z_all.deb
```

### 2. Install Package

Install the package using apt:
```bash
sudo apt install ./machinevision_1.0.0_all.deb
```

Apt will automatically install all system dependencies, including:
- Python 3.9+
- python3-pip
- systemd

**Note**: The installation will check for Python dependencies and fail if they are missing. Make sure you completed the Prerequisites step above.

### 3. Verify Installation

After installation, verify that the service is running:
```bash
machinevision status
```

You should see:
```
● machinevision.service - Machine Vision Backend
     Loaded: loaded (/etc/systemd/system/machinevision.service; enabled)
     Active: active (running)
```

Verify version:
```bash
machinevision version
```

### 4. Test API Endpoint

Test that the API is responding:
```bash
curl http://localhost:8000/health
```

You should see:
```json
{"status": "healthy"}
```

## Installation Structure

After installation, files will be located in these directories:

```
/usr/lib/machinevision/          # Application files
├── src/              # Backend application (Python code)
├── check-python-deps.sh         # Dependency checker script
└── VERSION                      # Version file

/etc/machinevision/              # Configuration
└── config.yaml                  # Main configuration file

/var/lib/machinevision/          # Runtime data
├── data/                        # Temporary data
└── templates/                   # Templates for template matching

/var/log/machinevision/          # Logs

/var/backups/machinevision/      # Backups during upgrade

/usr/bin/machinevision           # CLI wrapper
```

**Note**: The application uses system Python (/usr/bin/python3) and system-installed dependencies. No virtual environment is created.

## Service Management

Machine Vision Backend provides a simple CLI tool for service management:

```bash
# Start service
sudo machinevision start

# Stop service
sudo machinevision stop

# Restart service
sudo machinevision restart

# Service status
machinevision status

# Display logs (last 50 lines)
machinevision logs

# Follow logs in real-time
machinevision logs -f

# Display version
machinevision version

# Rollback to previous version
sudo machinevision rollback
```

## Configuration

The main configuration file is located at `/etc/machinevision/config.yaml`.

### Basic Configuration

```yaml
api:
  host: "0.0.0.0"
  port: 8000

image:
  max_images: 100
  max_memory_mb: 1000

camera:
  default_width: 1920
  default_height: 1080
```

After changing configuration, restart the service:
```bash
sudo machinevision restart
```

### Configuration via Environment Variables

You can override settings using environment variables with the `MV_` prefix:

```bash
# Example: change port
export MV_API__PORT=8080

# Restart service
sudo machinevision restart
```

## Testing Installation

### Test Basic Functionality

1. Check health endpoint:
```bash
curl http://localhost:8000/health
```

2. Get system information:
```bash
curl http://localhost:8000/api/system/info
```

3. View API documentation in browser:
```
http://localhost:8000/docs
```

## Access API Documentation

Machine Vision Backend provides automatic API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Permissions and Security

### System User

The service runs under the `machinevision` system user, which is automatically created during installation.

### Port 8000

By default, the service listens on port 8000. If you're using a firewall, allow this port:

```bash
sudo ufw allow 8000/tcp
```

### SSL/TLS

For production deployment, we recommend using a reverse proxy (nginx, Apache) with SSL certificate.

## Uninstallation

### Remove Package (Keep Data)

```bash
sudo apt remove machinevision
```

This removes the application but preserves data in `/var/lib/machinevision` and backups in `/var/backups/machinevision`.

### Complete Removal (Including Data)

```bash
sudo apt purge machinevision
```

This removes the application and all data (except backups).

## Troubleshooting

### Service Won't Start

1. Check logs:
```bash
machinevision logs
# or
journalctl -u machinevision.service -n 100
```

2. Verify configuration:
```bash
cat /etc/machinevision/config.yaml
```

3. Check permissions:
```bash
ls -la /var/lib/machinevision
ls -la /var/log/machinevision
```

### Port 8000 Already in Use

Change the port in configuration:
```bash
sudo nano /etc/machinevision/config.yaml
# Change api.port to another port (e.g., 8080)
sudo machinevision restart
```

### Out of Memory

Reduce the image cache limit in configuration:
```yaml
image:
  max_images: 50
  max_memory_mb: 500
```

### Missing Python Dependencies

If the service won't start due to missing dependencies:
```bash
# Check which dependencies are missing
/usr/lib/machinevision/check-python-deps.sh /usr/lib/machinevision/src/requirements.txt

# Install missing dependencies
sudo pip3 install -r /usr/lib/machinevision/src/requirements.txt

# Restart service
sudo machinevision restart
```

## Support

For more information and bug reports:
- GitHub Issues: https://github.com/your-org/machine-vision
- Documentation: README.md in the project
