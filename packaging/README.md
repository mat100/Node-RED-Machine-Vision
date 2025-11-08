# Machine Vision Backend - DEB Packaging

This directory contains files for creating a DEB package of Machine Vision Backend.

## Quick Start

### Build DEB Package

```bash
# From backend/ directory
make deb
```

Package will be created at `backend/build/machinevision_X.Y.Z_all.deb`

### Install Build Dependencies

If the build fails due to missing dependencies, install them:

```bash
sudo apt-get install debhelper devscripts dpkg-dev python3 python3-venv python3-pip
```

## File Structure

```
packaging/
├── README.md              # This file
├── VERSION                # Package version (1.0.0)
├── build-deb.sh           # Build script
├── machinevision-cli      # CLI wrapper for service management
├── INSTALL.md             # Installation guide for end users
└── UPDATE.md              # Update and rollback guide

../debian/
├── control                # Package metadata, dependencies
├── install                # What gets installed where
├── rules                  # Build rules
├── preinst                # Pre-install script (backups)
├── postinst               # Post-install script (service setup)
├── prerm                  # Pre-removal script (stop service)
├── postrm                 # Post-removal script (cleanup)
└── machinevision.service  # Systemd service file
```

## Change Version

Edit the `packaging/VERSION` file:
```bash
echo "1.0.1" > packaging/VERSION
```

The build script will automatically use this version.

## Manual Build

If you want to perform a manual build:

```bash
cd backend/
dpkg-buildpackage -us -uc -b
```

## Install Package

After creating the package:

```bash
sudo apt install ./build/machinevision_1.0.0_all.deb
```

## Service Management

After installation, you can manage the service using:

```bash
machinevision start|stop|restart|status|logs|version|rollback
```

## Documentation

- **INSTALL.md** - Complete installation guide
- **UPDATE.md** - Update and rollback guide

## Prerequisites

**IMPORTANT**: Python dependencies must be installed separately before installing this package.

```bash
# Install via pip (recommended)
sudo pip3 install -r python-backend/requirements.txt

# Or verify dependencies are available
python3 -c "import fastapi, uvicorn, pydantic, numpy, cv2, dotenv, yaml"
```

See `INSTALL.md` for detailed prerequisites instructions.

## What Installation Does

1. **Checks Python dependencies** (fails if any are missing)
2. **Creates system user** `machinevision`
3. **Installs files**:
   - Application → `/usr/lib/machinevision/`
   - Configuration → `/etc/machinevision/`
   - CLI wrapper → `/usr/bin/machinevision`
   - Dependency checker → `/usr/lib/machinevision/check-python-deps.sh`
4. **Sets file and directory permissions**
5. **Installs and starts systemd service** (uses system Python)
6. **Creates directories**:
   - Data → `/var/lib/machinevision/`
   - Logs → `/var/log/machinevision/`
   - Backups → `/var/backups/machinevision/`

## Upgrade and Rollback

During upgrade:
- Automatically creates backup of previous version
- Keeps last 3 backups
- Allows rollback: `sudo machinevision rollback`

## Testing Before Installation

Inspect package contents:
```bash
dpkg-deb --info build/machinevision_1.0.0_all.deb
dpkg-deb --contents build/machinevision_1.0.0_all.deb
```

## Uninstallation

```bash
# Remove package (keep data)
sudo apt remove machinevision

# Complete removal (including data)
sudo apt purge machinevision
```

## Notes

- Package is **architecture: all** (architecture independent)
- Package size: **~1 MB** (dependencies not bundled)
- Requires **Python 3.9+** with dependencies pre-installed
- Uses **system Python** (no virtualenv)
- Tested on **Debian 10+**, **Ubuntu 20.04+**, **RISC-V Linux**
- Service runs as `machinevision` user (security isolation)
