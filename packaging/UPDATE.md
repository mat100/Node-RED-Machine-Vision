# Machine Vision Backend - Update Guide

## Overview

This document describes the process of updating Machine Vision Backend to a new version.

## Before Updating

### 1. Check Current Version

```bash
machinevision version
```

### 2. Create Configuration Backup (Optional)

The system automatically creates backups during upgrade, but you can create a manual backup:

```bash
sudo cp -r /etc/machinevision /tmp/machinevision-config-backup
sudo cp -r /var/lib/machinevision /tmp/machinevision-data-backup
```

### 3. Check Service Status

```bash
machinevision status
```

Note whether the service is running so you can restore it to the same state after upgrade.

## Update Process

### Automatic Upgrade Using APT

If the package is available in a repository:

```bash
# Update package list
sudo apt update

# Upgrade to new version
sudo apt upgrade machinevision
```

### Manual Upgrade from DEB File

1. Download the new DEB package:
```bash
wget https://example.com/machinevision_X.Y.Z_all.deb
```

2. Install the new version (apt will automatically perform upgrade):
```bash
sudo apt install --reinstall ./machinevision_X.Y.Z_all.deb
```

## What Happens During Upgrade

1. **Pre-install scripts** create a backup:
   - Application files → `/var/backups/machinevision/backup-TIMESTAMP-app.tar.gz`
   - Configuration → `/var/backups/machinevision/backup-TIMESTAMP-config.tar.gz`
   - Version information → `/var/backups/machinevision/backup-TIMESTAMP.version`

2. **Service is stopped** before updating files

3. **New files are installed**

4. **Post-install scripts**:
   - Check Python dependencies (fail if missing)
   - Restore file permissions
   - Reload systemd daemon
   - Restart service

5. **Service starts** with the new version

## Verification After Upgrade

### 1. Check New Version

```bash
machinevision version
```

You should see the new version.

### 2. Verify Service Status

```bash
machinevision status
```

The service should be active (running).

### 3. Check Logs

```bash
machinevision logs
```

Check for any errors during startup.

### 4. Test API

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/system/info
```

## Rollback to Previous Version

If the new version doesn't work correctly, you can revert to the previous version.

### Automatic Rollback

```bash
sudo machinevision rollback
```

This command:
1. Displays the available backup and asks for confirmation
2. Stops the service
3. Restores the previous version of application and configuration
4. Starts the service again

### Manual Rollback

If automatic rollback doesn't work:

```bash
# Stop service
sudo systemctl stop machinevision.service

# Find backup
ls -lht /var/backups/machinevision/

# Restore application (replace TIMESTAMP with correct timestamp)
cd /var/backups/machinevision
sudo rm -rf /usr/lib/machinevision
sudo mkdir -p /usr/lib
sudo tar -xzf backup-TIMESTAMP-app.tar.gz -C /usr/lib

# Restore configuration (optional)
sudo rm -rf /etc/machinevision
sudo mkdir -p /etc
sudo tar -xzf backup-TIMESTAMP-config.tar.gz -C /etc

# Fix permissions
sudo chown -R machinevision:machinevision /usr/lib/machinevision

# Reload systemd and restart service
sudo systemctl daemon-reload
sudo systemctl start machinevision.service
```

## Backup Management

### View Backups

```bash
ls -lh /var/backups/machinevision/
```

### Automatic Cleanup

The system automatically keeps only the **last 3 backups**. Older backups are automatically removed when creating a new backup.

### Manual Backup Cleanup

If you need to free up space:

```bash
# Remove old backups (older than 30 days)
sudo find /var/backups/machinevision -name "backup-*.tar.gz" -mtime +30 -delete

# Or remove all backups (WARNING!)
sudo rm -rf /var/backups/machinevision/backup-*
```

## Version-Specific Upgrade

### Check Changes Between Versions

Before upgrading, read the release notes or changelog for the new version:
- Check if there are breaking changes
- Check if configuration changes are needed
- Learn about new features and improvements

### Testing Before Production

We recommend testing the upgrade on a test system before deploying to production:

1. Clone configuration:
```bash
scp -r root@production:/etc/machinevision /etc/machinevision-test
```

2. Perform upgrade on test system

3. Test all features

4. After successful test, perform upgrade in production

## Configuration Migration

### Configuration Changes Between Versions

If a new version requires configuration changes:

1. Compare old and new configuration:
```bash
diff /etc/machinevision/config.yaml /usr/lib/machinevision/python-backend/config.yaml
```

2. Update configuration as needed:
```bash
sudo nano /etc/machinevision/config.yaml
```

3. Restart service:
```bash
sudo machinevision restart
```

## Downgrade to Older Version

APT doesn't support automatic downgrade. To downgrade, use:

1. Manual rollback (see above)
2. Or install older version DEB file:
```bash
sudo apt remove machinevision
sudo apt install ./machinevision_OLD_VERSION_all.deb
```

## Troubleshooting Upgrade Issues

### Upgrade Failed

1. Check APT logs:
```bash
cat /var/log/apt/term.log
cat /var/log/dpkg.log
```

2. Try reinstalling:
```bash
sudo apt install --reinstall ./machinevision_X.Y.Z_all.deb
```

### Service Won't Start After Upgrade

1. Check systemd logs:
```bash
journalctl -u machinevision.service -n 50
```

2. Try manual start:
```bash
cd /usr/lib/machinevision/python-backend
sudo -u machinevision PYTHONPATH=/usr/lib/machinevision/python-backend /usr/bin/python3 main.py
```

3. If problem persists, perform rollback:
```bash
sudo machinevision rollback
```

### Missing Python Dependencies

Check and install dependencies:
```bash
# Check which dependencies are missing
/usr/lib/machinevision/check-python-deps.sh /usr/lib/machinevision/python-backend/requirements.txt

# Install missing dependencies
sudo pip3 install -r /usr/lib/machinevision/python-backend/requirements.txt

# Restart service
sudo machinevision restart
```

### Configuration Conflict

If dpkg reports a configuration file conflict:

```
Configuration file '/etc/machinevision/config.yaml'
 ==> Modified (by you or by a script) since installation.
 ==> Package distributor has shipped an updated version.
What would you like to do about it?
```

Recommended actions:
- `Y` - Install new version (your changes will be lost, but a backup will be saved)
- `N` - Keep current version (you can manually merge changes later)

### Insufficient Space for Backup

If you don't have enough space for backup:

1. Clean old backups:
```bash
sudo rm /var/backups/machinevision/backup-*
```

2. Or temporarily change backup location by editing `/var/lib/dpkg/info/machinevision.preinst`

## Best Practices

1. **Always test** upgrade on a test system
2. **Make backups** before upgrade (automatic + manual)
3. **Read release notes** before each upgrade
4. **Monitor logs** after upgrade
5. **Have a rollback plan** ready
6. **Upgrade off-peak** - perform upgrade during low traffic periods

## Upgrade Automation

To automate upgrades, you can use:

```bash
#!/bin/bash
# auto-upgrade.sh

# Download new package
wget -q https://example.com/machinevision_latest_all.deb -O /tmp/machinevision.deb

# Upgrade
sudo apt install -y /tmp/machinevision.deb

# Verification
if machinevision status | grep -q "active (running)"; then
    echo "Upgrade successful"
    rm /tmp/machinevision.deb
else
    echo "Upgrade failed, rolling back"
    sudo machinevision rollback
    exit 1
fi
```

## Support

If you need help with upgrade:
- GitHub Issues: https://github.com/your-org/machine-vision
- Documentation: INSTALL.md, README.md
