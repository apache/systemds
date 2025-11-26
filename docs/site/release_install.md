
# Install SystemDS from a Release

This guide explains how to install and set up SystemDS using the pre-built release archives. 

---

- [1. Download a Release](#1-download-a-release)
- [2. Install on Windows](#2-install-on-windows)
- [3. Install on Ubuntu 22.04](#3-install-on-ubuntu-2204)
- [4. Install on macOS](#4-install-pon-macos)
- [5. Verify the Installation](#5-verify-the-installation)

---

# 1. Download a Release

Download the official release archive from the Apache SystemDS website:

https://apache.org/dyn/closer.lua/systemds/

After downloading the file `systemds-<VERSION>.tar.gz`, place the file in any directory you choose for installation.

### Java Requirement ###
For compatability with Spark execution and parser components, **Java 17** is strongly recommended for SystemDS.

Verify Java 17:

```bash
java -version
```

If missing, install a JDK 17 distribution.

---

# 2. Install on Windows

### 2.1 Extract the Release Archive

Use Windows built-in extractor.

### 2.2 Set Evironment Variables

To run SystemDS from the command line, configure:
- `SYSTEMDS_ROOT`-> the extracted folder
- Add `%SYSTEMDS_ROOT%\bin` to your `PATH`

Example (PowerShell):

```bash
setx SYSTEMDS_ROOT "C:\path\to\systemds-<VERSION>"
setx PATH "$env:SYSTEMDS_ROOT\bin;$env:PATH"
```

Restart the terminal afterward.

# 3. Install on Ubuntu 22.04

### 3.1 Extract the Release

```bash
cd /path/to/install
tar -xvf systemds-<VERSION>.tar.gz
cd systemds-<VERSION>
```

### 3.2 Add SystemDS to PATH

```bash
export SYSTEMDS_ROOT=$(pwd)
export PATH="$SYSTEMDS_ROOT/bin:$PATH"
```

# 4. Install on macOS

### 4.1 Extract the Release

```bash
cd /path/to/install
tar -xvf systemds-<VERSION>.tar.gz
cd systemds-<VERSION>
```
### 4.2 Add SystemDS to PATH

```bash
export SYSTEMDS_ROOT=$(pwd)
export PATH="$SYSTEMDS_ROOT/bin:$PATH"
```

# Verify the Installation

### 5.1 Check the CLI

```bash
systemds -help
```

You should see usage information printed to the console.

### 5.2 Create a Simple Script

```bash
echo 'print("Hello World!")' > hello.dml
```

### 5.3 Run the Script

```bash
systemds -f hello.dml
```

Expected output:

```bash
Hello World!
```

# Next Steps

For running scripts in Spark mode or experimenting with federated workers, see the Execution Guide: [Execute SystemDS](run_extended.md)

