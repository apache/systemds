
# Install SystemDS from a Release

This guide explains how to install and set up SystemDS using the pre-built release archives. 

---

- [1. Download a Release](#1-download-a-release)
- [2. Install on Windows](#2-install-on-windows)
- [3. Install on Ubuntu 22.04](#3-install-on-ubuntu-2204)
- [4. Install on macOS](#4-install-on-macos)

---

# 1. Download a Release

Download the official release archive from the Apache SystemDS website:

https://systemds.apache.org/download

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

### 2.3 Verify the Installation by Checking the CLI

On Windows, the `systemds`CLI wrapper may not be executable. This is expected because the `bin/systemds`launcher is implemented as a shell script, which Windows cannot execute natively. To verify the installation on Windows, navigate to the bin directory and run the JAR directly. Note that running `systemds -help` without JAR may result in a CommandNotFoundExeption:

```bash
java -jar systemds-3.3.0.jar -help
```

You should see usage information as an output printed to the console.

### 2.4 Create a Simple Script

On Windows, especially when using PowerShell, creating text files via shell redirection (e.g., echo...) may result in unexpected encoding or invisible characters. This can lead to parsing errors when executing the script, even though the file appears correct in an editor. Therefore, you may try creating the file explicitly using PowerShell:
```bash
Set-Content -Path .\hello.dml -Value 'print("Hello World!")' -Encoding Ascii
```

This ensures the script is stored as plain text without additional encoding metadata.
Note: This behavior depends on the shell and environment configuration and may not affect all Windows setups.

Verify the file contents:
```bash
Get-Content .\hello.dml
```

Expected output:
```bash
print("Hello World!")
```

### 2.5 Run the Script

Now run the script:
```bash
java -jar systemds-3.3.0.jar -f .\hello.dml
```

Expected output:
```bash
Hello World!
SystemDS Statistics:
Total execution time: 0.012 sec.
```

# 3. Install on Ubuntu 22.04

### 3.1 Extract the Release

```bash
cd /path/to/install
tar -xvf systemds-<VERSION>-bin.tgz
cd systemds-<VERSION>-bin
```

### 3.2 Add SystemDS to PATH

```bash
export SYSTEMDS_ROOT=$(pwd)
export PATH="$SYSTEMDS_ROOT/bin:$PATH"
```

(Optional but recommended) To make SystemDS available in new terminals, add the following lines to your shell configuration (e.g., ~/.bashrc or ~/.profile):
```bash
export SYSTEMDS_ROOT=/absolute/path/to/systemds-<VERSION>
export PATH=$SYSTEMDS_ROOT/bin:$PATH
```

### 3.3 Verify the Installation by Checking the CLI

```bash
systemds -help
```

You should see usage information printed to the console.

### 3.4 Create a Simple Script

```bash
echo 'print("Hello World!")' > hello.dml
```

### 3.5 Run the Script

On some Ubuntu setups (including clean Docker images), running SystemDS directly may fail with `Invalid or corrupt jarfile hello.dml` Error. In this case, explicitly pass the SystemDS JAR shipped with the release.

Locate the JAR in the release root:
```bash
SYSTEMDS_JAR=$(find "$SYSTEMDS_ROOT" -maxdepth 1 -type f -name "systemds-*.jar" | head -n 1)
echo "Using SystemDS JAR: $SYSTEMDS_JAR"
```

Then run:
```bash
systemds "$SYSTEMDS_JAR" -f hello.dml
```

Expected output:
```bash
Hello World!
```

# 4. Install on macOS

### 4.1 Extract the Release

```bash
cd /path/to/install
tar -xvf systemds-<VERSION>-bin.tgz
cd systemds-<VERSION>-bin
```
### 4.2 Add SystemDS to PATH

```bash
export SYSTEMDS_ROOT=$(pwd)
export PATH="$SYSTEMDS_ROOT/bin:$PATH"
```

(Optional but recommended)
To make SystemDS available in new terminals, add the following lines
to your shell configuration (e.g., ~/.bashrc or ~/.profile):
```bash
export SYSTEMDS_ROOT=/absolute/path/to/systemds-<VERSION>
export PATH=$SYSTEMDS_ROOT/bin:$PATH
```

### 4.3 Verify the Installation by Checking the CLI

```bash
systemds -help
```

You should see usage information printed to the console.

### 4.4 Create a Simple Script

```bash
echo 'print("Hello World!")' > hello.dml
```

### 4.5 Run the Script

```bash
systemds -f hello.dml
```

Expected output:

```bash
Hello World!
```

# Next Steps

For running scripts in Spark mode or experimenting with federated workers, see the Execution Guide: [Execute SystemDS](run_extended.md)

