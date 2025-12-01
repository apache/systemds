
# SystemDS Kernel Setup Guide

This README outlines the steps to set up a Jupyter kernel for SystemDS, focusing on building SystemDS from source, preparing a custom repository with specific dependencies, and setting up a Jupyter kernel.

## Prerequisites

- Java JDK 11 or later
- Maven
- Git
- Jupyter Notebook or JupyterLab

## Step 1: Clone and Build SystemDS

Clone the Apache SystemDS repository and build it. This step ensures that the SystemDS JAR is locally available.

[How to install SystemDS?](https://apache.github.io/systemds/site/install.html)

After the build, save the JAR to your local Maven Repo.

```bash
mvn install:install-file -Dfile=path/to/your/jar-file.jar -DgroupId=org.apache.sds -DartifactId=sds -Dversion=3.2.0 -Dpackaging=jar
```
Note: Ensure that any modifications to groupId, artifactId, and version are carefully mirrored in the pom.xml dependencies section of the Kernel. Inconsistencies between these identifiers in your project setup and the pom.xml file can lead to build failures or dependency resolution issues.

## Step 2: Set Up Your Repository

Clone SystemDS kernel.

```bash
git clone https://github.com/kubieren/SystemDSKernel.git
cd SystemDSKernel/kernelsds
```

Build your project, which is configured with Maven. This step compiles your code and packages it, taking into account the dependencies specified in your `pom.xml`.

```bash
mvn clean package
```

## Step 3: Create the Kernel Specification

Navigate to or create a directory where you wish to store your kernel's configuration. For example, create `my_systemds_kernel` in your home directory:

```bash
mkdir -p ~/my_systemds_kernel
```

Within this directory, create a `kernel.json` file with the following content, adjusting the path to your JAR file as necessary:

```json
{
    "argv": ["java", "-jar", "/path/to/your/kernelJarFile/kernelsds-1.0-SNAPSHOT.jar", "{connection_file}"],
    "display_name": "SystemDS Kernel",
    "language": "systemds",
    "interrupt_mode": "message"
}
```

## Step 4: Install the Kernel Specification

Install your kernel specification with Jupyter by running:

```bash
jupyter kernelspec install ~/path_to_my_systemds_kernel --user
```

This command makes the SystemDS kernel available to Jupyter.

## Step 5: Launch Jupyter Notebook

Start Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

You should now be able to create new notebooks with the "SystemDS Kernel" option.

## Conclusion

Follow these steps to integrate SystemDS with Jupyter Notebook, allowing you to execute SystemDS operations directly from Jupyter notebooks. Ensure all paths and URLs are correct based on your environment and where you've placed the SystemDS JAR file.
