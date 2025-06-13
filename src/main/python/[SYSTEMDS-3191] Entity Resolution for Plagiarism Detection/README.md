# CASS Parsing and AST Representation

This repository implements a CASS (Context Aware Semantics Structure) parser using ANTLR4. It consists of the following key components:

- **`CASS.g4`**: Defines the grammar for parsing C-like syntax using ANTLR4.
- **`MyCASSVisitor.py`**: Implements a visitor pattern to generate and traverse the parse tree.
- **`DriverCASS.py`**: Acts as the main entry point for parsing and processing CASS input.
- **`CASSNode.py`**: Defines the Cass node structure and serialization utilities.

---

##  CASS Grammar (`CASS.g4`)
This file defines the ANTLR4 grammar for parsing a subset of C-like syntax, including:
- Function definitions
- Statements (if, while, for, return, switch, case etc.)
- Expressions (arithmetic, logical, assignment)
- Parenthesized expressions
- Function calls
- Variable declarations
- Arrays, lists and pointers

The grammar ensures a well-structured parse tree that is then visited by `MyCASSVisitor.py`.

---

##  CASS Visitor (`MyCASSVisitor.py`)
This module implements the visitor pattern for processing all parsed components using the grammar file. Key functionalities include:

- Constructing `CASSNode`'s
- **Labeling for nodes**, including variable declarations, expressions, and operators.
- **Child management**, allowing hierarchical tree representation.
- Distinguishing between **local and global variables**.
- Recognizing **parenthesized expressions** and **operator precedence**.
- Properly formatting function calls and argument lists.
  ...

The visitor ensures a structured transformation of the parsed syntax into an intermediate AST representation.

---

##  Driver (`DriverCASS.py`)
The driver script serves as the main entry point for executing the CASS parsing pipeline. It:
- Loads and **compiles the grammar** using ANTLR4.
- Instantiates the `MyCASSVisitor` to process the parse tree.
- Generates and prints the corresponding AST structure.
- Serializes the AST into the expected CASS string format.

This script acts as the core engine for testing and processing input files.

---

##  Node Representation (`CASSNode.py`)
This file defines the `CassNode` class, which represents nodes in our Cass tree. It includes:
- **Serialization to CASS format**, ensuring proper formatting for output.
- **GraphViz DOT export**, enabling visualization of the tree structure.

The `CassNode` class provides the foundational structure for storing and manipulating AST representations.

---

##  How to Run
Ensure you have ANTLR4 installed and available in your environment. To run the parser:
```
java -jar "antlr-4.13.2-complete.jar" -Dlanguage=Python3 -visitor CASS.g4
python DriverCASS.py <input_file.c>   
```

## Our Jupyter Notebook
We also created a Jupyter Notebook Execution.ipynb with integrated vectorization and similarity score calculation using a pretrained graph neural network provided by the authors of the MISIM paper. In order to run it, it might be necessary to install some packages. 

