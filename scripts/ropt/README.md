# Implementation notes for Resource Optimization in Cloud Environments

## Applied changes to the existing code
Here is a list all changes done to existing classes up to the current point of implementation

1. Adding a new _CompilerConfig_ flag to mark if the compilation is done indeed for resource optimization or for general script execution.
The purpose is to distinguish in particular places between the different compilation targets so existing 
optimization are still inplace for the actual script execution but these optimizations hinder the resource optimization.
2. 

## Implemented classes


## Tests


## Implemented scripts


