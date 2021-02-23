<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% end comment %}
-->

# LLVM Code Generator - Design Document 

## Introduction 
This document describes the initial design of the LLVM-based code generator backend.

The LLVM generator reuses the existing operator fusion optimizer. It has to compile LLVM IR and execute from C++ based operator templates. I will add the support for cellwise operation for dense matrices.   

## General Design

### C++ design 
I will add a folder to put the LLVM header files and the jni_bridge files (header and cpp) to interact through JNI in src/main/llvm, also I will use the already written helper functions GET/RELEASE_ARRAY to handle input arrays from java code. 
However, eventually only a native proxy shared library will be added to the repository to avoid unnecessary dependencies, while LLVM libraries will be loaded from the native library path similar to native BLAS libraries.

The following method will be exposed: 
- initialize_llvm_context(): handle the creation of the LLVMContext and retrieve the hardware specification (LLVM api); 
- compile_llvm(string: spoofLLVM) that take as input the generated spoof code and add to the LLVM runtime;
- execute_ir that pass the matrices and compute the cellwise operation and return the result to continue the computation flow. 

I will add CMakeLists.txt to support the compilation and linking pass as it was done for the CUDA files. Following the [LLVM documentation](https://llvm.org/doxygen/) I've made a simple example (10.0.0 version) of the usage of the LLVM api that can be found [here](https://github.com/FraCorti/llvm10.0.0-example/blob/main/main.cpp). Technical note: I don't know which [LLVM version](https://releases.llvm.org/) will be better to use since it has changed every two months the last year and there aren't any suggestions online about it, so any suggestions regarding this choice will be useful. 

The SpoofLLVMContext class will have the following structure: 

```
class SpoofLLVMContext{
private:
    std::unique_ptr<LLVMContext> context;  
    std::unique_ptr<SMDiagnostic> error;
    std::string targetTriple;  // target hardware specification 
    std::map<std::string, Module> loadedModules;   // store the spoof operator
    std::unique_ptr<ExecutionEngine> executionEngine;  // runtime executor
public:
    bool loadModule(const std::string& modulePath);
    GenericValue executeModuleFunction(std::string& functionName, GenericValue* params); // execute operation 
};
```

I will add the needed LLVM header manually through Maven to handle the build process ,as it was done for the CUDA files.

### Java design    
I will introduce the new GeneratorAPI value "LLVM" inside the SpoofCompiler class.
After that I will add: 
 - in SpoofCompiler loadNativeCodeGenerator() an initilization of the LLVM context through a native call; 
 - in SpoofCompiler optimize() a native call for compile LLVM IR retrieved from the 
 - in CNode getLanguageTemplateClass() the call to the LLVM creation API. 

I will create a folder llvm inside the cplan folder hops/codegen/cplan/llvm and I will create a CellWise class that follows the structure of the java/CellWise but will return LLVM IR code as a template when the getTemplate(SpoofCellwise.CellType ct) method is called.
Then, following the CUDA implemented structure I will create a SpoofLLVM class that store the name of the CNodeTpl generated. This SpoofLLVMs will be stored inside CodeGenUtils new HashMap<String, SpoofLLVM>  data structure. The SpoofLLVM will have a native method for passing the operands and execute the computation. 

## Steps 
I will first implement the syntactic part and then the runtime part. I will follow the following steps: 
1. Add LLVM to GeneratorAPI in SpoofCompiler and manage it in the SystemDS flow,  then create the llvm/Cellwise class;  
2. Integrate LLVM header through Maven (but for tests only), and create the JNI interface to interact with it; 
3. Creation of the SpoofLLVMContext C++ class and SpoofLLVM java class; 
4. Port SpoofCellWise.java to C++ and call it inside the generated LLVM IR spoof template.
 