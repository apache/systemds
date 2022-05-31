@ECHO OFF
::-------------------------------------------------------------
::
:: Licensed to the Apache Software Foundation (ASF) under one
:: or more contributor license agreements.  See the NOTICE file
:: distributed with this work for additional information
:: regarding copyright ownership.  The ASF licenses this file
:: to you under the Apache License, Version 2.0 (the
:: "License"); you may not use this file except in compliance
:: with the License.  You may obtain a copy of the License at
::
::   http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing,
:: software distributed under the License is distributed on an
:: "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
:: KIND, either express or implied.  See the License for the
:: specific language governing permissions and limitations
:: under the License.
::
::-------------------------------------------------------------

:: This shell script compiles the required shared libraries for Windows on x86-64
:: Make sure to have run "\"%INTEL_ROOT%\"\compilervars.bat intel64 vs2019" and set OpenBLAS_HOME to where
:: libopenblas.lib is located

:: configure and compile INTEL MKL
cmake -S . -B INTEL -DUSE_INTEL_MKL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build INTEL --target install --config Release
rmdir /Q /S INTEL

:: configure and compile OPENBLAS
cmake . -B OPENBLAS -DUSE_OPEN_BLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build OPENBLAS --target install --config Release
rmdir /Q /S OPENBLAS

cmake he\ -B HE -DCMAKE_BUILD_TYPE=Release
cmake --build HE --target install --config Release
rmdir /Q /S HE

echo.
echo "Make sure to re-run mvn package to make use of the newly compiled libraries"