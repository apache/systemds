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

setLocal EnableDelayedExpansion

:: directory to write test case specific data
SET TEMP=\temp\systemml_test

:: should this test suite continue after encountering an error? true false
SET CONTINUE_ON_ERROR=true

:: expecting this test script to be in directory %PROJECT_ROOT_DIR%\src\test\scripts
SET TEST_SCRIPT_REL_DIR=src\test\scripts

:: expecting the run script to be in directory %PROJECT_ROOT_DIR%\bin
SET RUN_SCRIPT=systemml.bat

:: the DML script with arguments we use to test the run script
SET DML_SCRIPT=genLinearRegressionData.dml
SET DML_SCRIPT_PATH=scripts\datagen
SET DML_OUTPUT=linRegData.csv
SET DML_ARGS=-nvargs numSamples=1000 numFeatures=50 maxFeatureValue=5 maxWeight=5 addNoise=FALSE b=0 sparsity=0.7 output=%DML_OUTPUT% format=csv perc=0.5
SET DML_SCRIPT_WITH_ARGS=%DML_SCRIPT% %DML_ARGS%

SET USER_DIR=%CD%

SET TEST_SCRIPT_PATH=%~dp0
SET TEST_SCRIPT_PATH=%TEST_SCRIPT_PATH:~0,-1%

SET PROJECT_ROOT_DIR=!TEST_SCRIPT_PATH:%TEST_SCRIPT_REL_DIR%=!
SET PROJECT_ROOT_DIR=%PROJECT_ROOT_DIR:~0,-1%

IF "%TEST_SCRIPT_PATH%"=="%PROJECT_ROOT_DIR%" (
    echo This test script "%~nx0" is expected to be located in folder "%TEST_SCRIPT_REL_DIR%" under the project root.
    echo Please update "%0" and correctly set the variable "TEST_SCRIPT_REL_DIR".
)
IF NOT EXIST "%PROJECT_ROOT_DIR%\bin\%RUN_SCRIPT%" (
  echo Could not find "bin\%RUN_SCRIPT%" in the project root directory. If the actual path of the run script is not "bin\%RUN_SCRIPT%", or if the actual project root directory is not "%PROJECT_ROOT_DIR%", please update this test script "%0".
  GOTO End
)

SET DATE_TIME=%date:/=-%-%time::=-%
SET DATE_TIME=%DATE_TIME: =-%
SET TEST_LOG="%PROJECT_ROOT_DIR%\Temp\test_runScript_%DATE_TIME%.log"

:: test setup

SET FAILED_TESTS=

IF NOT EXIST "%TEMP%" mkdir "%TEMP%"
IF NOT EXIST "%PROJECT_ROOT_DIR%\temp" mkdir "%PROJECT_ROOT_DIR%\temp"


:: start the test cases

echo Writing test log to file %TEST_LOG%.

:: invoke the run script from the project root directory

SET CURRENT_TEST=Test_root__DML_script_with_path
(
    echo Running test "%CURRENT_TEST%"
    echo Running test "%CURRENT_TEST%"                             >> %TEST_LOG%
    cd "%PROJECT_ROOT_DIR%"
    del /F /Q "%DML_OUTPUT%"                                       >  nul        2>&1
    del /F /Q "temp\%DML_OUTPUT%"                                  >  nul        2>&1
    call bin\%RUN_SCRIPT% %DML_SCRIPT_PATH%\%DML_SCRIPT_WITH_ARGS% >> %TEST_LOG% 2>&1
    IF ERRORLEVEL 1                  SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST% &                            IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF EXIST "%DML_OUTPUT%"          SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-outputdata-in-project-root & IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF NOT EXIST "temp\%DML_OUTPUT%" SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-missing-outputdata &         IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
)

SET CURRENT_TEST=Test_root__DML_script_file_name
(
    echo Running test "%CURRENT_TEST%"
    echo Running test "%CURRENT_TEST%"            >> %TEST_LOG%
    cd "%PROJECT_ROOT_DIR%"
    del /F /Q "%DML_OUTPUT%"                      >  nul        2>&1
    del /F /Q "temp\%DML_OUTPUT%"                 >  nul        2>&1
    call bin\%RUN_SCRIPT% %DML_SCRIPT_WITH_ARGS%  >> %TEST_LOG% 2>&1
    IF ERRORLEVEL 1                  SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST% &                            IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF EXIST "%DML_OUTPUT%"          SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-outputdata-in-project-root & IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF NOT EXIST "temp\%DML_OUTPUT%" SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-missing-outputdata &         IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
)


:: invoke the run script from the bin directory

SET CURRENT_TEST=Test_bin__DML_script_with_path
(
    echo Running test "%CURRENT_TEST%"
    echo Running test "%CURRENT_TEST%"                             >> %TEST_LOG%
    cd "%PROJECT_ROOT_DIR%\bin"
    del /F /Q "%DML_OUTPUT%"                                       >  nul         2>&1
    del /F /Q "..\temp\%DML_OUTPUT%"                               >  nul         2>&1
    call %RUN_SCRIPT% ..\%DML_SCRIPT_PATH%\%DML_SCRIPT_WITH_ARGS%  >> %TEST_LOG%  2>&1
    IF ERRORLEVEL 1                     SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST% &                          IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF EXIST "%DML_OUTPUT%"             SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-outputdata-in-bin-folder & IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF NOT EXIST "..\temp\%DML_OUTPUT%" SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-missing-outputdata &       IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
)


SET CURRENT_TEST=Test_bin__DML_script_file_name
(
    echo Running test "%CURRENT_TEST%"
    echo Running test "%CURRENT_TEST%"        >> %TEST_LOG%
    cd "%PROJECT_ROOT_DIR%\bin"
    echo Working directory: %CD%              >> %TEST_LOG%
    del /F /Q "%DML_OUTPUT%"                  >  nul         2>&1
    del /F /Q "..\temp\%DML_OUTPUT%"          >  nul         2>&1
    call %RUN_SCRIPT% %DML_SCRIPT_WITH_ARGS%  >> %TEST_LOG%  2>&1
    IF ERRORLEVEL 1                     SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST% &                          IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF EXIST "%DML_OUTPUT%"             SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-outputdata-in-bin-folder & IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF NOT EXIST "..\temp\%DML_OUTPUT%" SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-missing-outputdata &       IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
)



:: invoke the run script from a working directory outside of the project root

SET CURRENT_TEST=Test_out__DML_script_with_path
(
    echo Running test "%CURRENT_TEST%"
    echo Running test "%CURRENT_TEST%" >> %TEST_LOG%
    cd "%TEMP%"
    echo Working directory: %CD%       >> %TEST_LOG%
    del /F /Q "%DML_OUTPUT%"           >  nul 2>&1
    call "%PROJECT_ROOT_DIR%\bin\%RUN_SCRIPT%" "%PROJECT_ROOT_DIR%\%DML_SCRIPT_PATH%\%DML_SCRIPT%" %DML_ARGS%  >> %TEST_LOG%  2>&1
    IF ERRORLEVEL 1             SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST% &                    IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF NOT EXIST "%DML_OUTPUT%" SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-missing-outputdata & IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
)

SET CURRENT_TEST=Test_out__DML_script_file_name
(
    echo Running test "%CURRENT_TEST%"
    echo Running test "%CURRENT_TEST%"                                 >> %TEST_LOG%
    cd "%TEMP%"
    echo Working directory: %CD%                                       >> %TEST_LOG%
    del /F /Q "%DML_OUTPUT%"                                           >  nul         2>&1
    call "%PROJECT_ROOT_DIR%\bin\%RUN_SCRIPT%" %DML_SCRIPT_WITH_ARGS%  >> %TEST_LOG%  2>&1
    IF ERRORLEVEL 1             SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST% &                    IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF NOT EXIST "%DML_OUTPUT%" SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-missing-outputdata & IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
)



:: test again from a directory with spaces in path name

SET "SPACE_DIR=%TEMP%\Space Folder\SystemML"

IF NOT EXIST "%SPACE_DIR%" (
    echo mkdir "%SPACE_DIR%" >> %TEST_LOG%
    mkdir "%SPACE_DIR%"
)

echo Copying project contents from "%PROJECT_ROOT_DIR%" to "%SPACE_DIR%"  >> %TEST_LOG%
robocopy "%PROJECT_ROOT_DIR%" "%SPACE_DIR%" /S /nfl /ndl /xf *.java Test*.* /xd .git .idea .settings  >> %TEST_LOG%


:: invoke the run script from the project root directory

SET CURRENT_TEST=Test_space_root__DML_script_with_path
(
    echo Running test "%CURRENT_TEST%"
    echo Running test "%CURRENT_TEST%"                              >> %TEST_LOG%
    cd "%SPACE_DIR%"
    echo Working directory: %CD%                                    >> %TEST_LOG%
    del /F /Q "%DML_OUTPUT%"                                        >  nul         2>&1
    del /F /Q "temp\%DML_OUTPUT%"                                   >  nul         2>&1
    call bin\%RUN_SCRIPT% %DML_SCRIPT_PATH%\%DML_SCRIPT_WITH_ARGS%  >> %TEST_LOG%  2>&1
    IF ERRORLEVEL 1                  SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST% &                            IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF EXIST "%DML_OUTPUT%"          SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-outputdata-in-project-root & IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF NOT EXIST "temp\%DML_OUTPUT%" SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-missing-outputdata &         IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
)

SET CURRENT_TEST=Test_space_root__DML_script_file_name
(
    echo Running test "%CURRENT_TEST%"
    echo Running test "%CURRENT_TEST%"            >> %TEST_LOG%
    cd "%SPACE_DIR%"
    echo Working directory: %CD%                  >> %TEST_LOG%
    del /F /Q "%DML_OUTPUT%"                      >  nul         2>&1
    del /F /Q "temp\%DML_OUTPUT%"                 >  nul         2>&1
    call bin\%RUN_SCRIPT% %DML_SCRIPT_WITH_ARGS%  >> %TEST_LOG%  2>&1
    IF ERRORLEVEL 1                  SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST% &                            IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF EXIST "%DML_OUTPUT%"          SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-outputdata-in-project-root & IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF NOT EXIST "temp\%DML_OUTPUT%" SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-missing-outputdata &         IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
)


:: invoke the run script from the bin directory

SET CURRENT_TEST=Test_space_bin__DML_script_with_path
(
    echo Running test "%CURRENT_TEST%"
    echo Running test "%CURRENT_TEST%"                             >> %TEST_LOG%
    cd "%SPACE_DIR%\bin"
    echo Working directory: %CD%                                   >> %TEST_LOG%
    del /F /Q "%DML_OUTPUT%"                                       >  nul         2>&1
    del /F /Q "..\temp\%DML_OUTPUT%"                               >  nul         2>&1
    call %RUN_SCRIPT% ..\%DML_SCRIPT_PATH%\%DML_SCRIPT_WITH_ARGS%  >> %TEST_LOG%  2>&1
    IF ERRORLEVEL 1                     SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST% &                          IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF EXIST "%DML_OUTPUT%"             SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-outputdata-in-bin-folder & IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF NOT EXIST "..\temp\%DML_OUTPUT%" SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-missing-outputdata &       IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
)

SET CURRENT_TEST=Test_space_bin__DML_script_file_name
(
    echo Running test "%CURRENT_TEST%"
    echo Running test "%CURRENT_TEST%"        >> %TEST_LOG%
    cd "%SPACE_DIR%\bin"
    echo Working directory: %CD%              >> %TEST_LOG%
    del /F /Q "%DML_OUTPUT%"                  >  nul         2>&1
    del /F /Q "..\temp\%DML_OUTPUT%"          >  nul         2>&1
    call %RUN_SCRIPT% %DML_SCRIPT_WITH_ARGS%  >> %TEST_LOG%  2>&1
    IF ERRORLEVEL 1                     SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST% &                          IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF EXIST "%DML_OUTPUT%"             SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-outputdata-in-bin-folder & IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF NOT EXIST "..\temp\%DML_OUTPUT%" SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-missing-outputdata &       IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
)


:: invoke the run script from a working directory outside of the project root

SET CURRENT_TEST=Test_space_out__DML_script_with_path
(
    echo Running test "%CURRENT_TEST%":
    echo Running test "%CURRENT_TEST%" >> %TEST_LOG%
    cd "%TEMP%"
    echo Working directory: %CD%       >> %TEST_LOG%
    del /F /Q "%DML_OUTPUT%"           >  nul 2>&1
    call "%SPACE_DIR%\bin\%RUN_SCRIPT%" "%SPACE_DIR%\%DML_SCRIPT_PATH%\%DML_SCRIPT%" %DML_ARGS%  >> %TEST_LOG%  2>&1
    IF ERRORLEVEL 1             SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST% &                    IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF NOT EXIST "%DML_OUTPUT%" SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-missing-outputdata & IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
)

SET CURRENT_TEST=Test_space_out__DML_script_file_name
(
    echo Running test "%CURRENT_TEST%":
    echo Running test "%CURRENT_TEST%"                          >> %TEST_LOG%
    cd "%TEMP%"
    echo Working directory: %CD%                                >> %TEST_LOG%
    del /F /Q "%DML_OUTPUT%"                                    >  nul         2>&1
    call "%SPACE_DIR%\bin\%RUN_SCRIPT%" %DML_SCRIPT_WITH_ARGS%  >> %TEST_LOG%  2>&1
    IF ERRORLEVEL 1             SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST% &                    IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
    IF NOT EXIST "%DML_OUTPUT%" SET FAILED_TESTS=%FAILED_TESTS% %CURRENT_TEST%-missing-outputdata & IF "%CONTINUE_ON_ERROR%"=="false" GOTO Failure
)






:Check_results


IF "%FAILED_TESTS%"=="" GOTO Success


:Failure
ECHO ================================================================================
ECHO Failed test cases: [%FAILED_TESTS%]
GOTO Cleanup


:Success
ECHO ================================================================================
ECHO Tests succeeded.
GOTO Cleanup


:Cleanup
cd %USER_DIR%
IF EXIST "%TEMP%" ( rmdir /Q /S "%TEMP%" )
GOTO End


:End
echo Test log was written to file %TEST_LOG%.
