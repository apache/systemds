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

echo ================================================================================

IF "%~1" == ""      GOTO Err
IF "%~1" == "-help" GOTO Msg
IF "%~1" == "-h"    GOTO Msg

setLocal EnableDelayedExpansion

SET USER_DIR=%CD%

pushd %~dp0..
SET PROJECT_ROOT_DIR=%CD%
popd

SET BUILD_DIR=%PROJECT_ROOT_DIR%\target
SET HADOOP_LIB_DIR=%BUILD_DIR%\lib
SET DML_SCRIPT_CLASS=%BUILD_DIR%\classes\org\apache\sysml\api\DMLScript.class

SET BUILD_ERR_MSG=You must build the project before running this script.
SET BUILD_DIR_ERR_MSG=Could not find target directory "%BUILD_DIR%". %BUILD_ERR_MSG%
SET HADOOP_LIB_ERR_MSG=Could not find required libraries "%HADOOP_LIB_DIR%\*". %BUILD_ERR_MSG%
SET DML_SCRIPT_ERR_MSG=Could not find "%DML_SCRIPT_CLASS%". %BUILD_ERR_MSG%

:: check if the project had been built and the jar files exist
IF NOT EXIST "%BUILD_DIR%"        ( echo %BUILD_DIR_ERR_MSG%  & GOTO ExitErr )
IF NOT EXIST "%HADOOP_LIB_DIR%"   ( echo %HADOOP_LIB_ERR_MSG% & GOTO ExitErr )
IF NOT EXIST "%DML_SCRIPT_CLASS%" ( echo %DML_SCRIPT_ERR_MSG% & GOTO ExitErr )


:: if the present working directory is the project root or the bin folder, then use the temp folder as user.dir
IF "%USER_DIR%" == "%PROJECT_ROOT_DIR%" (
  SET USER_DIR=%PROJECT_ROOT_DIR%\temp
  ECHO Output dir: "!USER_DIR!"
)
IF "%USER_DIR%" == "%PROJECT_ROOT_DIR%\bin" (
  SET USER_DIR=%PROJECT_ROOT_DIR%\temp
  ECHO Output dir: "!USER_DIR!"
)


:: if the SystemML-config.xml does not exist, create it from the template
IF NOT EXIST "%PROJECT_ROOT_DIR%\conf\SystemML-config.xml" (
  copy "%PROJECT_ROOT_DIR%\conf\SystemML-config.xml.template" ^
       "%PROJECT_ROOT_DIR%\conf\SystemML-config.xml" > nul
  echo ...created "%PROJECT_ROOT_DIR%\conf\SystemML-config.xml"
)

:: if the log4j.properties do not exis, create them from the template
IF NOT EXIST "%PROJECT_ROOT_DIR%\conf\log4j.properties" (
  copy "%PROJECT_ROOT_DIR%\conf\log4j.properties.template" ^
       "%PROJECT_ROOT_DIR%\conf\log4j.properties" > nul
  echo ...created "%PROJECT_ROOT_DIR%\conf\log4j.properties"
)

SET SCRIPT_FILE=%1

:: if the script file path was omitted, try to complete the script path
IF NOT EXIST %SCRIPT_FILE% (
  FOR /R "%PROJECT_ROOT_DIR%" %%f IN (%SCRIPT_FILE%) DO IF EXIST %%f ( SET "SCRIPT_FILE_FOUND=%%f" )
)

IF NOT EXIST %SCRIPT_FILE% IF NOT DEFINED SCRIPT_FILE_FOUND (
  echo Could not find DML script: "%SCRIPT_FILE%"
  GOTO Err
) ELSE (
  SET SCRIPT_FILE="%SCRIPT_FILE_FOUND%"
  echo DML script: "%SCRIPT_FILE_FOUND%"
)


:: the hadoop winutils
SET HADOOP_HOME=%PROJECT_ROOT_DIR%\target\lib\hadoop

:: add dependent libraries to classpath (since Java 1.6 we can use wildcards)
set CLASSPATH=%PROJECT_ROOT_DIR%\target\lib\*

:: add compiled SystemML classes to classpath
set CLASSPATH=%CLASSPATH%;%PROJECT_ROOT_DIR%\target\classes


:: remove the DML script file from the list of arguments
:: allow for dml script path with spaces, enclosed in quotes
rem for /f "tokens=1,* delims= " %%a in ("%*") do set DML_OPT_ARGS=%%b
rem for /f tokens^=1^,*^ delims^=^" %%a in ("%*") do set DML_OPT_ARGS=%%b
set ARGS=%*
set DML_OPT_ARGS=!ARGS:%1 =!

echo ================================================================================

:: construct the java command with options and arguments
set CMD=java -Xmx4g -Xms2g -Xmn400m ^
     -cp "%CLASSPATH%" ^
     -Dlog4j.configuration=file:"%PROJECT_ROOT_DIR%\conf\log4j.properties" ^
     -Duser.dir="%USER_DIR%" ^
     org.apache.sysml.api.DMLScript ^
     -f %SCRIPT_FILE% ^
     -exec singlenode ^
     -config="%PROJECT_ROOT_DIR%\conf\SystemML-config.xml" ^
     %DML_OPT_ARGS%

:: execute the java command
%CMD%

:: if there was an error, display the full java command (in case some of the variable substitutions broke it)
IF  ERRORLEVEL 1 (
  ECHO Failed to run SystemML. Exit code: %ERRORLEVEL%
  SET LF=^


  :: keep empty lines above for the line breaks
  ECHO %CMD:      =!LF!     %
  EXIT /B %ERRORLEVEL%
)
GOTO End

:Err
ECHO Wrong Usage. Please provide DML filename to be executed.
GOTO Msg

:Msg
ECHO Usage: runStandaloneSystemML.bat ^<dml-filename^> [arguments] [-help]
ECHO Script internally invokes 'java -Xmx4g -Xms4g -Xmn400m -jar jSystemML.jar -f ^<dml-filename^> -exec singlenode -config=SystemML-config.xml [Optional-Arguments]'
GOTO ExitErr

:ExitErr
EXIT /B 1

:End
EXIT /B 0
