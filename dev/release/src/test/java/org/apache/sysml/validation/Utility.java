/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.    
 */

package org.apache.sysml.validation;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.lang.Process;
import java.lang.ProcessBuilder;
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;


/**
 * This is an utility class to include any utility functios.
 */
public class Utility
{
	/**
	 * This will be used to output on console in consistent and controlled based on DEBUG flag.
	 *
	 * @param	debugLevel is the debuglevel message user wants to prrint message.
	 * @param 	message is the message to be displayed.
	 * @return
	 */
	public static void debugPrint(int debugLevel, String message) {
		debugPrint(debugLevel, message, null);
	}

	/**
	 * This will be used to output on console in consistent and controlled based on DEBUG flag.
	 *
	 * @param	debugLevel is the debuglevel message user wants to prrint message.
	 * @param 	message is the message to be displayed.
	 * @param   strOutputFile is the filename to print output.
	 * @return
	 */
	public static void debugPrint(int debugLevel, String message, String strOutputFile) {

		if(debugLevel > Constants.DEBUG_PRINT_LEVEL)
			return;

		String displayMessage = "";
		switch (debugLevel) {
			case Constants.DEBUG_ERROR:
				displayMessage = "ERROR: " + message;
				break;
			case Constants.DEBUG_WARNING:
				displayMessage = "WARNING: " + message;
				break;
			case Constants.DEBUG_INFO:
				displayMessage = "INFO: " + message;
				break;
			case Constants.DEBUG_INFO2:
				displayMessage = "INFO2: " + message;
				break;
			case Constants.DEBUG_INFO3:
				displayMessage = "INFO3: " + message;
				break;
			case Constants.DEBUG_CODE:
				displayMessage = "DEBUG: " + message;
				break;
			default:
				break;
		}

		LocalDateTime currentDateTime = LocalDateTime.now();
		String strDateTime = currentDateTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);

		if(strOutputFile == null || strOutputFile.trim().length() == 0)
			System.out.println(strDateTime + ": " + displayMessage);
		else {
			PrintWriter printWriter = null;
			try {
				printWriter = new PrintWriter(new BufferedWriter(new FileWriter(strOutputFile, true)));
				printWriter.println(strDateTime + ": " + displayMessage);
			} catch (IOException ioe) {
				debugPrint(Constants.DEBUG_ERROR, "Exception while printing message to file " + strOutputFile + ioe);
				debugPrint(debugLevel, message);
			}
			if(printWriter != null)
				printWriter.close();
		}
	}

	/**
	 * This function will run command specified abd redirect stdout/stderr per instruction.
	 *
	 * @param
	 * @param
	 * @return
	 */
	public static int runCommand(String [] command, String strCurDir, String strOutputFile, String strErrorFile, String strMessage)
		throws IOException
	{
		if(strMessage != null && strMessage.trim().length() > 0)
			debugPrint(Constants.DEBUG_INFO, strMessage, strOutputFile);

		debugPrint(Constants.DEBUG_CODE, "Running command: '" + String.join(" ", command) + "'", strOutputFile);

		ProcessBuilder processBuilder = new ProcessBuilder(command);

		// Set current working directory for command to run.
		if(strCurDir != null && strCurDir.trim().length() > 0)
			processBuilder.directory(new File(strCurDir));

		// Start the process
		Process process = processBuilder.start();

		// Read the output from runtime and redirect to file/stdout
		InputStreamReader insIn = new InputStreamReader(process.getInputStream());
		BufferedReader bufIn = new BufferedReader(insIn);

		boolean bWriteOutToFile = false;
		PrintWriter writerOut = null;
		try {
			if(strOutputFile != null && strOutputFile.trim().length() > 0)
				bWriteOutToFile = true;
			if(bWriteOutToFile)
				writerOut = new PrintWriter(new BufferedWriter(new FileWriter(strOutputFile,true)));

			// Read standard output after running the command.
			String strLine;
			while ((strLine = bufIn.readLine()) != null) {
				if (bWriteOutToFile)
					writerOut.println(strLine);
				else
					debugPrint(Constants.DEBUG_INFO2, "StdOut: " + strLine);
			}
		} catch (IOException ioe) {
			debugPrint(Constants.DEBUG_ERROR, "Exception occured while reading from process output: " + ioe, strOutputFile);
		}
		if(bWriteOutToFile && writerOut != null)
			writerOut.close();

		// Read the output from runtime and redirect to file/stdout
		InputStreamReader insErr = new InputStreamReader(process.getErrorStream());
		BufferedReader bufErr = new BufferedReader(insErr);

		boolean bWriteErrToFile = false;
		PrintWriter writerErr = null;
		try {
			if(strErrorFile != null && strErrorFile.trim().length() > 0)
				bWriteErrToFile = true;
			if(bWriteErrToFile)
				writerErr = new PrintWriter(new BufferedWriter(new FileWriter(strErrorFile,true)));

			// Read standard output after running the command.
			String strLine;
			while ((strLine = bufErr.readLine()) != null) {
				if (bWriteErrToFile)
					writerErr.println(strLine);
				else
					debugPrint(Constants.DEBUG_INFO2, "StdErr: " + strLine);
			}
		} catch (IOException ioe) {
			debugPrint(Constants.DEBUG_ERROR, "Exception occured while reading from process error stream: " + ioe, strOutputFile);
		}
		if(bWriteErrToFile && writerErr != null)
			writerErr.close();

		// Wait for program to exit.
		int exitValue = -1;
		try {
			exitValue = process.waitFor();
		} catch (InterruptedException ie) {
			debugPrint(Constants.DEBUG_ERROR, "Program interrunpted: " + ie);
		}
		debugPrint(Constants.DEBUG_CODE, "Program '" + String.join(" ", command) + "' exited with exit status " + exitValue, strOutputFile);

		return exitValue;
	}

}
