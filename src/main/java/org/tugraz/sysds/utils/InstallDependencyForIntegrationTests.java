/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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
 package org.tugraz.sysds.utils;

import static org.junit.Assert.fail;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import org.tugraz.sysds.runtime.DMLRuntimeException;

/**
 * This class will be used to setup dependency on Eclipse environment as well as on Jenkins server
 *
 */
public class InstallDependencyForIntegrationTests {


	public static void main(String[] args) throws IOException, DMLRuntimeException, InterruptedException {
		boolean skip = Boolean.parseBoolean(args[0]);
		if(!skip) {
			// This version assumes that R is installed on the server
			runRScript(true);
		}
		if(args.length >= 2) {
			boolean enableGPU = Boolean.parseBoolean(args[1]);
			if(enableGPU) {
				String baseDir = "src" + File.separator + "main" + File.separator + "cpp" + File.separator + "kernels" + File.separator;
				System.out.println("==============================================");
				System.out.println("Compiling the kernels into ptx");
				String cmd = "nvcc -ptx " + baseDir + "SystemDS.cu -o " + baseDir + "SystemDS.ptx";
				Process child = runCommand(cmd);
				System.out.println("==============================================");
				try {
					if (child.exitValue() != 0) {
						throw new DMLRuntimeException("Compiling the kernels into ptx was unsuccessful.");
					}
				} catch (IllegalThreadStateException ie) {
					child.destroy();
				}
				if(!new File(baseDir + "SystemDS.ptx").exists()) {
					throw new DMLRuntimeException("Compiling the kernels into ptx was unsuccessful.");
				}
			}
		}
		
	}
	
	private static Process runCommand(String cmd) throws IOException, InterruptedException {
		Process child = Runtime.getRuntime().exec(cmd);     
		int c = 0;

		while ((c = child.getInputStream().read()) != -1) {
			System.out.print((char) c);
		}
		while ((c = child.getErrorStream().read()) != -1) {
			System.err.print((char) c);
		}

		//
		// To give any stream enough time to print all data, otherwise there
		// are situations where the test case fails, even before everything
		// has been printed
		//
		child.waitFor();
		return child;
	}

	/**
	 * Runs an R script in the old or the new way
	 * 
	 * @param newWay if true, run R script the new way
	 */
	@SuppressWarnings("unused")
	protected static void runRScript(boolean newWay) {
	
		String executionFile =  "./src/test/scripts/installDependencies.R"; 
		
		String cmd;
		if ( !newWay ) {
			executionFile = executionFile + "t";
			cmd = "R -f " + executionFile;
		}
		else {
			cmd = "Rscript" + " " + executionFile;
		}
		
		if (System.getProperty("os.name").contains("Windows")) {
			cmd = cmd.replace('/', '\\');                        
			executionFile = executionFile.replace('/', '\\');
		}
		if ( !newWay )	
			printRScript(executionFile); // TestUtils.printRScript(executionFile);
		
		try {
			long t0 = System.nanoTime();
			System.out.println("Installing packages required for running integration tests ...");           
			
			Process child = runCommand(cmd);
			try {
				if (child.exitValue() != 0) {
					throw new Exception("ERROR: R has ended irregularly.\nscript file: "
							+ executionFile);
				}
			} catch (IllegalThreadStateException ie) {
				//
				// In UNIX JVM does not seem to be able to close threads
				// correctly. However, give it a try, since R processed the
				// script, therefore we can terminate the process.
				//
				child.destroy();
			}

			// long t1 = System.nanoTime();
			// System.out.println("R is finished (in "+((double)t1-t0)/1000000000+" sec)");
			System.out.println("Done installing packages required for running integration tests.");

		} catch (Exception e) {
			e.printStackTrace();
			StringBuilder errorMessage = new StringBuilder();
			errorMessage.append("failed to run script " + executionFile);
			errorMessage.append("\nexception: " + e.toString());
			errorMessage.append("\nmessage: " + e.getMessage());
			errorMessage.append("\nstack trace:");
			for (StackTraceElement ste : e.getStackTrace()) {
				errorMessage.append("\n>" + ste);
			}
			fail(errorMessage.toString());
		}
	}
	
	/**
	 * <p>
	 * Prints out an R script.
	 * </p>
	 * 
	 * @param dmlScriptFile
	 *            filename of R script
	 */
	public static void printRScript(String dmlScriptFile) {
		try {
			System.out.println("Running script: " + dmlScriptFile + "\n");
			System.out.println("******************* R script *******************");
			try( BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(dmlScriptFile)))) {
				String content;
				while ((content = in.readLine()) != null) {
					System.out.println(content);
				}
			}
			System.out.println("**************************************************\n\n");
		} catch (IOException e) {
			e.printStackTrace();
			fail("unable to print R script: " + e.getMessage());
		}
	}
}
