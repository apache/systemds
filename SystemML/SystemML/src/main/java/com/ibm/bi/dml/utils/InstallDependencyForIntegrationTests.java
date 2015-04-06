/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */
package com.ibm.bi.dml.utils;

import static org.junit.Assert.fail;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * This class will be used to setup dependency on Eclipse environment as well as on Jenkins server
 *
 */
public class InstallDependencyForIntegrationTests {

	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public static void main(String[] args) {
		boolean skip = Boolean.parseBoolean(args[0]);
		if(!skip) {
			// This version assumes that R is installed on the server
			runRScript(true);
		}
	}

	/**
	 * Runs an R script in the old or the new way
	 */
	@SuppressWarnings("unused")
	protected static void runRScript(boolean newWay) {
	
		String executionFile =  "./src/test/scripts/installDependencies.R"; 
		
		String cmd;
		if (newWay == false) {
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
			TestUtils.printRScript(executionFile);
		
		try {
			long t0 = System.nanoTime();
			System.out.println("Installing packages required for running integration tests ...");           
			Process child = Runtime.getRuntime().exec(cmd);     
			String outputR = "";
			int c = 0;

			while ((c = child.getInputStream().read()) != -1) {
				System.out.print((char) c);
				outputR += String.valueOf((char) c);
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

			try {
				if (child.exitValue() != 0) {
					throw new Exception("ERROR: R has ended irregularly\n" + outputR + "\nscript file: "
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
}
