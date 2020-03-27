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

package org.apache.sysds.test.functions.misc;

import static org.junit.Assert.fail;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.apache.sysds.runtime.util.LocalFileUtils;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

/**
 *   
 */
public class SetWorkingDirTest extends AutomatedTestBase {
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_NAME1 = "PackageFunCall1";
	private final static String TEST_NAME2 = "PackageFunCall2";
	private final static String TEST_NAME0 = "PackageFunLib";
	private static final String TEST_CLASS_DIR = TEST_DIR + SetWorkingDirTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {}));
	}

	@Test
	public void testDefaultWorkingDirDml() {
		runTest(TEST_NAME1, false);
	}

	@Test
	public void testSetWorkingDirDml() {
		runTest(TEST_NAME2, false);
	}

	@Test
	public void testDefaultWorkingDirFailDml() {
		runTest(TEST_NAME1, true);
	}

	@Test
	public void testSetWorkingDirFailDml() {
		runTest(TEST_NAME2, true);
	}

	/**
	 * 
	 * @param testName
	 * @param fileMissingTest
	 * @param scriptType
	 */
	private void runTest(String testName, boolean fileMissingTest) {

		// construct source filenames of dml scripts
		String dir = SCRIPT_DIR + TEST_DIR;
		String nameCall = testName + ".dml";
		String nameLib = TEST_NAME0 + ".dml";

		PrintStream originalStdErr = System.err;

		try {
			ByteArrayOutputStream baos = null;
			if (fileMissingTest) {
				baos = new ByteArrayOutputStream();
				PrintStream newStdErr = new PrintStream(baos);
				System.setErr(newStdErr);
			}

			// copy dml/pydml scripts to current dir
			FileUtils.copyFile(new File(dir + nameCall), new File(nameCall));
			if (!fileMissingTest)
				FileUtils.copyFile(new File(dir + nameLib), new File(nameLib));

			// setup test configuration
			TestConfiguration config = getTestConfiguration(testName);
			fullDMLScriptName = nameCall;
			programArgs = new String[] {};
			loadTestConfiguration(config);

			// run tests
			runTest(true, false, null, -1);

			if (fileMissingTest) {
				String stdErrString = baos.toString();
				if (stdErrString == null) {
					fail("Standard error string is null"); // shouldn't happen
				} else if (!stdErrString.contains("Cannot find file")) {
					// the error message should contain "Cannot find file" if file is missing
					fail("Should not be able to find file: " + nameLib);
				}
				if (stdErrString != null) {
					originalStdErr.println(stdErrString); // send standard err string to console
				}
			}

		} catch (IOException e) {
			throw new RuntimeException(e);
		} finally {
			System.setErr(originalStdErr);
			// delete dml/pydml scripts from current dir (see above)
			LocalFileUtils.deleteFileIfExists(nameCall);
			if (!fileMissingTest)
				LocalFileUtils.deleteFileIfExists(nameLib);
		}
	}
}
