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

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.parser.ParseException;
import org.apache.sysds.runtime.util.LocalFileUtils;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class SetWorkingDirTest extends AutomatedTestBase {
	// Force Logging to error level on API file, to enforce test.
	static {
		Logger.getLogger("org.apache.sysds.api.DMLScript").setLevel(Level.ERROR);
	}

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

		try {
			FileUtils.copyFile(new File(dir + nameCall), new File(nameCall));
			if(!fileMissingTest)
				FileUtils.copyFile(new File(dir + nameLib), new File(nameLib));
		}
		catch(IOException e) {
			fail("Failed due to IO Exception: " + e.getMessage());
		}

		// setup test configuration
		TestConfiguration config = getTestConfiguration(testName);
		fullDMLScriptName = nameCall;
		programArgs = new String[] {};
		loadTestConfiguration(config);

		runTest(true, fileMissingTest, fileMissingTest ? ParseException.class : null, -1);

		LocalFileUtils.deleteFileIfExists(nameCall);
		if(!fileMissingTest)
			LocalFileUtils.deleteFileIfExists(nameLib);
	}
}
