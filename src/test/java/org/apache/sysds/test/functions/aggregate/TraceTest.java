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

package org.apache.sysds.test.functions.aggregate;

import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

/**
 * <p>
 * <b>Positive tests:</b>
 * </p>
 * <ul>
 * <li>general test</li>
 * </ul>
 * <p>
 * <b>Negative tests:</b>
 * </p>
 * <ul>
 * <li>scalar test</li>
 * </ul>
 * 
 * 
 */
public class TraceTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/aggregate/";
	private static final String TEST_CLASS_DIR = TEST_DIR + TraceTest.class.getSimpleName() + "/";
	private final static String TEST_GENERAL = "TraceTest";
	private final static String TEST_SCALAR = "TraceScalarTest";
	private final static String TEST_INVALID1 = "TraceInvalid1";
	private final static String TEST_INVALID2 = "TraceInvalid2";

	@Override
	public void setUp() {
		// positive tests
		addTestConfiguration(TEST_GENERAL, new TestConfiguration(TEST_CLASS_DIR, TEST_GENERAL, new String[] {"b"}));

		// negative tests
		addTestConfiguration(TEST_SCALAR, new TestConfiguration(TEST_CLASS_DIR, TEST_SCALAR, new String[] {"b"}));
		addTestConfiguration(TEST_INVALID1, new TestConfiguration(TEST_CLASS_DIR, TEST_INVALID1, new String[] {"b"}));
		addTestConfiguration(TEST_INVALID2, new TestConfiguration(TEST_CLASS_DIR, TEST_INVALID2, new String[] {"b"}));
	}

	@Test
	public void testGeneral() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = getTestConfiguration(TEST_GENERAL);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		loadTestConfiguration(config);

		createHelperMatrix();

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		writeInputMatrix("a", a);

		double b = 0;
		for(int i = 0; i < rows; i++) {
			b += a[i][i];
		}
		writeExpectedHelperMatrix("b", b);

		runTest();

		compareResults(1e-14);
	}

	@Test
	public void testScalar() {
		TestConfiguration config = getTestConfiguration(TEST_SCALAR);
		config.addVariable("scalar", 12);

		createHelperMatrix();
		loadTestConfiguration(config);
		runTest(true, LanguageException.class);
	}
	
	@Test
	public void testInvalid1() {
		TestConfiguration config = getTestConfiguration(TEST_INVALID1);
		loadTestConfiguration(config);
		runTest(true, LanguageException.class);
	}

	@Test
	public void testInvalid2() {
		TestConfiguration config = getTestConfiguration(TEST_INVALID2);
		loadTestConfiguration(config);
		runTest(true, LanguageException.class);
	}
}
