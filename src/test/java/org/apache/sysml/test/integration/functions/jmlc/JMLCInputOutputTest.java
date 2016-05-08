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

package org.apache.sysml.test.integration.functions.jmlc;

import java.io.File;
import java.io.IOException;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.jmlc.Connection;
import org.apache.sysml.api.jmlc.PreparedScript;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.junit.Test;

/**
 * Test input and output capabilities of JMLC API.
 *
 */
public class JMLCInputOutputTest extends AutomatedTestBase {
	private final static String TEST_NAME = "JMLCInputOutputTest";
	private final static String TEST_DIR = "functions/jmlc/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testScalarInputInt() throws IOException, DMLException {
		Connection conn = new Connection();
		String str = conn.readScript(baseDirectory + File.separator + "scalar-input.dml");
		PreparedScript script = conn.prepareScript(str, new String[] { "inScalar1", "inScalar2" }, new String[] {},
				false);
		int inScalar1 = 2;
		int inScalar2 = 3;
		script.setScalar("inScalar1", inScalar1);
		script.setScalar("inScalar2", inScalar2);

		setExpectedStdOut("total:5");
		script.executeScript();
		conn.close();
	}

	@Test
	public void testScalarInputDouble() throws IOException, DMLException {
		Connection conn = new Connection();
		String str = conn.readScript(baseDirectory + File.separator + "scalar-input.dml");
		PreparedScript script = conn.prepareScript(str, new String[] { "inScalar1", "inScalar2" }, new String[] {},
				false);
		double inScalar1 = 3.5;
		double inScalar2 = 4.5;
		script.setScalar("inScalar1", inScalar1);
		script.setScalar("inScalar2", inScalar2);

		setExpectedStdOut("total:8.0");
		script.executeScript();
		conn.close();
	}

	// See: https://issues.apache.org/jira/browse/SYSTEMML-656
	// @Test
	// public void testScalarInputBoolean() throws IOException, DMLException {
	// Connection conn = new Connection();
	// String str = conn.readScript(baseDirectory + File.separator +
	// "scalar-input.dml");
	// PreparedScript script = conn.prepareScript(str, new String[] {
	// "inScalar1", "inScalar2" }, new String[] {},
	// false);
	// boolean inScalar1 = true;
	// boolean inScalar2 = true;
	// script.setScalar("inScalar1", inScalar1);
	// script.setScalar("inScalar2", inScalar2);
	//
	// setExpectedStdOut("total:TRUE");
	// script.executeScript();
	// }

	@Test
	public void testScalarInputLong() throws IOException, DMLException {
		Connection conn = new Connection();
		String str = conn.readScript(baseDirectory + File.separator + "scalar-input.dml");
		PreparedScript script = conn.prepareScript(str, new String[] { "inScalar1", "inScalar2" }, new String[] {},
				false);
		long inScalar1 = 4;
		long inScalar2 = 5;
		script.setScalar("inScalar1", inScalar1);
		script.setScalar("inScalar2", inScalar2);

		setExpectedStdOut("total:9");
		script.executeScript();
		conn.close();
	}

	// See: https://issues.apache.org/jira/browse/SYSTEMML-658
	// @Test
	// public void testScalarInputString() throws IOException, DMLException {
	// Connection conn = new Connection();
	// String str = conn.readScript(baseDirectory + File.separator +
	// "scalar-input.dml");
	// PreparedScript script = conn.prepareScript(str, new String[] {
	// "inScalar1", "inScalar2" }, new String[] {},
	// false);
	// String inScalar1 = "hello";
	// String inScalar2 = "goodbye";
	// script.setScalar("inScalar1", inScalar1);
	// script.setScalar("inScalar2", inScalar2);
	//
	// setExpectedStdOut("total:hellogoodbye");
	// script.executeScript();
	// }

	@Test
	public void testScalarInputStringExplicitValueType() throws IOException, DMLException {
		Connection conn = new Connection();
		String str = conn.readScript(baseDirectory + File.separator + "scalar-input-string.dml");
		PreparedScript script = conn.prepareScript(str, new String[] { "inScalar1", "inScalar2" }, new String[] {},
				false);
		String inScalar1 = "hello";
		String inScalar2 = "goodbye";
		script.setScalar("inScalar1", inScalar1);
		script.setScalar("inScalar2", inScalar2);

		setExpectedStdOut("result:hellogoodbye");
		script.executeScript();
		conn.close();
	}

}