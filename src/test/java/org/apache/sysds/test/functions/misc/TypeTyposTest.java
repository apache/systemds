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

import org.junit.Assert;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.apache.sysds.parser.ParseException;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class TypeTyposTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_NAME1 = "WrongValueType";
	private final static String TEST_NAME2 = "WrongDataType";
	private final static String TEST_CLASS_DIR = TEST_DIR + TypeTyposTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {}));
	}
	
	@Test
	public void testValueTypes() {
		runTest( TEST_NAME1, true );
	}
	
	@Test
	public void testDataTypes() {
		runTest( TEST_NAME2, true );
	}
	
	private void runTest( String testName, boolean exExp ) {
		TestConfiguration config = getTestConfiguration(TEST_NAME1);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testName + ".dml";
		programArgs = new String[]{"-explain"};
		
		PrintStream old = System.err;
		String out = null;
		try {
			ByteArrayOutputStream buff = new ByteArrayOutputStream();
			System.setErr(new PrintStream(buff));
			runTest(true, exExp, ParseException.class, -1);
			out = buff.toString();
		}
		finally {
			System.setErr(old);
		}
		if( testName.equals(TEST_NAME1) )
			Assert.assertTrue(out.contains("invalid valuetype"));
		else if(testName.equals(TEST_NAME2))
			Assert.assertTrue(out.contains("incorrect datatype"));
	}
}
