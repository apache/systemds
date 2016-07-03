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

package org.apache.sysml.test.integration.functions.frame;


import java.util.Arrays;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.io.FrameReaderFactory;
import org.apache.sysml.runtime.io.FrameWriterFactory;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Test;

/**
 * 
 */
public class FrameScalarCastingTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_NAME1 = "Frame2ScalarCast";
	private final static String TEST_NAME2 = "Scalar2FrameCast";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameScalarCastingTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"B"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"B"}));		
	}
	
	@Test
	public void testFrame2ScalarString() {
		runFrameCastingTest(TEST_NAME1, ValueType.STRING);
	}
	
	@Test
	public void testFrame2ScalarDouble() {
		runFrameCastingTest(TEST_NAME1, ValueType.DOUBLE);
	}
	
	@Test
	public void testFrame2ScalarBoolean() {
		runFrameCastingTest(TEST_NAME1, ValueType.BOOLEAN);
	}
	
	@Test
	public void testFrame2ScalarInt() {
		runFrameCastingTest(TEST_NAME1, ValueType.INT);
	}
	
	@Test
	public void testScalar2FrameString() {
		runFrameCastingTest(TEST_NAME2, ValueType.STRING);
	}
	
	@Test
	public void testScalar2FrameDouble() {
		runFrameCastingTest(TEST_NAME2, ValueType.DOUBLE);
	}
	
	@Test
	public void testScalar2FrameBoolean() {
		runFrameCastingTest(TEST_NAME2, ValueType.BOOLEAN);
	}
	
	@Test
	public void testScalar2FrameInt() {
		runFrameCastingTest(TEST_NAME2, ValueType.INT);
	}
	
	/**
	 * 
	 * @param testname
	 * @param schema
	 * @param wildcard
	 */
	private void runFrameCastingTest( String testname, ValueType vt)
	{	
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"), vt.toString(), output("B") };
			
			//input data and compare
			FrameBlock fb = new FrameBlock(1, vt);
			Object inval = UtilFunctions.objectToObject(vt, 7);
			fb.ensureAllocatedColumns(1);
			fb.set(0, 0, inval);
			
			//write inputs
			if( testname.equals(TEST_NAME1) )
				FrameWriterFactory.createFrameWriter(OutputInfo.TextCellOutputInfo)
					.writeFrameToHDFS(fb, input("A"), 1, 1);
			else
				MapReduceTool.writeObjectToHDFS(inval, input("A"));
			
			//run testcase
			runTest(true, false, null, -1);
			
			//read and compare scalars
			Object retval = null;
			if( testname.equals(TEST_NAME1) ) {
				retval = MapReduceTool.readObjectFromHDFSFile(output("B"), vt);
			}
			else {
				retval = FrameReaderFactory.createFrameReader(InputInfo.TextCellInputInfo)
					.readFrameFromHDFS(output("B"), Arrays.asList(vt), 1, 1)
					.get(0, 0);
			}
			Assert.assertEquals("Wrong output: "+retval+" (expected: "+inval+")", inval, retval);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}
}
