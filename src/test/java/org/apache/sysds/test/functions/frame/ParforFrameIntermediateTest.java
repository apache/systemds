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

package org.apache.sysds.test.functions.frame;

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class ParforFrameIntermediateTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_NAME = "ParforFrameIntermediates";
	private final static String TEST_CLASS_DIR = TEST_DIR + ParforFrameIntermediateTest.class.getSimpleName() + "/";
	
	private final static int rows = 1382;
	private final static int cols = 5;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"F2"}));
	}

	@Test
	public void testParforFrameIntermediatesCP()  {
		runParforFrameIntermediatesTest(ExecType.CP);
	}
	
	@Test
	public void testParforFrameIntermediatesSpark()  {
		runParforFrameIntermediatesTest(ExecType.SPARK);
	}
	
	private void runParforFrameIntermediatesTest( ExecType et ) {
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK 
			|| rtplatform == ExecMode.HYBRID )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			//setup testcase
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args", input("F")};
			
			//generate input data and write as frame
			double[][] A = getRandomMatrix(rows, cols, -10, 10, 0.9, 8362);
			FrameBlock fA = DataConverter.convertToFrameBlock(
				DataConverter.convertToMatrixBlock(A));
			FrameWriterFactory.createFrameWriter(FileFormat.CSV)
				.writeFrameToHDFS(fA, input("F"), rows, cols);
			
			//run test
			runTest(true, false, null, -1); 
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
