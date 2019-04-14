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

package org.tugraz.sysds.test.functions.frame;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.io.FrameReaderFactory;
import org.tugraz.sysds.runtime.io.FrameWriterFactory;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class FrameMetaReadWriteTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_NAME = "FrameMetaReadWrite";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameMetaReadWriteTest.class.getSimpleName() + "/";
	
	private final static int rows = 1382;
	private final static int cols = 7;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test
	public void testFrameBinaryCP()  {
		runFrameReadWriteTest(OutputInfo.BinaryBlockOutputInfo, ExecType.CP);
	}

	@Test
	public void testFrameBinarySpark()  {
		runFrameReadWriteTest(OutputInfo.BinaryBlockOutputInfo, ExecType.SPARK);
	}
	
	@Test
	public void testFrameTextcellCP()  {
		runFrameReadWriteTest(OutputInfo.TextCellOutputInfo, ExecType.CP);
	}
	
	@Test
	public void testFrameTextcellSpark()  {
		runFrameReadWriteTest(OutputInfo.TextCellOutputInfo, ExecType.SPARK);
	}
	
	@Test
	public void testFrameCsvCP()  {
		runFrameReadWriteTest(OutputInfo.CSVOutputInfo, ExecType.CP);
	}

	@Test
	public void testFrameCsvSpark()  {
		runFrameReadWriteTest(OutputInfo.CSVOutputInfo, ExecType.SPARK);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runFrameReadWriteTest( OutputInfo oinfo, ExecType et)
	{
		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
	
		String ofmt = OutputInfo.outputInfoToStringExternal(oinfo);
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"), 
					String.valueOf(rows), String.valueOf(cols), ofmt, output("B") };
			
			//data generation and write input
			double[][] A = getRandomMatrix(rows, cols, -10, 10, 0.7, 3412); 
			FrameBlock fA = DataConverter.convertToFrameBlock(
					DataConverter.convertToMatrixBlock(A), ValueType.STRING);
			for( int j=0; j<cols; j++ ) {
				fA.getColumnMetadata(j).setMvValue(String.valueOf(j+1));
				fA.getColumnMetadata(j).setNumDistinct(j+1);
			}
			FrameWriterFactory.createFrameWriter(oinfo)
				.writeFrameToHDFS(fA, input("A"), rows, cols);
			
			//run testcase
			runTest(true, false, null, -1);
			
			//read output and compare meta data
			FrameBlock fB = FrameReaderFactory
					.createFrameReader(OutputInfo.getMatchingInputInfo(oinfo))
					.readFrameFromHDFS(output("B"), rows, cols);
			for( int j=0; j<cols; j++ ) {
				Assert.assertEquals("MV meta data wrong!",
						fA.getColumnMetadata(j).getMvValue(), fB.getColumnMetadata(j).getMvValue());
				Assert.assertEquals("Distinct meta data wrong!",
						fA.getColumnMetadata(j).getNumDistinct(), fB.getColumnMetadata(j).getNumDistinct());
			}
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
