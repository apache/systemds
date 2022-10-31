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

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class FrameMetaReadWriteTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_NAME = "FrameMetaReadWrite";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameMetaReadWriteTest.class.getSimpleName() + "/";
	
	private final static int rows = 1382;
	private final static int cols = 7;
	private final static String[] colNames = new String[] {"A","B","C","D","E","F","G"};
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test
	public void testFrameBinaryCP()  {
		runFrameReadWriteTest(FileFormat.BINARY, ExecType.CP);
	}

	@Test
	public void testFrameBinarySpark()  {
		runFrameReadWriteTest(FileFormat.BINARY, ExecType.SPARK);
	}
	
	@Test
	public void testFrameTextcellCP()  {
		runFrameReadWriteTest(FileFormat.TEXT, ExecType.CP);
	}
	
	@Test
	public void testFrameTextcellSpark()  {
		runFrameReadWriteTest(FileFormat.TEXT, ExecType.SPARK);
	}
	
	@Test
	public void testFrameCsvCP()  {
		runFrameReadWriteTest(FileFormat.CSV, ExecType.CP);
	}

	@Test
	public void testFrameCsvSpark()  {
		runFrameReadWriteTest(FileFormat.CSV, ExecType.SPARK);
	}
	
	private void runFrameReadWriteTest(FileFormat fmt, ExecType et)
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
	
		String ofmt = fmt.toString();
		setOutputBuffering(true);
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
			fA.setColumnNames(colNames);
			for( int j=0; j<cols; j++ ) {
				fA.getColumnMetadata(j).setMvValue(String.valueOf(j+1));
				fA.getColumnMetadata(j).setNumDistinct(j+1);
			}
			FrameWriterFactory.createFrameWriter(fmt)
				.writeFrameToHDFS(fA, input("A"), rows, cols);
			
			//run testcase
			runTest(true, false, null, -1);
			
			//read output and compare meta data
			FrameBlock fB = FrameReaderFactory
				.createFrameReader(fmt)
				.readFrameFromHDFS(output("B"), rows, cols);
			for( int j=0; j<cols; j++ ) {
				Assert.assertEquals("MV meta data wrong!",
					fA.getColumnMetadata(j).getMvValue(), fB.getColumnMetadata(j).getMvValue());
				Assert.assertEquals("Distinct meta data wrong!",
					fA.getColumnMetadata(j).getNumDistinct(), fB.getColumnMetadata(j).getNumDistinct());
			}
			if( fmt == FileFormat.BINARY )
				Assert.assertArrayEquals("Column names wrong!", fA.getColumnNames(), fB.getColumnNames());
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
