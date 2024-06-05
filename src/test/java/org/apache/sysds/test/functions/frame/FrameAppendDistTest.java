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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.BinaryOp.AppendMethod;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class FrameAppendDistTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "FrameAppend";
	private final static String TEST_NAME2 = "FrameNAryAppend";
	private final static String TEST_NAME3 = "FrameNAryAppendMisalign";
	private final static String TEST_NAME4 = "FrameNAryAppendMisalignRSP";
	private final static String TEST_NAME5 = "FrameNAryAppendMisalignRSP2";

	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameAppendDistTest.class.getSimpleName() + "/";

	private final static double epsilon=0.0000000001;
	private final static int min=1;
	private final static int max=100;
	
	private final static int rows1 = 1692;
	private final static int rows2 = 1192;
	//usecase a: inblock single
	private final static int cols1a = 375;
	private final static int cols2a = 92;
	//usecase b: inblock multiple
	private final static int cols1b = 1059;
	//usecase c: outblock blocksize 
	private final static int cols1d = 1460;
	private final static int cols3d = 990;
	
		
	private final static double sparsity1 = 0.5;
	private final static double sparsity2 = 0.01;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"C"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2,new String[] {"C"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3,new String[] {"C"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4,new String[] {"C"}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5,new String[] {"C"}));
	}

	@Test
	public void testAppendInBlock1DenseSP() {
		commonAppendTest(ExecMode.SPARK, rows1, rows1, cols1a, cols2a, false, AppendMethod.MR_RAPPEND, false, TEST_NAME);
	}
	
	@Test
	public void testAppendInBlock1SparseSP() {
		commonAppendTest(ExecMode.SPARK, rows1, rows1, cols1a, cols2a, true, AppendMethod.MR_RAPPEND, false, TEST_NAME);
	}
	
	@Test
	public void testAppendInBlock1DenseRBindSP() {
		commonAppendTest(ExecMode.SPARK, rows1, rows2, cols1a, cols1a, false, AppendMethod.MR_RAPPEND, true, TEST_NAME);
	}
	
	@Test
	public void testAppendInBlock1SparseRBindSP() {
		commonAppendTest(ExecMode.SPARK, rows1, rows1, cols1a, cols1a, true, AppendMethod.MR_RAPPEND, true, TEST_NAME);
	}
	
	//NOTE: mappend only applied for m2_cols<=blocksize
	@Test
	public void testMapAppendInBlock2DenseSP() {
		commonAppendTest(ExecMode.SPARK, rows1, rows1, cols1b, cols2a, false, AppendMethod.MR_MAPPEND, false, TEST_NAME);
	}
	
	@Test
	public void testMapAppendInBlock2SparseSP() {
		commonAppendTest(ExecMode.SPARK, rows1, rows1, cols1b, cols2a, true, AppendMethod.MR_MAPPEND, false, TEST_NAME);
	}
	
	@Test
	public void testMapAppendOutBlock2DenseSP() {
		commonAppendTest(ExecMode.SPARK, rows1, rows1, cols1d, cols3d, false, AppendMethod.MR_MAPPEND, false, TEST_NAME);
	}
	
	@Test
	public void testMapAppendOutBlock2SparseSP() {
		commonAppendTest(ExecMode.SPARK, rows1, rows1, cols1d, cols3d, true, AppendMethod.MR_MAPPEND, false, TEST_NAME);
	}

	@Test
	public void testNAryCAppendMSP(){
		commonAppendTest(ExecMode.SPARK ,100, 100, 5, 10, false, null, false, TEST_NAME2);
	}

	@Test
	public void testNAryCAppendRSP(){
		commonAppendTest(ExecMode.SPARK ,30, 30, 5, 1001, false, null, false, TEST_NAME2);
	}

	@Test
	public void testNAryRAppendSP(){
		commonAppendTest(ExecMode.SPARK ,100, 100, 5, 5, false, null, true, TEST_NAME2);
	}

	@Test
	public void testNAryAppendWithMisalignmentMSP(){
		commonAppendTest(ExecMode.SPARK ,5, 10, 5, 5, false, null, false, TEST_NAME3);
	}

	@Test
	public void testNAryAppendWithMisalignmentRSP() {
		commonAppendTest(ExecMode.SPARK, 5, 10, 1001, 1001, false, null, false, TEST_NAME3);
	}

// NAryAppendWithMisalignmentRSP2:
// LHS:                RHS:
// +---------+         +-----+
// |         |         +-----+
// |         |         +-----+
// |         |         +-----+
// +---------+         +-----+
	@Test
	public void testNAryAppendWithMisalignmentRSP2(){
		commonAppendTest(ExecMode.SPARK ,20, 5, 1001, 1005, false, null, false, TEST_NAME4);
	}

// NAryAppendWithMisalignmentRSP3:
//      LHS:            RHS:
//      +-----+         +---------+
//      +-----+         |         |
//      +-----+         |         |
//      +-----+         |         |
//      +-----+         +---------+
	@Test
	public void testNAryAppendWithMisalignmentRSP3(){
		commonAppendTest(ExecMode.SPARK ,5, 20, 1001, 1005, false, null, false, TEST_NAME4);
	}
// NAryAppendWithMisalignmentRSP4:
//      LHS:            RHS:
//      +-----+         +---------+
//      |     |         +---------+
//      +-----+         |         |
//      |     |         +---------+
//      +-----+         |         |
//      |     |         +---------+
//      +-----+         |         |
//      +-----+         +---------+
	@Test
	public void testNAryAppendWithMisalignmentRSP4(){
		commonAppendTest(ExecMode.SPARK ,20, 5, 1001, 1001, false, null, false, TEST_NAME5);
	}
// NAryAppendWithMisalignmentRSP5:
//      LHS:            RHS:
//      +-----+         +---------+
//      +-----+         |         |
//      +-----+         +---------+
//      +-----+         +---------+
//      |     |         +---------+
//      +-----+         +---------+
	@Test
	public void testNAryAppendWithMisalignmentRSP5(){
		commonAppendTest(ExecMode.SPARK ,8, 20, 1001, 1001, false, null, false, TEST_NAME5);
	}

	
	public void commonAppendTest(ExecMode platform, int rows1, int rows2, int cols1, int cols2, boolean sparse,
		AppendMethod forcedAppendMethod, boolean rbind, String test_name)
	{
		TestConfiguration config = getAndLoadTestConfiguration(test_name);
		
		ExecMode prevPlfm=rtplatform;
		
		double sparsity = (sparse) ? sparsity2 : sparsity1;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		//setOutputBuffering(true);
		try
		{
			if(forcedAppendMethod != null) {
				BinaryOp.FORCED_APPEND_METHOD = forcedAppendMethod;
			}
			rtplatform = platform;
			if( rtplatform == ExecMode.SPARK )
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
	
			config.addVariable("rows", rows1);
			config.addVariable("cols", cols1);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + test_name + ".dml";
			programArgs = new String[]{"-explain","-args",  input("A"),
				Long.toString(rows1), Long.toString(cols1), input("B"),
				Long.toString(rows2), Long.toString(cols2), output("C"),
				(rbind? "rbind": "cbind")};
			fullRScriptName = RI_HOME + test_name + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				inputDir() + " " + expectedDir() + " " + (rbind? "rbind": "cbind");
	
			//initialize the frame data.
			ValueType[] lschemaA = genMixSchema(cols1);
			double[][] A = getRandomMatrix(rows1, cols1, min, max, sparsity, 1111);
			writeInputFrameWithMTD("A", A, true, lschemaA, FileFormat.BINARY);

			ValueType[] lschemaB = genMixSchema(cols2);
			double[][] B = getRandomMatrix(rows2, cols2, min, max, sparsity, 2345);
			writeInputFrameWithMTD("B", B, true, lschemaB, FileFormat.BINARY);

			boolean exceptionExpected = false;
			int expectedNumberOfJobs = -1;
			runTest(true, exceptionExpected, null, expectedNumberOfJobs);
			runRScript(true);

			ValueType[] lschemaOut = rbind ? lschemaA : UtilFunctions.copyOf(lschemaA, lschemaB);
			if(!Objects.equals(test_name, TEST_NAME) && !rbind)
				lschemaOut =  UtilFunctions.copyOf(lschemaOut, lschemaB);
			for(String file: config.getOutputFiles())
			{
				FrameBlock frameBlock = readDMLFrameFromHDFS(file, FileFormat.BINARY);
				FrameBlock frameRBlock = readRFrameFromHDFS(file + ".csv", FileFormat.CSV, frameBlock.getNumRows(),
					frameBlock.getNumColumns());
				verifyFrameData(frameBlock, frameRBlock, lschemaOut);
			}
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally
		{
			//reset execution platform
			rtplatform = prevPlfm;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			BinaryOp.FORCED_APPEND_METHOD = null;
		}
	}
	
	ValueType[] genMixSchema(int cols)
	{
		List<ValueType> schemaMixedLargeListStr = Collections.nCopies(cols/4, ValueType.STRING);
		List<ValueType> schemaMixedLargeListDble  = Collections.nCopies(cols/4, ValueType.FP64);
		List<ValueType> schemaMixedLargeListInt  = Collections.nCopies(cols/4, ValueType.INT64);
		List<ValueType> schemaMixedLargeListBool  = Collections.nCopies(cols-(cols/4)*3, ValueType.BOOLEAN);

		final List<ValueType> schemaMixedLargeList = new ArrayList<>(schemaMixedLargeListStr);
		schemaMixedLargeList.addAll(schemaMixedLargeListDble);
		schemaMixedLargeList.addAll(schemaMixedLargeListInt);
		schemaMixedLargeList.addAll(schemaMixedLargeListBool);
		ValueType[] schemaMixedLarge = new ValueType[schemaMixedLargeList.size()];
		schemaMixedLarge = schemaMixedLargeList.toArray(schemaMixedLarge);
		if( schemaMixedLarge.length != cols)
			throw new RuntimeException("Invalid schema length generated");
		return schemaMixedLarge;
	}
	
	private static void verifyFrameData(FrameBlock frame1, FrameBlock frame2, ValueType[] schema) {
		for ( int i=0; i<frame1.getNumRows(); i++ )
			for( int j=0; j<frame1.getNumColumns(); j++ )	{
				Object val1 = UtilFunctions.stringToObject(schema[j], UtilFunctions.objectToString(frame1.get(i, j)));
				Object val2 = UtilFunctions.stringToObject(schema[j], UtilFunctions.objectToString(frame2.get(i, j)));
				if( TestUtils.compareToR(schema[j], val1, val2, epsilon) != 0)
					Assert.fail("The DML data for cell ("+ i + "," + j + ") is " + val1 + 
							", not same as the R value " + val2);
			}
	}
}
