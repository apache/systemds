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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.LeftIndexingOp;
import org.apache.sysds.hops.LeftIndexingOp.LeftIndexingMethod;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class FrameIndexingDistTest extends AutomatedTestBase {
	// private static final Log LOG = LogFactory.getLog(FrameIndexingDistTest.class.getName());
	
	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameIndexingDistTest.class.getSimpleName() + "/";
	private final static String TEST_NAME = "FrameLeftIndexing";
	private final static String RTEST_NAME = "FrameRightIndexing";
	
	private enum IXType {
		RIX,
		LIX,
	}
	

	private final static double epsilon=0.0000000001;

	// Test data with 2 blocks of rows and columns
	private final static int rows = 1279, cols=1060;
	
	private final static int min=0;
	private final static int max=100;
	
	private final static double sparsity1 = 1.0;
	private final static double sparsity2 = 0.5;
	private final static double sparsity3 = 0.1;
	private final static double sparsity4 = 0.01;

	
	private final static List<ValueType> schemaMixedLargeListStr = Collections.nCopies(cols/4, ValueType.STRING);
	private final static List<ValueType> schemaMixedLargeListDble  = Collections.nCopies(cols/4, ValueType.FP64);
	private final static List<ValueType> schemaMixedLargeListInt  = Collections.nCopies(cols/4, ValueType.INT64);
	private final static List<ValueType> schemaMixedLargeListBool  = Collections.nCopies(cols/4, ValueType.BITSET);
	private static ValueType[] schemaMixedLarge = null;
	static {
		final List<ValueType> schemaMixedLargeList = new ArrayList<>(schemaMixedLargeListStr);
		schemaMixedLargeList.addAll(schemaMixedLargeListDble);
		schemaMixedLargeList.addAll(schemaMixedLargeListInt);
		schemaMixedLargeList.addAll(schemaMixedLargeListBool);
		schemaMixedLarge = new ValueType[schemaMixedLargeList.size()];
		schemaMixedLarge = schemaMixedLargeList.toArray(schemaMixedLarge);
	}
	 
	@Override
	public void setUp() {
		addTestConfiguration("FrameLeftIndexing", new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
				new String[] {"AB", "AC", "AD"}));
		addTestConfiguration("FrameRightIndexing", new TestConfiguration(TEST_CLASS_DIR, RTEST_NAME, 
				new String[] {"B", "C", "D"}));
	}
	

	// Left Indexing Spark test cases
	@Test
	public void testMapLeftIndexingSP() throws IOException {
		runTestLeftIndexing(ExecType.SPARK, LeftIndexingMethod.SP_MLEFTINDEX_R, schemaMixedLarge, IXType.LIX, true);
	}
	
	// @Test
	// public void testGeneralLeftIndexingSP() throws IOException {
	// 	runTestLeftIndexing(ExecType.SPARK, LeftIndexingMethod.SP_GLEFTINDEX, schemaMixedLarge, IXType.LIX, true);
	// }
	
	
	// // Right Indexing Spark test cases
	// @Test
	// public void testRightIndexingSPSparse() throws IOException {
	// 	runTestLeftIndexing(ExecType.SPARK, null, schemaMixedLarge, IXType.RIX, true);
	// }
	
	// @Test
	// public void testRightIndexingSPDense() throws IOException {
	// 	runTestLeftIndexing(ExecType.SPARK, null, schemaMixedLarge, IXType.RIX, false);
	// }
	

	
	private void runTestLeftIndexing(ExecType et, LeftIndexingOp.LeftIndexingMethod indexingMethod, ValueType[] schema, IXType itype, boolean bSparse) throws IOException {
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		ExecMode oldRTP = rtplatform;
		TestConfiguration config = null;
		
		HashMap<String, ValueType[]> outputSchema = new HashMap<>();
		
		if (itype == IXType.LIX) 
			config = getTestConfiguration("FrameLeftIndexing");
		else
			config = getTestConfiguration("FrameRightIndexing");
		setOutputBuffering(true);
		try
		{
			if(indexingMethod != null) {
				LeftIndexingOp.FORCED_LEFT_INDEXING = indexingMethod;
			}
			
			if(et == ExecType.SPARK) {
				rtplatform = ExecMode.SPARK;
			}
			else {
				// rtplatform = (et==ExecType.MR)? ExecMode.HADOOP : ExecMode.SINGLE_NODE;
				rtplatform = ExecMode.HYBRID;
			}
			if( rtplatform == ExecMode.SPARK )
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			
			
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
		  
			long rowstart=816, rowend=1229, colstart=109 /*967*/, colend=1009;

			config.addVariable("rowstart", rowstart);
			config.addVariable("rowend", rowend);
			config.addVariable("colstart", colstart);
			config.addVariable("colend", colend);
			loadTestConfiguration(config);
			
			if (itype == IXType.LIX) {
				/* This is for running the junit test the new way, i.e., construct the arguments directly */
				String LI_HOME = SCRIPT_DIR + TEST_DIR;
				fullDMLScriptName = LI_HOME + TEST_NAME + ".dml";
				programArgs = new String[]{"-args",  input("A"),
					Long.toString(rows), Long.toString(cols),
					Long.toString(rowstart), Long.toString(rowend),
					Long.toString(colstart), Long.toString(colend),
					output("AB"), output("AC"), output("AD"),
					input("B"), input("C"), input("D"),
					Long.toString(rowend-rowstart+1), 
					Long.toString(colend-colstart+1),
					Long.toString(cols-colstart+1)};
				
				fullRScriptName = LI_HOME + TEST_NAME + ".R";
				rCmd = "Rscript" + " " + fullRScriptName + " " + 
					inputDir() + " " + rowstart + " " + rowend + " " + colstart + " " + colend + " " + expectedDir();
				
				//initialize the frame data.

				double sparsity=sparsity1;//rand.nextDouble(); 
				double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, 1111 /*\\System.currentTimeMillis()*/);
				writeInputFrameWithMTD("A", A, true, schema, FileFormat.BINARY);
				
				sparsity=sparsity3;//rand.nextDouble();
				double[][] B = getRandomMatrix((int)(rowend-rowstart+1), (int)(colend-colstart+1), min, max, sparsity, 2345 /*System.currentTimeMillis()*/);
				ValueType[] lschemaB = Arrays.copyOfRange(schema, (int)colstart-1, (int)colend);
				writeInputFrameWithMTD("B", B, true, lschemaB, FileFormat.BINARY);
	
				sparsity=sparsity2;//rand.nextDouble();
				double[][] C = getRandomMatrix((int)(rowend), (int)(cols-colstart+1), min, max, sparsity, 3267 /*System.currentTimeMillis()*/);
				ValueType[] lschemaC = Arrays.copyOfRange(schema, (int)colstart-1, cols);
				writeInputFrameWithMTD("C", C, true, lschemaC, FileFormat.BINARY);
	
				sparsity=sparsity4;//rand.nextDoublBe();
				double[][] D = getRandomMatrix(rows, (int)(colend-colstart+1), min, max, sparsity, 4856 /*System.currentTimeMillis()*/);
				writeInputFrameWithMTD("D", D, true, lschemaB, FileFormat.BINARY);
		
				boolean exceptionExpected = false;
				int expectedNumberOfJobs = -1;
				runTest(true, exceptionExpected, null, expectedNumberOfJobs);
				
				for(String file: config.getOutputFiles())
					outputSchema.put(file, schema);
			}
			else {
				/* This is for running the junit test the new way, i.e., construct the arguments directly */
				String RI_HOME = SCRIPT_DIR + TEST_DIR;
				fullDMLScriptName = RI_HOME + RTEST_NAME + ".dml";
				programArgs = new String[]{"-stats", "-explain","-args",  input("A"), 
					Long.toString(rows), Long.toString(cols),
					Long.toString(rowstart), Long.toString(rowend),
					Long.toString(colstart), Long.toString(colend),
					output("B"), output("C"), output("D")}; 
				
				fullRScriptName = RI_HOME + RTEST_NAME + ".R";
				rCmd = "Rscript" + " " + fullRScriptName + " " + 
					inputDir() + " " + rowstart + " " + rowend + " " + colstart + " " + colend + " " + expectedDir();
		
				//initialize the frame data.
		
				double sparsity = bSparse ? sparsity4 : sparsity2;
				double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, 1111 /*\\System.currentTimeMillis()*/);
				writeInputFrameWithMTD("A", A, true, schema, FileFormat.BINARY);
				
				ValueType[] schemaB = new ValueType[(int) (colend-colstart+1)]; 
				System.arraycopy(schema, (int)(colstart-1), schemaB, 0, (int)(colend-colstart+1)); 
				outputSchema.put(config.getOutputFiles()[0], schemaB);

				ValueType[] schemaC = new ValueType[(int) (cols-colstart+1)]; 
				System.arraycopy(schema, (int)(colstart-1), schemaC, 0, (int)(cols-colstart+1)); 
				outputSchema.put(config.getOutputFiles()[1], schemaC);

				outputSchema.put(config.getOutputFiles()[2], schemaB);
				
				boolean exceptionExpected = false;
				int expectedNumberOfJobs = -1;
				runTest(true, exceptionExpected, null, expectedNumberOfJobs);
			}

			runRScript(true);
		
			for(String file: config.getOutputFiles())
			{
				FrameBlock frameBlock = readDMLFrameFromHDFS(file, FileFormat.BINARY);
				FrameBlock frameRBlock = readRFrameFromHDFS(file + ".csv", FileFormat.CSV, frameBlock.getNumRows(),
					frameBlock.getNumColumns());
				ValueType[] schemaOut = outputSchema.get(file);
				verifyFrameData(frameBlock, frameRBlock, schemaOut);
			}
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally
		{
			rtplatform = oldRTP;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			LeftIndexingOp.FORCED_LEFT_INDEXING = null;
		}
		
	}
	
	private static void verifyFrameData(FrameBlock frame1, FrameBlock frame2, ValueType[] schema) {
		for ( int i=0; i<frame1.getNumRows(); i++ )
			for( int j=0; j<frame1.getNumColumns(); j++ )	{
				Object val1 = UtilFunctions.stringToObject(schema[j], UtilFunctions.objectToString(frame1.get(i, j)));
				Object val2 = UtilFunctions.stringToObject(schema[j], UtilFunctions.objectToString(frame2.get(i, j)));
				
				if( TestUtils.compareToR(schema[j], val1, val2, epsilon) != 0)
					Assert.fail("The DML data for cell ("+ i + "," + j + ") is " + val1 + 
							", not same as the R value " + val2 + "  with valueType : " + schema[j]);
			}
	}

}

