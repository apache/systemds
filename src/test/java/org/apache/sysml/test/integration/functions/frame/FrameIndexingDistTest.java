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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.LeftIndexingOp;
import org.apache.sysml.hops.LeftIndexingOp.LeftIndexingMethod;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class FrameIndexingDistTest extends AutomatedTestBase
{
	
	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameIndexingDistTest.class.getSimpleName() + "/";
	private final static String TEST_NAME = "FrameIndexingDistTest";
	
	private final static double epsilon=0.0000000001;

	// Test data with 2 blocks of rows and columns
	private final static int rows = 1279, cols=1060;
	
	private final static int min=0;
	private final static int max=100;
	
	private final static List<ValueType> schemaMixedLargeListStr = Collections.nCopies(cols/4, ValueType.STRING);
	private final static List<ValueType> schemaMixedLargeListDble  = Collections.nCopies(cols/4, ValueType.DOUBLE);
	private final static List<ValueType> schemaMixedLargeListInt  = Collections.nCopies(cols/4, ValueType.INT);
	private final static List<ValueType> schemaMixedLargeListBool  = Collections.nCopies(cols/4, ValueType.BOOLEAN);
	private static ValueType[] schemaMixedLarge = null;
	static {
		final List<ValueType> schemaMixedLargeList = new ArrayList<ValueType>(schemaMixedLargeListStr);
		schemaMixedLargeList.addAll(schemaMixedLargeListDble);
		schemaMixedLargeList.addAll(schemaMixedLargeListInt);
		schemaMixedLargeList.addAll(schemaMixedLargeListBool);
		schemaMixedLarge = new ValueType[schemaMixedLargeList.size()];
		schemaMixedLarge = (ValueType[]) schemaMixedLargeList.toArray(schemaMixedLarge);
	}
	 
	@Override
	public void setUp() {
		addTestConfiguration("FrameIndexingDistTest", new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
				new String[] {"AB", "AC", "AD"}));
	}
	

	// Left Indexing Spark test cases
	@Test
	public void testMapLeftIndexingSP() throws DMLRuntimeException, IOException {
		runTestLeftIndexing(ExecType.SPARK, LeftIndexingMethod.SP_MLEFTINDEX, schemaMixedLarge);
	}
	
	@Test
	public void testGeneralLeftIndexingSP() throws DMLRuntimeException, IOException {
		runTestLeftIndexing(ExecType.SPARK, LeftIndexingMethod.SP_GLEFTINDEX, schemaMixedLarge);
	}
	
	private void runTestLeftIndexing(ExecType et, LeftIndexingOp.LeftIndexingMethod indexingMethod, ValueType[] schema) throws DMLRuntimeException, IOException {
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		RUNTIME_PLATFORM oldRTP = rtplatform;
		TestConfiguration config = getTestConfiguration("FrameIndexingDistTest");
		try
		{
			if(indexingMethod != null) {
				LeftIndexingOp.FORCED_LEFT_INDEXING = indexingMethod;
			}
			
			if(et == ExecType.SPARK) {
		    	rtplatform = RUNTIME_PLATFORM.SPARK;
		    }
			else {
				// rtplatform = (et==ExecType.MR)? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.SINGLE_NODE;
			    rtplatform = RUNTIME_PLATFORM.HYBRID;
			}
			if( rtplatform == RUNTIME_PLATFORM.SPARK )
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			
		    
	        config.addVariable("rows", rows);
	        config.addVariable("cols", cols);
	      
	        long rowstart=816, rowend=1229, colstart=109 /*967*/, colend=1009;

	        config.addVariable("rowstart", rowstart);
	        config.addVariable("rowend", rowend);
	        config.addVariable("colstart", colstart);
	        config.addVariable("colend", colend);
			loadTestConfiguration(config);
	        
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
			List<ValueType> lschema = Arrays.asList(schema);
	
			double sparsity=1.0;//rand.nextDouble(); 
	        double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, 1111 /*\\System.currentTimeMillis()*/);
	        writeInputFrameWithMTD("A", A, true, lschema, OutputInfo.BinaryBlockOutputInfo);	        
	        
	        sparsity=0.1;//rand.nextDouble();
	        double[][] B = getRandomMatrix((int)(rowend-rowstart+1), (int)(colend-colstart+1), min, max, sparsity, 2345 /*System.currentTimeMillis()*/);
	        List<ValueType> lschemaB = lschema.subList((int)colstart-1, (int)colend); 
	        writeInputFrameWithMTD("B", B, true, lschemaB, OutputInfo.BinaryBlockOutputInfo);	        

	        sparsity=0.5;//rand.nextDouble();
	        double[][] C = getRandomMatrix((int)(rowend), (int)(cols-colstart+1), min, max, sparsity, 3267 /*System.currentTimeMillis()*/);
	        List<ValueType> lschemaC = lschema.subList((int)colstart-1, (int)cols); 
	        writeInputFrameWithMTD("C", C, true, lschemaC, OutputInfo.BinaryBlockOutputInfo);	        

	        sparsity=0.01;//rand.nextDoublBe();
	        double[][] D = getRandomMatrix(rows, (int)(colend-colstart+1), min, max, sparsity, 4856 /*System.currentTimeMillis()*/);
	        writeInputFrameWithMTD("D", D, true, lschemaB, OutputInfo.BinaryBlockOutputInfo);	        
	
	        boolean exceptionExpected = false;
			int expectedNumberOfJobs = -1;
			runTest(true, exceptionExpected, null, expectedNumberOfJobs);
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
		
		runRScript(true);
	
		for(String file: config.getOutputFiles())
		{
			FrameBlock frameBlock = readDMLFrameFromHDFS(file, InputInfo.BinaryBlockInputInfo);
			MatrixCharacteristics md = new MatrixCharacteristics(frameBlock.getNumRows(), frameBlock.getNumColumns(), -1, -1);
			FrameBlock frameRBlock = readRFrameFromHDFS(file+".csv", InputInfo.CSVInputInfo, md);
			verifyFrameData(frameBlock, frameRBlock, schema);
			System.out.println("File processed is " + file);
		}
	}
	
	private void verifyFrameData(FrameBlock frame1, FrameBlock frame2, ValueType[] schema) {
		for ( int i=0; i<frame1.getNumRows(); ++i )
			for( int j=0; j<frame1.getNumColumns(); j++ )	{
				Object val1 = UtilFunctions.stringToObject(schema[j], UtilFunctions.objectToString(frame1.get(i, j)));
				Object val2 = UtilFunctions.stringToObject(schema[j], UtilFunctions.objectToString(frame2.get(i, j)));
				if( TestUtils.compareToR(schema[j], val1, val2, epsilon) != 0)
					Assert.fail("The DML data for cell ("+ i + "," + j + ") is " + val1 + 
							", not same as the R value " + val2);
			}
	}

}

