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

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.io.MatrixReaderFactory;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

/**
 * 
 */
public class FrameMatrixCastingTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_NAME1 = "Frame2MatrixCast";
	private final static String TEST_NAME2 = "Matrix2FrameCast";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameMatrixCastingTest.class.getSimpleName() + "/";

	private final static int rows = 2593;
	private final static int cols1 = 372;
	private final static int cols2 = 1102;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"B"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"B"}));		
	}
	
	@Test
	public void testStringFrame2MatrixCastSingleCP() {
		runFrameCastingTest(TEST_NAME1, false, ValueType.STRING, ExecType.CP);
	}
	
	@Test
	public void testStringFrame2MatrixCastMultiCP() {
		runFrameCastingTest(TEST_NAME1, true, ValueType.STRING, ExecType.CP);
	}
	
	@Test
	public void testDoubleFrame2MatrixCastSingleCP() {
		runFrameCastingTest(TEST_NAME1, false, ValueType.FP64, ExecType.CP);
	}
	
	@Test
	public void testDoubleFrame2MatrixCastMultiCP() {
		runFrameCastingTest(TEST_NAME1, true, ValueType.FP64, ExecType.CP);
	}

	@Test
	public void testMatrix2FrameCastSingleCP() {
		runFrameCastingTest(TEST_NAME2, false, null, ExecType.CP);
	}
	
	@Test
	public void testMatrix2FrameCastMultiCP() {
		runFrameCastingTest(TEST_NAME2, true, null, ExecType.CP);
	}
	
	@Test
	public void testStringFrame2MatrixCastSingleSpark() {
		runFrameCastingTest(TEST_NAME1, false, ValueType.STRING, ExecType.SPARK);
	}
	
	@Test
	public void testStringFrame2MatrixCastMultiSpark() {
		runFrameCastingTest(TEST_NAME1, true, ValueType.STRING, ExecType.SPARK);
	}
	
	@Test
	public void testDoubleFrame2MatrixCastSingleSpark() {
		runFrameCastingTest(TEST_NAME1, false, ValueType.FP64, ExecType.SPARK);
	}
	
	@Test
	public void testDoubleFrame2MatrixCastMultiSpark() {
		runFrameCastingTest(TEST_NAME1, true, ValueType.FP64, ExecType.SPARK);
	}

	@Test
	public void testMatrix2FrameCastSingleSpark() {
		runFrameCastingTest(TEST_NAME2, false, null, ExecType.SPARK);
	}
	
	@Test
	public void testMatrix2FrameCastMultiSpark() {
		runFrameCastingTest(TEST_NAME2, true, null, ExecType.SPARK);
	}
	
	private void runFrameCastingTest( String testname, boolean multColBlks, ValueType vt, ExecType et)
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
		setOutputBuffering(true);
		try
		{
			int cols = multColBlks ? cols2 : cols1;
			
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"), output("B") };
			
			//data generation
			double[][] A = getRandomMatrix(rows, cols, -1, 1, 0.9, 7); 
			DataType dtin = testname.equals(TEST_NAME1) ? DataType.FRAME : DataType.MATRIX;
			ValueType vtin = testname.equals(TEST_NAME1) ? vt : ValueType.FP64;
			writeMatrixOrFrameInput(input("A"), A, rows, cols, dtin, vtin);
			
			//run testcase
			runTest(true, false, null, -1);
			
			//compare matrices
			DataType dtout = testname.equals(TEST_NAME1) ? DataType.MATRIX : DataType.FRAME;
			double[][] B = readMatrixOrFrameInput(output("B"), rows, cols, dtout);
			TestUtils.compareMatrices(A, B, rows, cols, 0);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
	private static void writeMatrixOrFrameInput(String fname, double[][] A, int rows, int cols, DataType dt, ValueType vt) 
		throws IOException 
	{
		int blksize = ConfigurationManager.getBlocksize();
		
		//write input data
		if( dt == DataType.FRAME ) {
			FrameBlock fb = DataConverter.convertToFrameBlock(DataConverter.convertToMatrixBlock(A), vt);
			FrameWriter writer = FrameWriterFactory.createFrameWriter(FileFormat.BINARY);
			writer.writeFrameToHDFS(fb, fname, rows, cols);
		}
		else {
			MatrixBlock mb = DataConverter.convertToMatrixBlock(A);
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(FileFormat.BINARY);
			writer.writeMatrixToHDFS(mb, fname, rows, cols, blksize, -1);
		}
		
		//write meta data
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, blksize, blksize);
		HDFSTool.writeMetaDataFile(fname+".mtd", vt, null, dt, mc, FileFormat.BINARY);
	
	}
	
	private static double[][] readMatrixOrFrameInput(String fname, int rows, int cols, DataType dt) 
		throws IOException 
	{
		MatrixBlock ret = null;
		
		//read input data
		if( dt == DataType.FRAME ) {
			FrameReader reader = FrameReaderFactory.createFrameReader(FileFormat.BINARY);
			FrameBlock fb = reader.readFrameFromHDFS(fname, rows, cols);
			ret = DataConverter.convertToMatrixBlock(fb);
		}
		else {
			int blksize = ConfigurationManager.getBlocksize();
			MatrixReader reader = MatrixReaderFactory.createMatrixReader(FileFormat.BINARY);
			ret = reader.readMatrixFromHDFS(fname, rows, cols, blksize, -1);
		}
		
		return DataConverter.convertToDoubleMatrix(ret);
	}
}
