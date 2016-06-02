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

package org.apache.sysml.test.integration.functions.transform;

import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.io.FrameReader;
import org.apache.sysml.runtime.io.FrameReaderFactory;
import org.apache.sysml.runtime.io.MatrixWriter;
import org.apache.sysml.runtime.io.MatrixWriterFactory;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.transform.meta.TfMetaUtils;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * 
 * 
 */
public class TransformReadMetaTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "TransformReadMeta";
	private static final String TEST_NAME2 = "TransformReadMeta2";
	private static final String TEST_DIR = "functions/transform/";
	private static final String TEST_CLASS_DIR = TEST_DIR + TransformReadMetaTest.class.getSimpleName() + "/";
	private static final String SPEC_X = "TransformReadMetaSpecX.json";
	
	private static final int rows = 1432;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"M1, M"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2,new String[]{"M1, M"}));
	}
	
	@Test
	public void runTestCsvCP() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", ",");
	}
	
	@Test
	public void runTestCsvHadoop() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.HADOOP, "csv", ",");
	}

	@Test
	public void runTestCsvSpark() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.SPARK, "csv", ",");
	}
	
	@Test
	public void runTestCsvTabCP() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", "\t");
	}
	
	@Test
	public void runTestCsvTabHadoop() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.HADOOP, "csv", "\t");
	}

	@Test
	public void runTestCsvTabSpark() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.SPARK, "csv", "\t");
	}
	
	@Test
	public void runTestCsvColonCP() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", ":");
	}
	
	@Test
	public void runTestCsvColonHadoop() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.HADOOP, "csv", ":");
	}

	@Test
	public void runTestCsvColonSpark() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.SPARK, "csv", ":");
	}
	
	
	@Test
	public void runTestTextCP() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.SINGLE_NODE, "text", ",");
	}
	
	@Test
	public void runTestTextHadoop() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.HADOOP, "text", ",");
	}

	@Test
	public void runTestTextSpark() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.SPARK, "text", ",");
	}

	@Test
	public void runTestBinaryCP() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.SINGLE_NODE, "binary", ",");
	}
	
	@Test
	public void runTestBinaryHadoop() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.HADOOP, "binary", ",");
	}

	@Test
	public void runTestBinarySpark() throws DMLRuntimeException, IOException {
		runTransformReadMetaTest(RUNTIME_PLATFORM.SPARK, "binary", ",");
	}

	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 * @throws IOException 
	 * @throws DMLRuntimeException 
	 */
	private void runTransformReadMetaTest( RUNTIME_PLATFORM rt, String ofmt, String delim) throws IOException, DMLRuntimeException
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = rt;
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK  || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			String testname = delim.equals(",") ? TEST_NAME1 : TEST_NAME2;
			
			getAndLoadTestConfiguration(testname);
			
			//generate input data
			double[][] X = DataConverter.convertToDoubleMatrix(
					MatrixBlock.seqOperations(0.5, rows/2, 0.5).appendOperations(
					MatrixBlock.seqOperations(0.5, rows/2, 0.5), new MatrixBlock()));
			MatrixBlock mbX = DataConverter.convertToMatrixBlock(X);
			CSVFileFormatProperties fprops = new CSVFileFormatProperties(false, delim, false);
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(OutputInfo.CSVOutputInfo, 1, fprops);
			writer.writeMatrixToHDFS(mbX, input("X"), rows, 2, -1, -1, -1);
			
			//read specs transform X and Y
			String specX = MapReduceTool.readStringFromHDFSFile(SCRIPT_DIR+TEST_DIR+SPEC_X);
			
			fullDMLScriptName = SCRIPT_DIR+TEST_DIR + testname + ".dml";
			programArgs = new String[]{"-args", input("X"), specX, output("M1"), output("M"), ofmt, delim};
			
			//run test
			runTest(true, false, null, -1); 
			
			//compare meta data frames
			InputInfo iinfo = InputInfo.stringExternalToInputInfo(ofmt, DataType.FRAME);
			FrameReader reader = FrameReaderFactory.createFrameReader(iinfo); 
			FrameBlock mExpected = TfMetaUtils.readTransformMetaDataFromFile(specX, output("M1"), delim);
			FrameBlock mRet = reader.readFrameFromHDFS(output("M"), rows, 2);
			for( int i=0; i<rows; i++ )
				for( int j=0; j<2; j++ ) {
					Assert.assertTrue("Wrong result: "+mRet.get(i, j)+".", 
						UtilFunctions.compareTo(ValueType.STRING, mExpected.get(i, j), mRet.get(i, j))==0);
				}
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}	
}
