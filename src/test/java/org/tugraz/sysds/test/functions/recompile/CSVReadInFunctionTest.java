/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.test.functions.recompile;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.io.MatrixWriter;
import org.tugraz.sysds.runtime.io.MatrixWriterFactory;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.runtime.util.HDFSTool;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;
import org.tugraz.sysds.utils.Statistics;

public class CSVReadInFunctionTest extends AutomatedTestBase {

	private final static String TEST_NAME1 = "csv_read_function1";
	private final static String TEST_NAME2 = "csv_read_function2";
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_CLASS_DIR = TEST_DIR + CSVReadInFunctionTest.class.getSimpleName() + "/";

	private final static int rows = 123;
	private final static int cols = 45;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"X"}));
		addTestConfiguration(new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"X"}));
	}

	@Test
	public void testCSVReadNoFunctionNoMTD() {
		runCSVReadInFunctionTest(TEST_NAME1, false);
	}

	@Test
	public void testCSVReadFunctionNoMTD() {
		runCSVReadInFunctionTest(TEST_NAME2, false);
	}
	
	@Test
	public void testCSVReadNoFunctionMTD() {
		runCSVReadInFunctionTest(TEST_NAME1, true);
	}

	@Test
	public void testCSVReadFunctionMTD() {
		runCSVReadInFunctionTest(TEST_NAME2, true);
	}

	private void runCSVReadInFunctionTest(String testname, boolean withMtD) {
		try {
			getAndLoadTestConfiguration(testname);
			Statistics.reset();
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-stats", "-explain",
				"-args", input("A"), input("B"), output("R") };

			//write csv matrix without size information (no mtd file)
			double[][] A = getRandomMatrix(rows, cols, -1, 1, 1.0d, 7);
			MatrixBlock mbA = DataConverter.convertToMatrixBlock(A);
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(OutputInfo.CSVOutputInfo);
			writer.writeMatrixToHDFS(mbA, input("A"), rows, cols, -1, mbA.getNonZeros());
			
			double[][] B = getRandomMatrix(rows, 1, -1, 1, 1.0d, 7);
			MatrixBlock mbB = DataConverter.convertToMatrixBlock(B);
			MatrixWriter writer2 = MatrixWriterFactory.createMatrixWriter(OutputInfo.CSVOutputInfo);
			writer2.writeMatrixToHDFS(mbB, input("B"), rows, 1, -1, mbB.getNonZeros());
			
			if( withMtD ) {
				HDFSTool.writeMetaDataFile(input("A")+".mtd", ValueType.FP64,
					mbA.getDataCharacteristics(), OutputInfo.CSVOutputInfo);
				HDFSTool.writeMetaDataFile(input("B")+".mtd", ValueType.FP64,
					mbB.getDataCharacteristics(), OutputInfo.CSVOutputInfo);
			}
			
			runTest(true, false, null, -1);
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			Assert.assertEquals(dmlfile.get(new CellIndex(1,1)), new Double(mbA.sum()+mbB.sum()));
			
			//check no executed spark instructions
			Assert.assertEquals(Statistics.getNoOfExecutedSPInst(), 0);
			Assert.assertTrue(!Statistics.createdSparkContext());
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			try {
				HDFSTool.deleteFileIfExistOnHDFS(input("A"));
				HDFSTool.deleteFileIfExistOnHDFS(input("A")+".mtd");
				HDFSTool.deleteFileIfExistOnHDFS(input("B"));
				HDFSTool.deleteFileIfExistOnHDFS(input("B")+".mtd");
			} catch(Exception ex) {} //ignore
		}
	}
}
