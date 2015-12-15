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

package org.apache.sysml.test.integration.functions.mlcontext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.api.MLContext;
import org.apache.sysml.api.MLOutput;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt;

import org.apache.spark.mllib.linalg.distributed.MatrixEntry;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;

@RunWith(value = Parameterized.class)
public class GNMFTest extends AutomatedTestBase 
{
	private final static String TEST_DIR = "applications/gnmf/";
	private final static String TEST_NAME = "GNMF";
	private final static String TEST_CLASS_DIR = TEST_DIR + GNMFTest.class.getSimpleName() + "/";
	
	int numRegisteredInputs;
	int numRegisteredOutputs;
	
	public GNMFTest(int in, int out) {
		numRegisteredInputs = in;
		numRegisteredOutputs = out;
	}
	
	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { { 0, 0 }, { 3, 2 }, { 2, 2 }, { 2, 1 }, { 2, 0 }, { 3, 0 }};
	   return Arrays.asList(data);
	 }
	 
	@Override
	public void setUp() {
		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
	}
	
	@Test
	public void testGNMFWithRDMLAndJava() throws IOException, DMLException, ParseException {
		System.out.println("------------ BEGIN " + TEST_NAME + " TEST {" + numRegisteredInputs + ", "
				+ numRegisteredOutputs + "} ------------");
		this.scriptType = ScriptType.DML;
		
		int m = 2000;
		int n = 1500;
		int k = 50;
		int maxiter = 2;
		double Eps = Math.pow(10, -8);
		
		getAndLoadTestConfiguration(TEST_NAME);

		List<String> proArgs = new ArrayList<String>();
		proArgs.add(input("v"));
		proArgs.add(input("w"));
		proArgs.add(input("h"));
		proArgs.add(Integer.toString(maxiter));
		proArgs.add(output("w"));
		proArgs.add(output("h"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		fullDMLScriptName = getScript();
		
		rCmd = getRCmd(inputDir(), Integer.toString(maxiter), expectedDir());

		double[][] v = getRandomMatrix(m, n, 1, 5, 0.2, System.currentTimeMillis());
		double[][] w = getRandomMatrix(m, k, 0, 1, 1, System.currentTimeMillis());
		double[][] h = getRandomMatrix(k, n, 0, 1, 1, System.currentTimeMillis());

		writeInputMatrixWithMTD("v", v, true);
		writeInputMatrixWithMTD("w", w, true);
		writeInputMatrixWithMTD("h", h, true);

		for (int i = 0; i < maxiter; i++) {
			double[][] tW = TestUtils.performTranspose(w);
			double[][] tWV = TestUtils.performMatrixMultiplication(tW, v);
			double[][] tWW = TestUtils.performMatrixMultiplication(tW, w);
			double[][] tWWH = TestUtils.performMatrixMultiplication(tWW, h);
			for (int j = 0; j < k; j++) {
				for (int l = 0; l < n; l++) {
					h[j][l] = h[j][l] * (tWV[j][l] / (tWWH[j][l] + Eps));
				}
			}

			double[][] tH = TestUtils.performTranspose(h);
			double[][] vTH = TestUtils.performMatrixMultiplication(v, tH);
			double[][] hTH = TestUtils.performMatrixMultiplication(h, tH);
			double[][] wHTH = TestUtils.performMatrixMultiplication(w, hTH);
			for (int j = 0; j < m; j++) {
				for (int l = 0; l < k; l++) {
					w[j][l] = w[j][l] * (vTH[j][l] / (wHTH[j][l] + Eps));
				}
			}
		}
		
		boolean oldConfig = DMLScript.USE_LOCAL_SPARK_CONFIG; 
		DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		RUNTIME_PLATFORM oldRT = DMLScript.rtplatform;
		
		try 
		{
			DMLScript.rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK;
		
			MLContext mlCtx = getMLContextForTesting();
			SparkContext sc = mlCtx.getSparkContext();
			mlCtx.reset();
			
			// Read two matrices through RDD and one through HDFS
			if(numRegisteredInputs >= 1) {
				JavaRDD<String> vIn = sc.textFile(input("v"), 2).toJavaRDD();
				mlCtx.registerInput("V", vIn, "text", m, n);
			}
			
			if(numRegisteredInputs >= 2) {
				JavaRDD<String> wIn = sc.textFile(input("w"), 2).toJavaRDD();
				mlCtx.registerInput("W", wIn, "text", m, k);
			}
			
			if(numRegisteredInputs >= 3) {
				JavaRDD<String> hIn = sc.textFile(input("h"), 2).toJavaRDD();
				mlCtx.registerInput("H", hIn, "text", k, n);
			}
			
			// Output one matrix to HDFS and get one as RDD
			if(numRegisteredOutputs >= 1) {
				mlCtx.registerOutput("H");
			}
			
			if(numRegisteredOutputs >= 2) {
				mlCtx.registerOutput("W");
			}
			
			MLOutput out = mlCtx.execute(fullDMLScriptName, programArgs);
			
			if(numRegisteredOutputs >= 1) {
				JavaRDD<String> hOut = out.getStringRDD("H", "text");
				String fName = output("h");
				try {
					MapReduceTool.deleteFileIfExistOnHDFS( fName );
				} catch (IOException e) {
					throw new DMLRuntimeException("Error: While deleting file on HDFS");
				}
				hOut.saveAsTextFile(fName);
			}
			
			if(numRegisteredOutputs >= 2) {
//				Test converter: Text -> CoordinateMatrix -> BinaryBlock -> Text
//				JavaRDD<String> wOut = out.getStringRDD("W", "text");
				JavaRDD<MatrixEntry> matRDD = out.getStringRDD("W", "text").map(new StringToMatrixEntry());
				MatrixCharacteristics mcW = out.getMatrixCharacteristics("W");
				CoordinateMatrix coordinateMatrix = new CoordinateMatrix(matRDD.rdd(), mcW.getRows(), mcW.getCols());
				JavaPairRDD<MatrixIndexes, MatrixBlock> binaryRDD = RDDConverterUtilsExt.coordinateMatrixToBinaryBlock(sc, coordinateMatrix, mcW, true);
				JavaRDD<String> wOut = RDDConverterUtilsExt.binaryBlockToStringRDD(binaryRDD, mcW, "text");
				
				String fName = output("w");
				try {
					MapReduceTool.deleteFileIfExistOnHDFS( fName );
				} catch (IOException e) {
					throw new DMLRuntimeException("Error: While deleting file on HDFS");
				}
				wOut.saveAsTextFile(fName);
			}
			
			runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> hmWDML = readDMLMatrixFromHDFS("w");
			HashMap<CellIndex, Double> hmHDML = readDMLMatrixFromHDFS("h");
			HashMap<CellIndex, Double> hmWR = readRMatrixFromFS("w");
			HashMap<CellIndex, Double> hmHR = readRMatrixFromFS("h");
			TestUtils.compareMatrices(hmWDML, hmWR, 0.000001, "hmWDML", "hmWR");
			TestUtils.compareMatrices(hmHDML, hmHR, 0.000001, "hmHDML", "hmHR");
			
			//cleanup mlcontext (prevent test memory leaks)
			mlCtx.reset();
		}
		finally {
			DMLScript.rtplatform = oldRT;
			DMLScript.USE_LOCAL_SPARK_CONFIG = oldConfig;
		}
	}
	
	public static class StringToMatrixEntry implements Function<String, MatrixEntry> {

		private static final long serialVersionUID = 7456391906436606324L;

		@Override
		public MatrixEntry call(String str) throws Exception {
			String [] elem = str.split(" ");
			if(elem.length != 3) {
				throw new Exception("Expected text entry but got: " + str);
			}
			
			long row = Long.parseLong(elem[0]);
			long col = Long.parseLong(elem[1]);
			double value = Double.parseDouble(elem[2]);
			return new MatrixEntry(row, col, value);
		}
		
	}
}