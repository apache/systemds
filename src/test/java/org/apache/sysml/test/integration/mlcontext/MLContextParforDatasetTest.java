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

package org.apache.sysml.test.integration.mlcontext;

import static org.apache.sysml.api.mlcontext.ScriptFactory.dml;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.sysml.api.mlcontext.MLContext;
import org.apache.sysml.api.mlcontext.MLResults;
import org.apache.sysml.api.mlcontext.MatrixFormat;
import org.apache.sysml.api.mlcontext.MatrixMetadata;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.api.mlcontext.MLContext.ExplainLevel;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;


public class MLContextParforDatasetTest extends AutomatedTestBase 
{
	protected final static String TEST_DIR = "org/apache/sysml/api/mlcontext";
	protected final static String TEST_NAME = "MLContext";

	private final static int rows = 100;
	private final static int cols = 1600;
	private final static double sparsity = 0.7;
	
	private static SparkConf conf;
	private static JavaSparkContext sc;
	private static MLContext ml;

	@BeforeClass
	public static void setUpClass() {
		if (conf == null)
			conf = SparkExecutionContext.createSystemMLSparkConf()
				.setAppName("MLContextTest").setMaster("local");
		if (sc == null)
			sc = new JavaSparkContext(conf);
		ml = new MLContext(sc);
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}


	@Test
	public void testParforDatasetVector() {
		runMLContextParforDatasetTest(true, false, false);
	}
	
	@Test
	public void testParforDatasetRow() {
		runMLContextParforDatasetTest(false, false, false);
	}
	
	@Test
	public void testParforDatasetVectorUnkownDims() {
		runMLContextParforDatasetTest(true, true, false);
	}
	
	@Test
	public void testParforDatasetRowUnknownDims() {
		runMLContextParforDatasetTest(false, true, false);
	}
	
	@Test
	public void testParforDatasetVectorMulti() {
		runMLContextParforDatasetTest(true, false, true);
	}
	
	@Test
	public void testParforDatasetRowMulti() {
		runMLContextParforDatasetTest(false, false, true);
	}
	
	@Test
	public void testParforDatasetVectorUnkownDimsMulti() {
		runMLContextParforDatasetTest(true, true, true);
	}
	
	@Test
	public void testParforDatasetRowUnknownDimsMulti() {
		runMLContextParforDatasetTest(false, true, true);
	}
	
	private void runMLContextParforDatasetTest(boolean vector, boolean unknownDims, boolean multiInputs) 
	{
		//modify memory budget to trigger fused datapartition-execute
		long oldmem = InfrastructureAnalyzer.getLocalMaxMemory();
		InfrastructureAnalyzer.setLocalMaxMemory(1*1024*1024); //1MB
		
		try
		{
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 76543); 
			MatrixBlock mbA = DataConverter.convertToMatrixBlock(A); 
			int blksz = ConfigurationManager.getBlocksize();
			MatrixCharacteristics mc1 = new MatrixCharacteristics(rows, cols, blksz, blksz, mbA.getNonZeros());
			MatrixCharacteristics mc2 = unknownDims ? new MatrixCharacteristics() : new MatrixCharacteristics(mc1);
			
			//create input dataset
			SparkSession sparkSession = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();
			JavaPairRDD<MatrixIndexes,MatrixBlock> in = SparkExecutionContext.toMatrixJavaPairRDD(sc, mbA, blksz, blksz);
			Dataset<Row> df = RDDConverterUtils.binaryBlockToDataFrame(sparkSession, in, mc1, vector);
			MatrixMetadata mm = new MatrixMetadata(vector ? MatrixFormat.DF_VECTOR_WITH_INDEX : MatrixFormat.DF_DOUBLES_WITH_INDEX);
			mm.setMatrixCharacteristics(mc2);
			
			String s1 = "v = matrix(0, rows=nrow(X), cols=1)"
					+ "parfor(i in 1:nrow(X), log=DEBUG) {"
					+ "   v[i, ] = sum(X[i, ]);"
					+ "}"
					+ "r = sum(v);";
			String s2 = "v = matrix(0, rows=nrow(X), cols=1)"
					+"Y = X;"
					+ "parfor(i in 1:nrow(X), log=DEBUG) {"
					+ "   v[i, ] = sum(X[i, ]+Y[i, ]);"
					+ "}"
					+ "r = sum(v);";
			String s = multiInputs ? s2 : s1;
			
			ml.setExplain(true);
			ml.setExplainLevel(ExplainLevel.RUNTIME);
			ml.setStatistics(true);
			
			Script script = dml(s).in("X", df, mm).out("r");
			MLResults results = ml.execute(script);
			
			//compare aggregation results
			double sum1 = results.getDouble("r");
			double sum2 = mbA.sum() * (multiInputs ? 2 : 1);
			
			TestUtils.compareScalars(sum2, sum1, 0.000001);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally {
			InfrastructureAnalyzer.setLocalMaxMemory(oldmem);	
		}
	}

	@After
	public void tearDown() {
		super.tearDown();
	}

	@AfterClass
	public static void tearDownClass() {
		// stop spark context to allow single jvm tests (otherwise the
		// next test that tries to create a SparkContext would fail)
		sc.stop();
		sc = null;
		conf = null;

		// clear status mlcontext and spark exec context
		ml.close();
		ml = null;
	}
}
