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
import org.apache.sysml.api.mlcontext.MLContext;
import org.apache.sysml.api.mlcontext.MLContext.ExplainLevel;
import org.apache.sysml.api.mlcontext.MLResults;
import org.apache.sysml.api.mlcontext.Matrix;
import org.apache.sysml.api.mlcontext.MatrixMetadata;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;


public class MLContextOutputBlocksizeTest extends AutomatedTestBase
{
	protected final static String TEST_DIR = "org/apache/sysml/api/mlcontext";
	protected final static String TEST_NAME = "MLContext";

	private final static int rows = 100;
	private final static int cols = 63;
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
	public void testOutputBlocksizeTextcell() {
		runMLContextOutputBlocksizeTest("text");
	}

	@Test
	public void testOutputBlocksizeCSV() {
		runMLContextOutputBlocksizeTest("csv");
	}

	@Test
	public void testOutputBlocksizeMM() {
		runMLContextOutputBlocksizeTest("mm");
	}

	@Test
	public void testOutputBlocksizeBinary() {
		runMLContextOutputBlocksizeTest("binary");
	}


	private void runMLContextOutputBlocksizeTest(String format)
	{
		try
		{
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 76543);
			MatrixBlock mbA = DataConverter.convertToMatrixBlock(A);
			int blksz = ConfigurationManager.getBlocksize();
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, blksz, blksz, mbA.getNonZeros());

			//create input dataset
			JavaPairRDD<MatrixIndexes,MatrixBlock> in = SparkExecutionContext.toMatrixJavaPairRDD(sc, mbA, blksz, blksz);
			Matrix m = new Matrix(in, new MatrixMetadata(mc));

			ml.setExplain(true);
			ml.setExplainLevel(ExplainLevel.HOPS);

			//execute script
			String s ="if( sum(X) > 0 )"
					+ "   X = X/2;"
					+ "R = X;"
					+ "write(R, \"/tmp\", format=\""+format+"\");";
			Script script = dml(s).in("X", m).out("R");
			MLResults results = ml.execute(script);

			//compare output matrix characteristics
			MatrixCharacteristics mcOut = results.getMatrix("R")
				.getMatrixMetadata().asMatrixCharacteristics();
			Assert.assertEquals(blksz, mcOut.getRowsPerBlock());
			Assert.assertEquals(blksz, mcOut.getColsPerBlock());
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
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
