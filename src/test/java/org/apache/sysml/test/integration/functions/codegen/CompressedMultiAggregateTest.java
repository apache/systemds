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

package org.apache.sysml.test.integration.functions.codegen;

import java.io.File;
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.compress.CompressedMatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class CompressedMultiAggregateTest extends AutomatedTestBase 
{	
	private static final String TEST_NAME1 = "CompressedMultiAggregateMain";
	private static final String TEST_NAME2 = "CompressedMultiAggregateSide";
	private static final String TEST_DIR = "functions/codegen/";
	private static final String TEST_CLASS_DIR = TEST_DIR + CompressedMultiAggregateTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemML-config-codegen-compress.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	
	private static final int rows = 2023;
	private static final int cols = 20;
	private static final double sparsity1 = 0.9;
	private static final double sparsity2 = 0.1;
	private static final double sparsity3 = 0.0;
	private static final double eps = Math.pow(10, -6);
	
	public enum SparsityType {
		DENSE,
		SPARSE,
		EMPTY,
	}
	
	public enum ValueType {
		RAND, //UC
		CONST, //RLE
		RAND_ROUND_OLE, //OLE
		RAND_ROUND_DDC, //RLE
	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
	}
		
	@Test
	public void testCompressedMultiAggregateMainDenseConstCP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.DENSE, ValueType.CONST, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateMainDenseRandCP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.DENSE, ValueType.RAND, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateMainDenseRand2CP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.DENSE, ValueType.RAND_ROUND_DDC, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateMainDenseRand3CP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.DENSE, ValueType.RAND_ROUND_OLE, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateMainSparseConstCP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.SPARSE, ValueType.CONST, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateMainSparseRandCP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.SPARSE, ValueType.RAND, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateMainSparseRand2CP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.SPARSE, ValueType.RAND_ROUND_DDC, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateMainSparseRand3CP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.SPARSE, ValueType.RAND_ROUND_OLE, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateMainEmptyConstCP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.EMPTY, ValueType.CONST, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateMainEmptyRandCP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.EMPTY, ValueType.RAND, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateMainEmptyRand2CP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.EMPTY, ValueType.RAND_ROUND_DDC, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateMainEmptyRand3CP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.EMPTY, ValueType.RAND_ROUND_OLE, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateSideDenseConstCP() {
		testCompressedMultiAggregate( TEST_NAME2, SparsityType.DENSE, ValueType.CONST, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateSideDenseRandCP() {
		testCompressedMultiAggregate( TEST_NAME2, SparsityType.DENSE, ValueType.RAND, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateSideDenseRand2CP() {
		testCompressedMultiAggregate( TEST_NAME2, SparsityType.DENSE, ValueType.RAND_ROUND_DDC, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateSideDenseRand3CP() {
		testCompressedMultiAggregate( TEST_NAME2, SparsityType.DENSE, ValueType.RAND_ROUND_OLE, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateSideSparseConstCP() {
		testCompressedMultiAggregate( TEST_NAME2, SparsityType.SPARSE, ValueType.CONST, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateSideSparseRandCP() {
		testCompressedMultiAggregate( TEST_NAME2, SparsityType.SPARSE, ValueType.RAND, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateSideSparseRand2CP() {
		testCompressedMultiAggregate( TEST_NAME2, SparsityType.SPARSE, ValueType.RAND_ROUND_DDC, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateSideSparseRand3CP() {
		testCompressedMultiAggregate( TEST_NAME2, SparsityType.SPARSE, ValueType.RAND_ROUND_OLE, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateSideEmptyConstCP() {
		testCompressedMultiAggregate( TEST_NAME2, SparsityType.EMPTY, ValueType.CONST, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateSideEmptyRandCP() {
		testCompressedMultiAggregate( TEST_NAME2, SparsityType.EMPTY, ValueType.RAND, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateSideEmptyRand2CP() {
		testCompressedMultiAggregate( TEST_NAME2, SparsityType.EMPTY, ValueType.RAND_ROUND_DDC, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateSideEmptyRand3CP() {
		testCompressedMultiAggregate( TEST_NAME2, SparsityType.EMPTY, ValueType.RAND_ROUND_OLE, ExecType.CP );
	}
	
	@Test
	public void testCompressedMultiAggregateMainDenseConstSP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.DENSE, ValueType.CONST, ExecType.SPARK );
	}
	
	@Test
	public void testCompressedMultiAggregateMainDenseRandSP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.DENSE, ValueType.RAND, ExecType.SPARK );
	}
	
	@Test
	public void testCompressedMultiAggregateMainDenseRand2SP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.DENSE, ValueType.RAND_ROUND_DDC, ExecType.SPARK );
	}
	
	@Test
	public void testCompressedMultiAggregateMainDenseRand3SP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.DENSE, ValueType.RAND_ROUND_OLE, ExecType.SPARK );
	}
	
	@Test
	public void testCompressedMultiAggregateMainSparseConstSP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.SPARSE, ValueType.CONST, ExecType.SPARK );
	}
	
	@Test
	public void testCompressedMultiAggregateMainSparseRandSP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.SPARSE, ValueType.RAND, ExecType.SPARK );
	}
	
	@Test
	public void testCompressedMultiAggregateMainSparseRand2SP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.SPARSE, ValueType.RAND_ROUND_DDC, ExecType.SPARK );
	}
	
	@Test
	public void testCompressedMultiAggregateMainSparseRand3SP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.SPARSE, ValueType.RAND_ROUND_OLE, ExecType.SPARK );
	}
	
	@Test
	public void testCompressedMultiAggregateMainEmptyConstSP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.EMPTY, ValueType.CONST, ExecType.SPARK );
	}
	
	@Test
	public void testCompressedMultiAggregateMainEmptyRandSP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.EMPTY, ValueType.RAND, ExecType.SPARK );
	}
	
	@Test
	public void testCompressedMultiAggregateMainEmptyRand2SP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.EMPTY, ValueType.RAND_ROUND_DDC, ExecType.SPARK );
	}
	
	@Test
	public void testCompressedMultiAggregateMainEmptyRand3SP() {
		testCompressedMultiAggregate( TEST_NAME1, SparsityType.EMPTY, ValueType.RAND_ROUND_OLE, ExecType.SPARK );
	}
	
	//TODO compressed side inputs in spark
	
	
	private void testCompressedMultiAggregate(String testname, SparsityType stype, ValueType vtype, ExecType et)
	{	
		boolean oldRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = true;
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "-stats", 
					"-args", input("X"), output("R") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());			

			//generate input data
			double sparsity = -1;
			switch( stype ){
				case DENSE: sparsity = sparsity1; break;
				case SPARSE: sparsity = sparsity2; break;
				case EMPTY: sparsity = sparsity3; break;
			}
			
			//generate input data
			double min = (vtype==ValueType.CONST)? 10 : -10;
			double[][] X = TestUtils.generateTestMatrix(rows, cols, min, 10, sparsity, 7);
			if( vtype==ValueType.RAND_ROUND_OLE || vtype==ValueType.RAND_ROUND_DDC ) {
				CompressedMatrixBlock.ALLOW_DDC_ENCODING = (vtype==ValueType.RAND_ROUND_DDC);
				X = TestUtils.round(X);
			}
			writeInputMatrixWithMTD("X", X, true);
			
			//run tests
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");	
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			Assert.assertTrue(heavyHittersContainsSubString("spoofMA") 
				|| heavyHittersContainsSubString("sp_spoofMA"));
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldRewrites;
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
			CompressedMatrixBlock.ALLOW_DDC_ENCODING = true;
		}
	}	

	/**
	 * Override default configuration with custom test configuration to ensure
	 * scratch space and local temporary directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		// Instrumentation in this test's output log to show custom configuration file used for template.
		System.out.println("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
		return TEST_CONF_FILE;
	}
}
