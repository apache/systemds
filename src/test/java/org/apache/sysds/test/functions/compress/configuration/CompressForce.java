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

package org.apache.sysds.test.functions.compress.configuration;

import static org.junit.Assert.fail;

import java.io.File;

import org.apache.sysds.common.Types.ExecType;
import org.junit.Ignore;
import org.junit.Test;

public class CompressForce extends CompressBase {

	public String TEST_NAME = "compress";
	public String TEST_DIR = "functions/compress/force/";
	public String TEST_CLASS_DIR = TEST_DIR + CompressForce.class.getSimpleName() + "/";
	private String TEST_CONF = "SystemDS-config-compress.xml";
	private File TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);

	protected String getTestClassDir() {
		return TEST_CLASS_DIR;
	}

	protected String getTestName() {
		return TEST_NAME;
	}

	protected String getTestDir() {
		return TEST_DIR;
	}

	@Test
	public void testTranspose_CP() {
		runTest(1500, 20, 2, 1, ExecType.CP, "transpose");
	}

	@Test
	public void testTranspose_SP() {
		runTest(1500, 1, 2, 1, ExecType.SPARK, "transpose");
	}

	@Test
	public void testSum_CP() {
		runTest(1500, 20, 0, 1, ExecType.CP, "sum");
	}

	@Test
	public void testSum_SP() {
		runTest(1500, 1, 0, 1, ExecType.SPARK, "sum");
	}

	@Test
	public void testRowAggregate_CP() {
		runTest(1500, 20, 0, 1, ExecType.CP, "row_min");
	}

	@Test
	public void testRowAggregate_SP() {
		runTest(1500, 1, 0, 1, ExecType.SPARK, "row_min");
	}

	@Test
	public void testSequence_CP() {
		runTest(1500, 1, -1, 1, ExecType.CP, "plus_mm_ewbm_sum");
	}

	@Test
	public void testSequence_SP() {
		runTest(1500, 1, -1, 1, ExecType.SPARK, "plus_mm_ewbm_sum");
	}

	@Test
	public void testPlus_CP() {
		runTest(1500, 1, 0, 1, ExecType.CP, "plus");
	}

	@Test
	public void testPlus_MM_SP() {
		runTest(1500, 1, 0, 1, ExecType.SPARK, "plus_mm");
	}

	@Test
	public void test_ElementWiseBinaryMultiplyOp_right_CP() {
		runTest(1500, 1, -1, 1, ExecType.CP, "ewbm_right");
	}

	@Test
	public void test_ElementWiseBinaryMultiplyOp_right_SP() {
		runTest(1500, 1, -1, 1, ExecType.SPARK, "ewbm_right");
	}

	@Test
	public void test_ElementWiseBinaryMultiplyOp_left_CP() {
		runTest(1500, 1, -1, 1, ExecType.CP, "ewbm_left");
	}

	@Test
	public void test_ElementWiseBinaryMultiplyOp_left_SP() {
		runTest(1500, 1, 2, 1, ExecType.SPARK, "ewbm_left");
	}

	@Test
	public void test_ElementWiseBinaryMultiplyOp_left_SP_larger() {
		runTest(1500, 15, -1, 1, ExecType.SPARK, "ewbm_left");
	}

	@Test
	public void test_ElementWiseBinaryPlusOp_CP() {
		runTest(1500, 1, 0, 1, ExecType.CP, "ewbp");
	}

	@Test
	public void test_ElementWiseBinaryPlusOp_SP() {
		runTest(1500, 1, 0, 1, ExecType.SPARK, "ewbp");
	}

	@Test
	public void testPlus_MM_CP() {
		runTest(1500, 1, 0, 1, ExecType.CP, "plus_mm");
	}

	@Test
	public void testPlus_SP() {
		runTest(1500, 1, 0, 1, ExecType.SPARK, "plus");
	}

	@Test
	public void testMatrixMultRightSum_Smaller_CP() {
		runTest(1500, 1, 0, 1, ExecType.CP, "mmr_sum");
	}

	@Test
	public void testMatrixMultRightSum_Smaller_SP() {
		runTest(1500, 1, 0, 1, ExecType.SPARK, "mmr_sum");
	}

	@Test
	public void testMatrixMultRightSum_Larger_CP() {
		runTest(1500, 11, 0, 1, ExecType.CP, "mmr_sum");
	}

	@Test
	public void testMatrixMultRightSum_Larger_SP() {
		runTest(1500, 11, 0, 1, ExecType.SPARK, "mmr_sum");
	}

	@Test
	public void testMatrixMultRightSumPlus_Larger_CP() {
		runTest(1500, 11, 0, 1, ExecType.CP, "mmr_sum_plus");
	}

	@Test
	public void testMatrixMultRightSumPlus_Larger_SP() {
		runTest(1500, 11, 0, 1, ExecType.SPARK, "mmr_sum_plus");
	}

	@Test
	public void testMatrixMultRightSumPlusOnOverlap_Larger_CP() {
		runTest(1500, 11, 0, 1, ExecType.CP, "mmr_sum_plus_2");
	}

	@Test
	public void testMatrixMultRightSumPlusOnOverlap_Larger_SP() {
		// be aware that with multiple blocks it is likely that the small blocks
		// initially compress, but is to large for overlapping state will decompress.
		// In this test it does not decompress
		runTest(1010, 11, 0, 1, ExecType.SPARK, "mmr_sum_plus_2");
	}

	@Test
	@Ignore // WIP if we should decompress here.
	public void testMatrixMultRightSumPlusOnOverlapDecompress_Larger_SP() {
		// be aware that with multiple blocks it is likely that the small blocks
		// initially compress, but is to large for overlapping state therefor will decompress.
		// In this test it decompress the second small block but keeps the first in overlapping state.
		compressTest(1110, 10, 1.0, ExecType.SPARK, 1, 6, 1, 1, 2, "mmr_sum_plus_2");
	}

	@Test
	public void testMatrixMultLeftSum_CP() {
		runTest(1500, 1, 0, 1, ExecType.CP, "mml_sum");
	}

	@Test
	// @Ignore
	public void testMatrixMultLeftSum_SP_SmallerThanLeft() {
		try{
			runTest(1500, 1, 2, 1, ExecType.SPARK, "mml_sum");
		}
		catch(Exception e){
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testMatrixMultLeftSum_SP_LargerThanLeft() {
		runTest(1500, 11, 0, 1, ExecType.SPARK, "mml_sum");
	}

	/**
	 * Override default configuration with custom test configuration to ensure scratch space and local temporary
	 * directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		return TEST_CONF_FILE;
	}
}
