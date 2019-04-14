/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.test.functions.lineage;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.runtime.lineage.LineageItem;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestUtils;

public class LineageTraceSelfReferenceTest extends AutomatedTestBase {
	
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME = "LineageTraceSelfReference";
	protected String TEST_CLASS_DIR = TEST_DIR + LineageTraceSelfReferenceTest.class.getSimpleName() + "/";
	
	protected static final int numRecords = 10;
	protected static final int numFeatures = 5;
	
	public LineageTraceSelfReferenceTest() {
		
	}
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
	}
	
	@Test
	public void testLineageTrace() {
		boolean old_simplification = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean old_sum_product = OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES;
		
		try {
			System.out.println("------------ BEGIN " + TEST_NAME + "------------");
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = false;
			
			int rows = numRecords;
			int cols = numFeatures;
			
			getAndLoadTestConfiguration(TEST_NAME);
			
			List<String> proArgs = new ArrayList<String>();
			
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add("-args");
			proArgs.add(input("X"));
			proArgs.add(output("X"));
			proArgs.add(output("Y"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			
			fullDMLScriptName = getScript();
			
			double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.8, -1);
			writeInputMatrixWithMTD("X", X, true);
			
			String expected_X_lineage =
					"(0) target/testTemp/functions/lineage/LineageTraceSelfReferenceTest/in/X\n" +
							"(1) false\n" +
							"(2) createvar (0) (1)\n" +
							"(6) rblk (2)\n" +
							"(9) cpvar (6)\n" +
							"(19) 7\n" +
							"(20) + (9) (19)\n" +
							"(21) cpvar (20)\n" +
							"(34) * (21) (21)\n" +
							"(38) * (34) (21)\n" +
							"(39) cpvar (38)\n" +
							"(52) 7\n" +
							"(53) + (39) (52)\n" +
							"(54) cpvar (53)\n";
			String expected_Y_lineage =
					expected_X_lineage + "(63) tsmm (54)\n";
			
			LineageItem.resetIDSequence();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			
			String X_lineage = readDMLLineageFromHDFS("X");
			String Y_lineage = readDMLLineageFromHDFS("Y");
			
			TestUtils.compareScalars(expected_X_lineage, X_lineage);
			TestUtils.compareScalars(expected_Y_lineage, Y_lineage);
		} finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = old_simplification;
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = old_sum_product;
		}
	}
}
