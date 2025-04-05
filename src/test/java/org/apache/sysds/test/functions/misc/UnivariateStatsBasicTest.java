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

package org.apache.sysds.test.functions.misc;

import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

/**
 * Regression test for sanity test with univariate statistics.
 * 
 */
public class UnivariateStatsBasicTest extends AutomatedTestBase 
{
	private static final String TEST_NAME = "Univar-Stats";
	private static final String TEST_NAME_DATAGEN = "genRandData4Univariate";

	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + UnivariateStatsBasicTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "R" }) );
	}
	
	@Test
	public void testUnivariateStatsSingleColumn()  {
		testUnivariateStats( false );
	}
	
	@Test
	public void testUnivariateStatsSingleColumnRewrites()  {
		testUnivariateStats( true );
	}
	
	private void testUnivariateStats( boolean rewrites )
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			//run univariate stats data generator
			fullDMLScriptName = "./scripts/perftest/datagen/"+TEST_NAME_DATAGEN+".dml";
			programArgs = new String[]{ "-args", "100000", "100", "10", "1", "2", "3", "4", input("uni.mtx") };
			runTest(true, false, null, -1);
			
			//write input types
			MatrixBlock mb = new MatrixBlock(1d);
			MatrixWriterFactory.createMatrixWriter(FileFormat.CSV)
				.writeMatrixToHDFS(mb, input("uni-types.csv"), 1, 1, 1, 1);
			HDFSTool.writeMetaDataFile(input("uni-types.csv.mtd"), ValueType.FP64, 
					new MatrixCharacteristics(1,1,1,1), FileFormat.CSV);
			
			//run univariate stats 
			fullDMLScriptName = "./scripts/algorithms/"+TEST_NAME+".dml";
			programArgs = new String[]{ "-explain", "-nvargs", "X="+input("uni.mtx"), 
				"TYPES="+input("uni-types.csv"), "STATS="+output("uni-stats.txt"), "CONSOLE_OUTPUT=TRUE" };
			runTest(true, false, null, -1);
		}
		catch (Exception e) {
			throw new RuntimeException(e);
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}	
	}	
}