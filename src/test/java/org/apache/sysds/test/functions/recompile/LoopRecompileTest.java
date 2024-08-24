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

package org.apache.sysds.test.functions.recompile;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.stats.RecompileStatistics;

public class LoopRecompileTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "loop_recompile";
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_CLASS_DIR = TEST_DIR + LoopRecompileTest.class.getSimpleName() + "/";
	private final static String DATA = DATASET_DIR + "wine/winequality-white.csv";
	
	@Override
	public void setUp()  {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "Rout" }) );
	}

	@Test
	public void testLoopWithoutIPA() {
		runLoopTest(false);
	}
	
	@Test
	public void testLoopWithIPA() {
		runLoopTest(true);
	}

	private void runLoopTest( boolean IPA )
	{
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME1));
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-args", DATA , "-explain", "-stats" };
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
			runTest(true, false, null, -1); 
			
			if( IPA )
				Assert.assertTrue(RecompileStatistics.getRecompiledSBDAGs() <= 4);
			else
				Assert.assertTrue(RecompileStatistics.getRecompiledSBDAGs() >= 4890);
		}
		finally {
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
		}
	}
}
