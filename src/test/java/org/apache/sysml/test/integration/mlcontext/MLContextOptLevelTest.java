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

import org.apache.sysml.api.mlcontext.MLContext.ExplainLevel;
import org.apache.sysml.api.mlcontext.MLResults;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class MLContextOptLevelTest extends MLContextTestBase
{
	private final static int rows = 200;
	private final static int cols = 100;

	@Test
	public void testOptLevel1() {
		runMLContextOptLevelTest(1);
	}
	
	@Test
	public void testOptLevel2() {
		runMLContextOptLevelTest(2);
	}

	private void runMLContextOptLevelTest(int optLevel)
	{
		try
		{
			String s = "R = sum(matrix(0,"+rows+","+cols+") + 7);";
			ml.setExplain(true);
			ml.setExplainLevel(ExplainLevel.RUNTIME);
			ml.setStatistics(true);
			ml.setConfigProperty(DMLConfig.OPTIMIZATION_LEVEL, String.valueOf(optLevel));
			
			Script script = dml(s).out("R");
			MLResults results = ml.execute(script);
			
			//check result correctness
			TestUtils.compareScalars(results.getDouble("R"), rows*cols*7, 0.000001);
		
			//check correct opt level
			Assert.assertTrue(heavyHittersContainsString("+") == (optLevel==1));
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}
}
