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

package org.apache.sysds.test.functions.jmlc;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.api.jmlc.Connection;
import org.apache.sysds.api.jmlc.PreparedScript;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.utils.Statistics;
import org.apache.sysds.utils.stats.ParForStatistics;

public class JMLCParfor2ForCompileTest extends AutomatedTestBase 
{
	@Override
	public void setUp() {
		//do nothing
	}
	
	@Test
	public void testParfor2ParforCompile() {
		runJMLCParFor2ForTest(true);
	}
	
	@Test
	public void testParfor2ForCompile() {
		runJMLCParFor2ForTest(false);
	}

	private static void runJMLCParFor2ForTest(boolean par) {
		try {
			Connection conn = new Connection();
			conn.setConfigTypes(par, ConfigType.PARALLEL_LOCAL_OR_REMOTE_PARFOR);
			String script =
				"  X = rand(rows=10, cols=10);"
				+ "R = matrix(0, rows=10, cols=1)"
				+ "parfor(i in 1:nrow(X))"
				+ "  R[i,] = sum(X[i,])"
				+ "print(sum(R))";
			DMLScript.STATISTICS = true;
			Statistics.reset();
		
			PreparedScript pscript = conn.prepareScript(
				script, new String[]{}, new String[]{});
			pscript.executeScript();
			conn.close();
		}
		catch(Exception ex) {
			ex.printStackTrace();
			Assert.fail("JMLC parfor test failed: "+ex.getMessage());
		}
		
		//check for existing or non-existing parfor
		Assert.assertTrue(ParForStatistics.getOptCount()==(par?1:0));
	}
}
