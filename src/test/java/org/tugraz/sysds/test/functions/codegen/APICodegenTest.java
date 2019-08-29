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

package org.tugraz.sysds.test.functions.codegen;

import static org.tugraz.sysds.api.mlcontext.ScriptFactory.dml;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.After;
import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.api.jmlc.Connection;
import org.tugraz.sysds.api.jmlc.PreparedScript;
import org.tugraz.sysds.api.mlcontext.MLContext;
import org.tugraz.sysds.api.mlcontext.Script;
import org.tugraz.sysds.conf.DMLConfig;
import org.tugraz.sysds.conf.CompilerConfig.ConfigType;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.utils.Statistics;


public class APICodegenTest extends AutomatedTestBase
{
	protected final static String TEST_DIR = "org/tugraz/sysds/api/mlcontext";
	protected final static String TEST_NAME = "MLContext";

	private final static int rows = 100;
	private final static int cols = 1600;
	private final static double sparsity = 0.7;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}
	
	@Test
	public void testCodegenMLContext() {
		runMLContextParforDatasetTest(false);
	}
	
	@Test
	public void testCodegenJMLCTest() {
		runMLContextParforDatasetTest(true);
	}

	private void runMLContextParforDatasetTest(boolean jmlc) 
	{
		try {
			double[][] X = getRandomMatrix(rows, cols, -10, 10, sparsity, 76543); 
			MatrixBlock mX = DataConverter.convertToMatrixBlock(X); 
			
			String s = "X = read(\"/tmp\");"
				+ "R = colSums(X/rowSums(X));"
				+ "write(R, \"tmp2\")";
			
			//execute scripts
			if( jmlc ) {
				DMLScript.STATISTICS = true;
				Connection conn = new Connection(ConfigType.CODEGEN_ENABLED, 
					ConfigType.ALLOW_DYN_RECOMPILATION);
				PreparedScript pscript = conn.prepareScript(
					s, new String[]{"X"}, new String[]{"R"}); 
				pscript.setMatrix("X", mX, false);
				pscript.executeScript();
				conn.close();
				System.out.println(Statistics.display());
			}
			else {
				SparkConf conf = SparkExecutionContext.createSystemDSSparkConf()
					.setAppName("MLContextTest").setMaster("local");
				JavaSparkContext sc = new JavaSparkContext(conf);
				MLContext ml = new MLContext(sc);
				ml.setConfigProperty(DMLConfig.CODEGEN, "true");
				ml.setStatistics(true);
				Script script = dml(s).in("X", mX).out("R");
				ml.execute(script);
				ml.resetConfig();
				sc.stop();
				ml.close();
			}
			
			//check for generated operator
			Assert.assertTrue(heavyHittersContainsSubString("spoofRA"));
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}

	@After
	public void tearDown() {
		super.tearDown();
	}
}
