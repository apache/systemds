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

package org.apache.sysds.test.functions.linearization;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.Arrays;

import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

//additional imports
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.lops.compile.linearization.IDagLinearizer;
import org.apache.sysds.lops.compile.linearization.IDagLinearizerFactory;
import org.apache.sysds.lops.compile.linearization.IDagLinearizerFactory.DagLinearizer;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.lops.BinaryScalar;
import org.apache.sysds.lops.Data;

public class ILinearizeTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/linearization/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ILinearizeTest.class.getSimpleName() + "/";
	private final static String TEST_NAME = "LinearizationTest";
	private static String CONF_FILE = "";

	enum Config {
		BREADTH_FIRST("SystemDS-config-breadth-first.xml"),
		DEPTH_FIRST("SystemDS-config-depth-first.xml"),
		MIN_INTERMEDIATE("SystemDS-config-min-intermediate.xml"),
		MAX_PARALLELIZE("SystemDS-config-max-parallelize.xml"),
		PIPELINE_DEPTH_FIRST("SystemDS-config-pipeline-depth-first.xml"),
		RESOURCE_AWARE_FAST("SystemDS-config-resource-aware-fast.xml"),
		RESOURCE_AWARE_OPTIMAL("SystemDS-config-resource-aware-optimal.xml");

		public final String filePath;

		Config(String filePath) {
			this.filePath = filePath;
		}
	}

	private String getConfPath(Config config) {
		return SCRIPT_DIR + TEST_DIR + config.filePath;
	}

	@Override
	public void setUp() {
		setOutputBuffering(true);
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testLinearize_Pipeline() {

		try {
			DMLConfig dmlconf = DMLConfig.readConfigurationFile(getConfPath(Config.PIPELINE_DEPTH_FIRST));
			ConfigurationManager.setGlobalConfig(dmlconf);
			Assert.assertTrue(ConfigurationManager.getLinearizationOrder()==DagLinearizer.PIPELINE_DEPTH_FIRST);
		}
		catch(Exception e) {
			e.printStackTrace();
		}

		// Set up Test example
		// For testing we don't build a runnable programm
		// We only want to know, whether the assignement of pipeline ids work as intended
		//			   n1   n2
		//				 \ / \
		//		 o1	   n3  n4
		//		  \		   |  
		//		  |		   n5
		//	r1   o2 
		//	  \ /  \
		//	  r2	|
		//	  |	 |
		//	  r3	|
		//	  |	 |
		//	  r4	|
		//	  |  \ /
		//	  r5  y1
		//		 /  \
		//		y2  y5
		//		|   |   
		//		y3  |
		//		 \ /
		//		 y6

		List<Lop> lops = new ArrayList<>();
		// Dummy inputs for filling the inputs of the lops
		// Needed for the constructors
		Lop d1 = Data.createLiteralLop(ValueType.INT32, "1");
		Lop d2 = Data.createLiteralLop(ValueType.INT32, "1");

		// Start with creating the leafs, as the constructors await inputs
		// Disconnected graph 
		Lop n1 = new BinaryScalar(d1, d2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		Lop n2 = new BinaryScalar(d1, d2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		Lop n3 = new BinaryScalar(n1, n2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		Lop n4 = new BinaryScalar(n2, d2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		Lop n5 = new BinaryScalar(n4, d2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		lops.add(n1);
		lops.add(n2);
		lops.add(n3);
		lops.add(n4);
		lops.add(n5);

		// First pipeline (after step 1)
		Lop o1 = new BinaryScalar(d1, d2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		Lop o2 = new BinaryScalar(o1, d2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		// Second pipeline (after step 1)
		Lop r1 = new BinaryScalar(d1, d2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		Lop r2 = new BinaryScalar(r1, o2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		Lop r3 = new BinaryScalar(r2, d2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		Lop r4 = new BinaryScalar(r3, d2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		Lop r5 = new BinaryScalar(r4, d2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		// Third pipeline (after step 1)
		Lop y1 = new BinaryScalar(o2, r4, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		Lop y2 = new BinaryScalar(y1, d2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		Lop y3 = new BinaryScalar(y2, d2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		Lop y4 = new BinaryScalar(y1, d2, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);
		Lop y5 = new BinaryScalar(y3, y4, OpOp2.PLUS, DataType.SCALAR, ValueType.INT32);

		// add all lops to the list
		lops.add(o1);
		lops.add(o2);
		lops.add(r1);
		lops.add(r2);
		lops.add(r3);
		lops.add(r4);
		lops.add(r5);
		lops.add(y1);
		lops.add(y2);
		lops.add(y3);
		lops.add(y4);
		lops.add(y5);

		// Remove dummy inputs
		lops.forEach(l -> {
			l.getInputs().remove(d1);
			l.getInputs().remove(d2);
		});

		// RUN LINEARIZATION
		IDagLinearizer dl = IDagLinearizerFactory.createDagLinearizer();
		dl.linearize(lops); //TODO results

		// Set up expected pipelines
		Map<Integer, List<Lop>> pipelineMap = new HashMap<>();
		pipelineMap.put(4, Arrays.asList(n1, n2, n3, n4, n5, o1, o2));
		pipelineMap.put(3, Arrays.asList(r1, r2, r3, r4, r5));
		pipelineMap.put(5, Arrays.asList(y1, y2, y3, y4, y5));

		// Check if all lops are in the correct pipeline
		pipelineMap.get(4).forEach(l -> {
			if(l.getPipelineID() != 4)
				fail("Pipeline ID not set correctly");
		});
		pipelineMap.get(3).forEach(l -> {
			if(l.getPipelineID() != 3)
				fail("Pipeline ID not set correctly");
		});
		pipelineMap.get(5).forEach(l -> {
			if(l.getPipelineID() != 5)
				fail("Pipeline ID not set correctly");
		});
	}

	@Test
	public void testBreadthFirstLinearization() {
		runLinearizationTest(Config.BREADTH_FIRST);
	}

	@Test
	public void testDepthFirstLinearization() {
		runLinearizationTest(Config.DEPTH_FIRST);
	}

	@Test
	public void testMinIntermediateLinearization() {
		runLinearizationTest(Config.MIN_INTERMEDIATE);
	}

	@Test
	public void testResourceAwareFastLinearization() {
		runLinearizationTest(Config.RESOURCE_AWARE_FAST);
	}
	
	@Test
	@Ignore
	public void testResourceAwareOptimalLinearization() {
		runLinearizationTest(Config.RESOURCE_AWARE_OPTIMAL);
	}

	private void runLinearizationTest(Config config) {
		//load config
		CONF_FILE = getConfPath(config);

		TestConfiguration testConfiguration = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(testConfiguration);

		//generate paths and commands
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-args", output("R")};
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + expectedDir();

		//run test
		runTest(true, false, null, -1);
		runRScript(true);

		//check that the correct linearization order is used
		assertEquals(config.toString(), ConfigurationManager.getLinearizationOrder().toString());

		//compare matrices
		HashMap<MatrixValue.CellIndex, Double> dmlFile = readDMLMatrixFromOutputDir("R");
		HashMap<MatrixValue.CellIndex, Double> rFile = readRMatrixFromExpectedDir("R");
		TestUtils.compareMatrices(dmlFile, rFile, 0, "Stat-DML", "Stat-R");
	}

	@Override
	protected File getConfigTemplateFile() {
		return new File(CONF_FILE);
	}
}
