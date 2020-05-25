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

package org.apache.sysds.test.functions.builtin;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BuiltinComponentsTest extends AutomatedTestBase {
	private final static String TEST_NAME = "ConnectedComponents";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinComponentsTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"r"}));
	}

	@Test
	public void testConnectedComponents11CP() {
		runConnectedComponentsTest(11, 0, LopProperties.ExecType.CP);
	}
	
	@Test
	public void testConnectedComponents201CP() {
		runConnectedComponentsTest(201, 0, LopProperties.ExecType.CP);
	}
	
	@Test
	public void testConnectedComponents2001CP() {
		runConnectedComponentsTest(2001, 0, LopProperties.ExecType.CP);
	}
	
	@Test
	public void testConnectedComponents11Maxi100CP() {
		runConnectedComponentsTest(11, 100, LopProperties.ExecType.CP);
	}

	private void runConnectedComponentsTest(int numVertices, int maxi, ExecType instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-args",
				input("X"), String.valueOf(maxi), output("R")};

			//generate actual dataset (3 components)
			double[][] X = new double[numVertices-3][2];
			for(int i=0; i<numVertices-3; i++) {
				X[i][0] = i<(numVertices/2-1) ? i+1 : i+3;
				X[i][1] = i<(numVertices/2-1) ? i+2 : i+4;
			}
			writeInputMatrixWithMTD("X", X, true);

			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			for( int i=0; i<numVertices; i++ ) {
				int expected = i<(numVertices/2) ? (numVertices/2) :
					i==(numVertices/2) ? i+1 : numVertices;
				Assert.assertEquals(new Double(expected), dmlfile.get(new MatrixValue.CellIndex(i+1,1)));
			}
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
