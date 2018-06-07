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

package org.apache.sysml.test.integration.functions.jmlc;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.apache.sysml.api.jmlc.Connection;
import org.apache.sysml.api.jmlc.PreparedScript;
import org.apache.sysml.api.jmlc.ResultVariables;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;

public class NamespaceFunctionTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "foo.dml";
	private final static String TEST_NAME2 = "bar1.dml";
	private final static String TEST_NAME3 = "bar2.dml";
	
	private final static String TEST_NAME4 = "foo2.dml";
	private final static String TEST_NAME5 = "bar3.dml";
	
	private final static String TEST_DIR = "functions/jmlc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + NamespaceFunctionTest.class.getSimpleName() + "/";
	
	private final static int rows = 700;
	private final static int cols = 30;
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "F2" }) );
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "F2" }) );
	}
	
	@Test
	public void testJMLCNamespaceAcyclicDense() throws IOException {
		runJMLCNamespaceTest(false, false);
	}
	
	@Test
	public void testJMLCNamespaceAcyclicSparse() throws IOException {
		runJMLCNamespaceTest(true, false);
	}
	
	@Test
	public void testJMLCNamespaceCyclicDense() throws IOException {
		runJMLCNamespaceTest(false, true);
	}
	
	@Test
	public void testJMLCNamespaceCyclicSparse() throws IOException {
		runJMLCNamespaceTest(true, true);
	}
	

	private void runJMLCNamespaceTest(boolean sparse, boolean cyclic) 
		throws IOException
	{
		TestConfiguration config = getTestConfiguration(TEST_NAME1);
		loadTestConfiguration(config);
		
		//load scripts and create prepared script
		Connection conn = new Connection();
		Map<String,String> nsscripts = new HashMap<>();
		String script1 = null;
		if( cyclic ) {
			script1 = conn.readScript(SCRIPT_DIR + TEST_DIR + TEST_NAME4);
			nsscripts.put(TEST_NAME4, conn.readScript(SCRIPT_DIR + TEST_DIR + TEST_NAME4));
			nsscripts.put(TEST_NAME5, conn.readScript(SCRIPT_DIR + TEST_DIR + TEST_NAME5));
		}
		else {
			script1 = conn.readScript(SCRIPT_DIR + TEST_DIR + TEST_NAME1);
			nsscripts.put(TEST_NAME2, conn.readScript(SCRIPT_DIR + TEST_DIR + TEST_NAME2));
			nsscripts.put(TEST_NAME3, conn.readScript(SCRIPT_DIR + TEST_DIR + TEST_NAME3));
		}
		PreparedScript pstmt = conn.prepareScript(script1,
			nsscripts, Collections.emptyMap(), new String[]{"X"}, new String[]{"Z"}, false);
		
		//generate input data
		double sparsity = sparse ? sparsity2 : sparsity1;
		MatrixBlock X = MatrixBlock.randOperations(rows, cols, sparsity, -1, 1, "uniform", 7);
		
		//execute script and get result
		pstmt.setMatrix("X", X, true);
		ResultVariables rs = pstmt.executeScript();
		MatrixBlock Z = rs.getMatrixBlock("Z");
		IOUtilFunctions.closeSilently(conn);
		
		//compare results
		for(int i=0; i<rows; i++)
			for(int j=0; j<cols; j++)
				Assert.assertEquals(X.quickGetValue(i, j)+10,
					Z.quickGetValue(i, j), 1e-10);
	}
}
