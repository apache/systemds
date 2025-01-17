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

package org.apache.sysds.test.functions.io;

import java.io.File;
import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class RenameIssueTest extends AutomatedTestBase {

	protected static final Log LOG = LogFactory.getLog(RenameIssueTest.class.getName());

	private final static String TEST_NAME1 = "Rename";
	private final static String TEST_DIR = "functions/io/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RenameIssueTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"L","R1"}) );
	}

	@Test
	public void testCSVSinglenode() {
		runRameTest(FileFormat.CSV, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testCSVHybrid() {
		runRameTest(FileFormat.CSV, ExecMode.HYBRID);
	}
	
	@Test
	public void testCSVSpark() {
		runRameTest(FileFormat.CSV, ExecMode.SPARK);
	}
	
	@Test
	public void testTextSinglenode() {
		runRameTest(FileFormat.TEXT, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTextHybrid() {
		runRameTest(FileFormat.TEXT, ExecMode.HYBRID);
	}
	
	@Test
	public void testTextSpark() {
		runRameTest(FileFormat.TEXT, ExecMode.SPARK);
	}
	
	@Test
	public void testBinarySinglenode() {
		runRameTest(FileFormat.BINARY, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testBinaryHybrid() {
		runRameTest(FileFormat.BINARY, ExecMode.HYBRID);
	}
	
	@Test
	public void testBinarySpark() {
		runRameTest(FileFormat.BINARY, ExecMode.SPARK);
	}
	
	private void runRameTest(FileFormat fmt, ExecMode mode)
	{
		ExecMode modeOld = setExecMode(mode);
		
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME1);
			loadTestConfiguration(config);
			
			MatrixBlock a = DataConverter.convertToMatrixBlock(getRandomMatrix(7, 7, -1, 1, 0.5, -1));
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(fmt);
			writer.writeMatrixToHDFS(a, input("A"), 
				(long)a.getNumRows(), (long)a.getNumColumns(), (int)a.getNonZeros(), 1000);
			HDFSTool.writeMetaDataFile(input("A")+".mtd", ValueType.FP64,
				new MatrixCharacteristics(7,7,1000), fmt);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-explain", 
				"-args", input("A"), fmt.toString().toLowerCase(), output("B")};
			runTest(true, false, null, -1);
			
			//check file existence (no rename to output)
			Assert.assertTrue(new File(input("A")).exists());
			Assert.assertTrue(new File(output("B")).exists());
		} 
		catch (IOException e) {
			e.printStackTrace();
			Assert.fail();
		}
		finally {
			resetExecMode(modeOld);
		}
	}
}
