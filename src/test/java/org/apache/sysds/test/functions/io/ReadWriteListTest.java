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

import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.runtime.meta.MetaDataAll;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class ReadWriteListTest extends AutomatedTestBase {

	protected static final Log LOG = LogFactory.getLog(ReadWriteListTest.class.getName());

	private final static String TEST_NAME1 = "ListWrite";
	private final static String TEST_NAME2 = "ListRead";
	private final static String TEST_DIR = "functions/io/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ReadWriteListTest.class.getSimpleName() + "/";
	private final static double eps = 1e-6;
	
	private final static int rows = 350;
	private final static int cols = 110;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"L","R1"}) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"R2"}) );
	}

	@Test
	public void testListBinarySinglenode() {
		runListReadWriteTest(false, FileFormat.BINARY, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testListBinaryHybrid() {
		runListReadWriteTest(false, FileFormat.BINARY, ExecMode.HYBRID);
	}
	
	@Test
	public void testListTextSinglenode() {
		runListReadWriteTest(false, FileFormat.TEXT, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testListTextHybrid() {
		runListReadWriteTest(false, FileFormat.TEXT, ExecMode.HYBRID);
	}
	
	@Test
	public void testNamedListBinarySinglenode() {
		runListReadWriteTest(true, FileFormat.BINARY, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testNamedListBinaryHybrid() {
		runListReadWriteTest(true, FileFormat.BINARY, ExecMode.HYBRID);
	}
	
	@Test
	public void testNamedListTextSinglenode() {
		runListReadWriteTest(true, FileFormat.TEXT, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testNamedListTextHybrid() {
		runListReadWriteTest(true, FileFormat.TEXT, ExecMode.HYBRID);
	}
	
	//TODO support for Spark write/read
	
	private void runListReadWriteTest(boolean named, FileFormat format, ExecMode mode)
	{
		ExecMode modeOld = setExecMode(mode);
		
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME1);
			loadTestConfiguration(config);
			
			//run write
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-args", String.valueOf(rows),
				String.valueOf(cols), output("R1"), output("L"), format.toString(), String.valueOf(named)};
			runTest(true, false, null, -1);
			double val1 = HDFSTool.readDoubleFromHDFSFile(output("R1"));
			
			//check no crc files
			//disabled due to delete on exist, but for temporary validation via delete
			//File[] files = new File(output("L")).listFiles();
			//LOG.error(Arrays.toString(files));
			//Assert.assertFalse(Arrays.stream(files).anyMatch(f -> f.getName().endsWith(".crc")));
			
			//run read
			fullDMLScriptName = HOME + TEST_NAME2 + ".dml";
			programArgs = new String[]{"-args", output("L"), output("R2")};
			runTest(true, false, null, -1);
			
			//check meta data, incl implicitly format 
			MetaDataAll meta = getMetaData("L", OUTPUT_DIR);
			Assert.assertEquals(format.toString(), meta.getFormatTypeString());
			Assert.assertEquals(4, meta.getDim1());
			Assert.assertEquals(1, meta.getDim2());
			
			double val2 = HDFSTool.readDoubleFromHDFSFile(output("R2"));
			Assert.assertEquals(Double.valueOf(val1), Double.valueOf(val2), eps);
		}
		catch(IOException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(modeOld);
		}
	}
}
