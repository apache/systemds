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

package org.apache.sysds.test.functions.rewrite;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class RewriteSPCheckpoint extends AutomatedTestBase {

	protected static final Log LOG = LogFactory.getLog(RewriteSPCheckpoint.class.getName());
	private static final String TEST_NAME1 = "RewriteSPCheckpoint";
	private static final String TEST_NAME2 = "RewriteSPCheckpointRemove";

	private static final String TEST_DIR = "functions/rewrite/SPCheckpoint/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteSPCheckpoint.class.getSimpleName() + "/";

	private static final int dim1 = 1324;
	private static final int dim2 = 1100;

	private static final double sparsity = 0.7;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2));
	}

	@Test
	public void testRewriteCheckpointTransientWrite() {
		testRewriteCheckpoint(TEST_NAME1, true);
	}

	@Test
	public void testRewriteCheckpointTransientWriteRemove() {
		testRewriteCheckpoint(TEST_NAME2, false);
	}

	private void testRewriteCheckpoint(String testName, boolean rewrite) {
		setOutputBuffering(true);
		Types.ExecMode platformOld = setExecMode(ExecMode.SPARK);
		try {

			TestConfiguration config = getTestConfiguration(testName);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testName + ".dml";
			programArgs = new String[] {"-explain", "-stats", "-args", input("A")};
			write(TestUtils.generateTestMatrixBlock(dim1, dim2, -1, 1, sparsity, 6), input("A"));
			runTest(null);

			assertTrue(rewrite == heavyHittersContainsString("sp_chkpoint"));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		finally {

			rtplatform = platformOld;
		}
	}

	public void write(MatrixBlock mb, String path) {
		try {
			MatrixWriter w = MatrixWriterFactory.createMatrixWriter(FileFormat.BINARY);
			w.writeMatrixToHDFS(mb, path, mb.getNumRows(), mb.getNumColumns(), 1000, mb.getNonZeros());
			MatrixCharacteristics mc = new MatrixCharacteristics(mb.getNumRows(), mb.getNumColumns(), 1000,
				mb.getNonZeros());
			HDFSTool.writeMetaDataFile(path + ".mtd", ValueType.FP64, mc, FileFormat.BINARY);
		}
		catch(Exception e) {
			fail(e.getMessage());
		}
	}
}
