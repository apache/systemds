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

package org.apache.sysds.test.functions.compress;

import static org.junit.Assert.assertTrue;

import java.io.File;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;


@net.jcip.annotations.NotThreadSafe
public class CompressRewriteSpark extends AutomatedTestBase {
	// private static final Log LOG = LogFactory.getLog(CompressRewriteSpark.class.getName());

	private static final String dataPath = "src/test/scripts/functions/compress/densifying/";
	private final static String TEST_DIR = "functions/compress/";

	protected String getTestClassDir() {
		return getTestDir() + this.getClass().getSimpleName() + "/";
	}

	protected String getTestName() {
		return "compress";
	}

	protected String getTestDir() {
		return "functions/compress/densifying/";
	}

	@Test
	public void testCompressInstruction_small() {
		compressTest(ExecMode.HYBRID, "01", "small.ijv");
	}

	@Test
	public void testCompressInstruction_large() {
		compressTest(ExecMode.HYBRID, "01", "large.ijv");
	}

	@Test
	public void testCompressInstruction_large_vector_right() {
		compressTest(ExecMode.HYBRID, "02", "large.ijv");
	}

	@Test
	public void testCompressionInstruction_colmean() {
		compressTest(ExecMode.HYBRID, "submean", "large.ijv");
	}

	@Test
	public void testCompressionInstruction_scale() {
		compressTest(ExecMode.HYBRID, "scale", "large.ijv");
	}

	@Test
	public void testCompressionInstruction_seq_large() {
		compressTest(ExecMode.HYBRID, "seq", "large.ijv");
	}

	@Test
	public void testCompressionInstruction_pca_large() {
		compressTest(ExecMode.HYBRID, "pca", "large.ijv");
	}

	public void compressTest(ExecMode instType, String name, String data) {

		OptimizerUtils.ALLOW_SCRIPT_LEVEL_COMPRESS_COMMAND =true;
		Types.ExecMode platformOld = setExecMode(instType);
		try {

			loadTestConfiguration(getTestConfiguration(getTestName()));

			fullDMLScriptName = SCRIPT_DIR + "/" + getTestDir() + "compress_" + name + ".dml";

			programArgs = new String[] {"-stats", "100","-explain", "-args", dataPath + data};

			String out = runTest(null).toString();

			Assert.assertTrue(out + "\nShould not containing spark compression instruction",
				!heavyHittersContainsString("sp_compress"));
			Assert.assertTrue(out + "\nShould not contain spark instruction on compressed input",
				!heavyHittersContainsString("sp_+"));

		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Exception in execution: " + e.getMessage(), false);
		}
		finally {
			rtplatform = platformOld;
		}
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(), new TestConfiguration(getTestClassDir(), getTestName()));
	}

	@Override
	protected File getConfigTemplateFile() {
		return new File(SCRIPT_DIR + TEST_DIR, "SystemDS-config-compress.xml");
	}
}
