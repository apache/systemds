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

package org.apache.sysds.test.functions.ooc;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.io.IOUtilFunctions;
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

public class BinaryWritePartitioningTest extends AutomatedTestBase {
	private static final String TEST_NAME = "UnaryWrite";
	private static final String TEST_DIR = "functions/ooc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BinaryWritePartitioningTest.class.getSimpleName() + "/";
	private static final String INPUT_NAME = "X";
	private static final String OUTPUT_NAME = "res";
	private static final int ROWS = 3000;
	private static final int COLS = 3000;
	private static final int BLEN = 1000;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
	}

	@Test
	public void testOOCBinaryWriteMultipartWhenParallelEnabled() {
		runBinaryWritePartitioningTest(true, true);
	}

	@Test
	public void testOOCBinaryWriteSingleFileWhenParallelDisabled() {
		runBinaryWritePartitioningTest(false, false);
	}

	private void runBinaryWritePartitioningTest(boolean parallelBinaryWrite, boolean expectMultipart) {
		Types.ExecMode oldPlatform = setExecMode(ExecMode.SINGLE_NODE);

		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String home = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = home + TEST_NAME + ".dml";
			File configFile = createParallelIOConfig(parallelBinaryWrite);
			programArgs = new String[] {
				"-config", configFile.getPath(),
				"-explain", "-stats", "-ooc",
				"-args", input(INPUT_NAME), output(OUTPUT_NAME)};

			MatrixBlock in = MatrixBlock.randOperations(ROWS, COLS, 1.0, -1, 1, "uniform", 7);
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(FileFormat.BINARY);
			writer.writeMatrixToHDFS(in, input(INPUT_NAME), ROWS, COLS, BLEN, in.getNonZeros());
			HDFSTool.writeMetaDataFile(input(INPUT_NAME + ".mtd"), ValueType.FP64,
				new MatrixCharacteristics(ROWS, COLS, BLEN, in.getNonZeros()), FileFormat.BINARY);

			runTest(true, false, null, -1);

			int numBinaryFiles = countBinaryFiles(output(OUTPUT_NAME));
			boolean shouldBeMultipart = expectMultipart
				&& OptimizerUtils.getParallelBinaryWriteParallelism() > 1;
			if(shouldBeMultipart)
				Assert.assertTrue("Expected multipart binary output but found " + numBinaryFiles + " file(s).",
					numBinaryFiles > 1);
			else
				Assert.assertEquals("Expected single-file binary output.", 1, numBinaryFiles);

			MatrixBlock out = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME), FileFormat.BINARY, ROWS, COLS, BLEN,
				in.getNonZeros());
			Assert.assertEquals(ROWS, out.getNumRows());
			Assert.assertEquals(COLS, out.getNumColumns());
		}
		catch(Exception ex) {
			Assert.fail(ex.getMessage());
		}
		finally {
			resetExecMode(oldPlatform);
		}
	}

	private File createParallelIOConfig(boolean parallelIO) throws IOException {
		File baseConfig = getCurConfigFile();
		String xml = Files.readString(baseConfig.toPath(), StandardCharsets.UTF_8);
		String prop = "   <sysds.cp.parallel.io>" + parallelIO + "</sysds.cp.parallel.io>\n";
		String updated = xml.contains("</root>") ? xml.replace("</root>", prop + "</root>") : xml + "\n<root>\n" + prop + "</root>\n";

		File out = new File(getCurLocalTempDir(), "SystemDS-config-ooc-pario-" + parallelIO + ".xml");
		Files.writeString(out.toPath(), updated, StandardCharsets.UTF_8);
		return out;
	}

	private int countBinaryFiles(String path) throws IOException {
		Path outPath = new Path(path);
		FileSystem fs = IOUtilFunctions.getFileSystem(outPath);
		return IOUtilFunctions.getSequenceFilePaths(fs, outPath).length;
	}
}
