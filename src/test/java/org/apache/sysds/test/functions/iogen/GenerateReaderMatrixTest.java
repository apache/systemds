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

package org.apache.sysds.test.functions.iogen;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public abstract class GenerateReaderMatrixTest extends AutomatedTestBase {

	protected final static String TEST_DIR = "functions/iogen/";
	protected final static String TEST_CLASS_DIR = TEST_DIR + GenerateReaderMatrixTest.class.getSimpleName() + "/";
	protected String sampleRaw;
	protected double[][] sampleMatrix;

	protected abstract String getTestName();

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(), new TestConfiguration(TEST_DIR, getTestName(), new String[] {"Y"}));
	}

	protected void generateRandomSymmetric(int size, double min, double max, double sparsity, boolean isSkew) {
		sampleMatrix = getRandomMatrix(size, size, min, max, sparsity, 714);
		int conf = isSkew ? -1 : 1;
		for(int i = 0; i < size; i++) {
			for(int j = 0; j <= i; j++) {

				if(i != j)
					sampleMatrix[i][j] = sampleMatrix[j][i] * conf;
				else
					sampleMatrix[i][j] = 0;
			}
		}
	}

	@SuppressWarnings("unused")
	protected void runGenerateReaderTest() {

		Types.ExecMode oldPlatform = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;

		try {
			CompilerConfig.FLAG_PARREADWRITE_TEXT = false;

			TestConfiguration config = getTestConfiguration(getTestName());
			loadTestConfiguration(config);

			MatrixBlock sampleMB = DataConverter.convertToMatrixBlock(sampleMatrix);

			String HOME = SCRIPT_DIR + TEST_DIR;
			File directory = new File(HOME);
			if (! directory.exists()){
				directory.mkdir();
			}
			String dataPath = HOME + "matrix_data.raw";
			int clen = sampleMatrix[0].length;
			writeRawString(sampleRaw, dataPath);
			GenerateReader.GenerateReaderMatrix gr = new GenerateReader.GenerateReaderMatrix(sampleRaw, sampleMB);

			MatrixReader mr= gr.getReader();
			MatrixBlock matrixBlock = mr.readMatrixFromHDFS(dataPath, -1, clen, -1, -1);
		}
		catch(Exception exception) {
			exception.printStackTrace();
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

	private static void writeRawString(String raw, String fileName) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
		writer.write(raw);
		writer.close();
	}
}
