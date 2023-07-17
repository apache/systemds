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
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.iogen.GenerateReader;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public abstract class GenerateReaderFrameTest extends AutomatedTestBase {

	protected final static String TEST_DIR = "functions/iogen/";
	protected final static String TEST_CLASS_DIR = TEST_DIR + GenerateReaderFrameTest.class.getSimpleName() + "/";

	protected abstract String getTestName();

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(), new TestConfiguration(TEST_DIR, getTestName(), new String[] {"Y"}));
	}

	protected void runGenerateReaderTest(String sampleRawFileName, String sampleFrameFileName, String schemaFileName,
		boolean parallel) {

		Types.ExecMode oldPlatform = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;

		try {
			CompilerConfig.FLAG_PARREADWRITE_TEXT = false;

			TestConfiguration config = getTestConfiguration(getTestName());
			loadTestConfiguration(config);
			String sampleRawDelimiter = "\t";
			String dataFileName = sampleRawFileName;

			long rows = 200;
			Util util = new Util();

			Types.ValueType[] sampleSchema = util.getSchema(schemaFileName);
			int ncols = sampleSchema.length;
			String[][] sampleFrameStrings = util.loadFrameData(sampleFrameFileName, sampleRawDelimiter, ncols);
			FrameBlock sampleFrame = new FrameBlock(sampleSchema, sampleFrameStrings);
			String sampleRaw = util.readEntireTextFile(sampleRawFileName);
			GenerateReader.GenerateReaderFrame gr = new GenerateReader.GenerateReaderFrame(sampleRaw, sampleFrame,
				parallel);
			FrameReader fr = gr.getReader();
			FrameBlock frameBlock = fr.readFrameFromHDFS(dataFileName, sampleSchema, rows, sampleSchema.length);
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
}