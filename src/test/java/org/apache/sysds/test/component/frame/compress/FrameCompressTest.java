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

package org.apache.sysds.test.component.frame.compress;

import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.compress.CompressedFrameBlockFactory;
import org.apache.sysds.runtime.frame.data.compress.FrameCompressionSettings;
import org.apache.sysds.runtime.frame.data.lib.FrameLibCompress;
import org.apache.sysds.runtime.frame.data.lib.FrameLibDetectSchema;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class FrameCompressTest {
	protected static final Log LOG = LogFactory.getLog(FrameCompressTest.class.getName());

	@Test
	public void testSingleThread() {
		FrameBlock a = FrameCompressTestUtils.generateCompressableBlock(200, 5, 1232, ValueType.STRING);
		runTest(a, 1);
	}

	@Test
	public void testParallel() {
		FrameBlock a = FrameCompressTestUtils.generateCompressableBlock(200, 5, 1232, ValueType.STRING);
		runTest(a, 4);
	}

	@Test
	public void testParallelWithSchema() {
		FrameBlock a = FrameCompressTestUtils.generateCompressableBlock(200, 5, 1232, ValueType.STRING);
		FrameBlock sc = FrameLibDetectSchema.detectSchema(a, 4);
		a.applySchema(sc);
		runTest(a, 4);
	}

	@Test
	public void testParallelWithRandom() {
		FrameBlock a = FrameCompressTestUtils.generateCompressableBlockRandomTypes(200, 5, 1232);
		FrameBlock sc = FrameLibDetectSchema.detectSchema(a, 4);
		a = a.applySchema(sc);
		runTest(a, 4);
	}

	public void runTest(FrameBlock a, int k) {
		try {
			FrameBlock b = FrameLibCompress.compress(a, k);
			TestUtils.compareFrames(a, b, true);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	public void runTestConfig(FrameBlock a, FrameCompressionSettings cs) {
		try {
			FrameBlock b = CompressedFrameBlockFactory.compress(a, cs);
			TestUtils.compareFrames(a, b, true);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

}
