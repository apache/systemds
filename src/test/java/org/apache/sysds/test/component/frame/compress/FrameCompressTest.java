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
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.lib.FrameLibCompress;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.frame.array.FrameArrayTests;
import org.junit.Test;

public class FrameCompressTest {
	protected static final Log LOG = LogFactory.getLog(FrameCompressTest.class.getName());

	@Test
	public void testCompressionStatisticsConstruction() {
		try {

			FrameBlock a = generateCompressableBlock(200, 5, 1232);
			FrameBlock b = FrameLibCompress.compress(a, 1);
			TestUtils.compareFrames(a, b, true);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

	}

	private FrameBlock generateCompressableBlock(int rows, int cols, int seed) {
		Array<?>[] data = new Array<?>[cols];
		for(int i = 0; i < cols; i++) {
			data[i] = ArrayFactory.create(//
				FrameArrayTests.generateRandomStringNUniqueLengthOpt(rows, seed + i, i + 1, 55 + i));
		}
		return new FrameBlock(data);
	}
}
