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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.frame.data.columns.StringArray;
import org.apache.sysds.runtime.frame.data.compress.ArrayCompressionStatistics;
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
		runTest(a, 10);
	}

	@Test
	public void testParallelWithSchema() {
		try {
			FrameBlock a = FrameCompressTestUtils.generateCompressableBlock(200, 5, 1232, ValueType.STRING);
			FrameBlock sc = FrameLibDetectSchema.detectSchema(a, 10);
			a.applySchema(sc);
			runTest(a, 10);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}

	@Test
	public void testParallelWithRandom1() {
		testParallelWithRandom(132);
	}

	@Test
	public void testParallelWithRandom2() {
		testParallelWithRandom(321);
	}

	@Test
	public void testParallelWithRandom3() {
		testParallelWithRandom(1316262);
	}

	@Test
	public void testParallelWithRandom4() {
		testParallelWithRandom(53535251);
	}

	public void testParallelWithRandom(int seed) {
		FrameBlock a = FrameCompressTestUtils.generateCompressableBlockRandomTypes(200, 5, seed);
		runTest(a, 10);
	}

	@Test
	public void testParallelCompressAgain1() {
		testParallelWithRandom(3223);
	}

	public void testParallelCompressAgain(int seed) {
		FrameBlock a = FrameCompressTestUtils.generateCompressableBlockRandomTypes(200, 5, seed);
		FrameBlock b = FrameLibCompress.compress(a, 10);
		runTest(b, 10);
	}

	@Test
	public void testParallelIncompressable1() {
		testParallelIncompressable(132);
	}

	@Test
	public void testParallelIncompressable2() {
		testParallelIncompressable(321);
	}

	@Test
	public void testParallelIncompressable3() {
		testParallelIncompressable(1316262);
	}

	@Test
	public void testParallelIncompressable4() {
		testParallelIncompressable(53535251);
	}

	public void testParallelIncompressable(int seed) {
		FrameBlock a = TestUtils.generateRandomFrameBlock(200, 5, seed);
		runTest(a, 4);
	}

	@Test
	public void testParallelWithRandomIntegratedSchemaDetect() {
		FrameBlock a = FrameCompressTestUtils.generateCompressableBlockRandomTypes(200, 5, 1232);
		runTest(a, 4);
	}

	@Test
	public void testParallelWithRandomIntegratedSchemaDetectAllStrings() {
		FrameBlock a = FrameCompressTestUtils.generateCompressableBlockRandomTypes(200, 5, 1232);
		FrameBlock sc = new FrameBlock(new Array<?>[] {ArrayFactory.create(new String[] {"STRING"}),
			ArrayFactory.create(new String[] {"STRING"}), ArrayFactory.create(new String[] {"STRING"}),
			ArrayFactory.create(new String[] {"STRING"}), ArrayFactory.create(new String[] {"STRING"})});
		a = a.applySchema(sc);
		runTest(a, 10);
	}

	@Test
	public void testSingleWithRandomIntegratedSchemaDetectAllStrings() {
		FrameBlock a = FrameCompressTestUtils.generateCompressableBlockRandomTypes(200, 5, 1232);
		FrameBlock sc = new FrameBlock(new Array<?>[] {ArrayFactory.create(new String[] {"STRING"}),
			ArrayFactory.create(new String[] {"STRING"}), ArrayFactory.create(new String[] {"STRING"}),
			ArrayFactory.create(new String[] {"STRING"}), ArrayFactory.create(new String[] {"STRING"})});
		a = a.applySchema(sc);
		runTest(a, 1);
	}

	@Test
	public void testFailingDetect() {
		try {

			FrameBlock a = FrameCompressTestUtils.generateCompressableBlock(1000, 6, 2314, ValueType.INT32);
			Array<String> s = ArrayFactory.create(new String[] {"STRING"});
			FrameBlock sc = new FrameBlock(new Array<?>[] {s, s, s, s, s, s});
			FrameBlock as = a.applySchema(sc);
			StringArray cs = (StringArray) as.getColumn(2);
			StringArray m = spy(cs);

			doThrow(new RuntimeException()).when(m).changeTypeWithNulls(any(), anyInt(), anyInt());
			as.setColumn(2, m);

			FrameBlock b = FrameLibCompress.compress(as, 42);
			// LOG.error(b.getColumn(2));
			// LOG.error(a.getColumn(2));
			TestUtils.compareFrames(a, b, true);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail();
		}
	}

	@Test(expected = RuntimeException.class)
	public void test() {
		try {

			FrameBlock a = FrameCompressTestUtils.generateCompressableBlock(1000, 6, 2314, ValueType.INT32);
			Array<String> s = ArrayFactory.create(new String[] {"STRING"});
			FrameBlock sc = new FrameBlock(new Array<?>[] {s, s, s, s, s, s});
			FrameBlock as = a.applySchema(sc);
			StringArray cs = (StringArray) as.getColumn(2);
			StringArray m = spy(cs);

			when(m.statistics(anyInt())).thenReturn(new ArrayCompressionStatistics(0, 1, true, ValueType.STRING, true,
				FrameArrayType.STRING, 100000, 10, true));

			as.setColumn(2, m);

			FrameBlock b = FrameLibCompress.compress(as, 42);

			TestUtils.compareFrames(a, b, true);
		}
		catch(Exception e) {
			throw e;
		}
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
