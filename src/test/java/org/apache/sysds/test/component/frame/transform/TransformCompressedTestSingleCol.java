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

package org.apache.sysds.test.component.frame.transform;

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.lib.FrameLibCompress;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.CompressedEncode;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class TransformCompressedTestSingleCol {
	protected static final Log LOG = LogFactory.getLog(TransformCompressedTestSingleCol.class.getName());

	private final FrameBlock data;
	private final int k;

	public TransformCompressedTestSingleCol(FrameBlock data, int k) {
		Thread.currentThread().setName("test_transformThread");
		Logger.getLogger(CommonThreadPool.class.getName()).setLevel(Level.OFF);
		CompressedEncode.ROW_PARALLELIZATION_THRESHOLD = 10;
		this.data = data;
		this.k = k;
	}

	@Parameters
	public static Collection<Object[]> data() {
		final ArrayList<Object[]> tests = new ArrayList<>();
		final int[] threads = new int[] {1, 4};
		try {
			FrameBlock[] blocks = new FrameBlock[] {
				TestUtils.generateRandomFrameBlock(100, new ValueType[] {ValueType.UINT4}, 231),
				TestUtils.generateRandomFrameBlock(100, new ValueType[] {ValueType.UINT4}, 231, 0.2),
				TestUtils.generateRandomFrameBlock(100, new ValueType[] {ValueType.UINT4}, 231, 1.0),
				TestUtils.generateRandomFrameBlock(100, new ValueType[] {ValueType.UINT4}, 231, 1.0),

				FrameLibCompress
					.compress(TestUtils.generateRandomFrameBlock(103, new ValueType[] {ValueType.UINT4}, 231, 1.0), 2),
				FrameLibCompress
					.compress(TestUtils.generateRandomFrameBlock(235, new ValueType[] {ValueType.UINT4}, 23132, 0.0), 2),

				// Above block size of number of unique elements
				TestUtils.generateRandomFrameBlock(1200, new ValueType[] {ValueType.FP32}, 231, 0.1),};

			blocks[3].set(40, 0, "14");
			for(FrameBlock block : blocks)
				for(int k : threads)
					tests.add(new Object[] {block, k});
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		return tests;
	}

	@Test
	public void testRecode() {
		test("{recode:[C1]}");
	}

	@Test
	public void testDummyCode() {
		test("{dummycode:[C1]}");
	}

	@Test
	public void testBin() {
		test("{ids:true, bin:[{id:1, method:equi-width, numbins:4}]}");
	}

	@Test
	public void testBin2() {
		test("{ids:true, bin:[{id:1, method:equi-width, numbins:100}]}");
	}

	@Test
	public void testBin3() {
		test("{ids:true, bin:[{id:1, method:equi-width, numbins:2}]}");
	}

	@Test
	public void testBin4() {
		test("{ids:true, bin:[{id:1, method:equi-height, numbins:2}]}");
	}

	@Test
	public void testBin5() {
		test("{ids:true, bin:[{id:1, method:equi-height, numbins:10}]}");
	}

	@Test
	public void passThrough() {
		test("{ids:true}");
	}

	@Test
	public void testBinToDummy() {
		test("{ids:true, bin:[{id:1, method:equi-height, numbins:10}], dummycode:[1] }");
	}

	@Test
	public void testHash() {
		test("{ids:true, hash:[1], K:10}");
	}

	@Test
	public void testHashDomain1() {
		test("{ids:true, hash:[1], K:1}");
	}

	@Test
	public void testHashToDummy() {
		test("{ids:true, hash:[1], K:10, dummycode:[1]}");
	}

	@Test
	public void testHashToDummyDomain1() {
		test("{ids:true, hash:[1], K:1, dummycode:[1]}");
	}

	public void test(String spec) {
		try {

			FrameBlock meta = null;
			MultiColumnEncoder encoderNormal = EncoderFactory.createEncoder(spec, data.getColumnNames(),
				data.getNumColumns(), meta);
			MatrixBlock outNormal = encoderNormal.encode(data, k);

			MultiColumnEncoder encoderCompressed = EncoderFactory.createEncoder(spec, data.getColumnNames(),
				data.getNumColumns(), meta);
			MatrixBlock outCompressed = encoderCompressed.encode(data, k, true);

			TestUtils.compareMatrices(outNormal, outCompressed, 0, "Not Equal after encode");

			// meta data is allowed to be different but!
			// when applied inversely should return the same.

			MultiColumnEncoder ec = EncoderFactory.createEncoder(spec, data.getColumnNames(), data.getNumColumns(),
				encoderCompressed.getMetaData(null));

			MatrixBlock outMeta1 = ec.apply(data, k);


			TestUtils.compareMatrices(outNormal, outMeta1, 0, "Not Equal after apply");

			MultiColumnEncoder ec2 = EncoderFactory.createEncoder(spec, data.getColumnNames(), data.getNumColumns(),
				encoderNormal.getMetaData(null));

			MatrixBlock outMeta12 = ec2.apply(data, k);
			TestUtils.compareMatrices(outNormal, outMeta12, 0, "Not Equal after apply2");

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
