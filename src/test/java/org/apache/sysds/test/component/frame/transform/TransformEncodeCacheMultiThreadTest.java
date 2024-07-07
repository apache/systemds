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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderType;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.TestUtils;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.*;

@RunWith(value = Parameterized.class)
public class TransformEncodeCacheMultiThreadTest {
	protected static final Log LOG = LogFactory.getLog(TransformEncodeCacheMultiThreadTest.class.getName());

	private final FrameBlock data;
	private final int k;
	private final List<String> specs;
	private final EncoderType encoderType;

	public TransformEncodeCacheMultiThreadTest(FrameBlock data, int k, List<String> specs, EncoderType encoderType) {
		this.data = data;
		this.k = k;
		this.specs = specs;
		this.encoderType = encoderType;
	}

	@BeforeClass
	public static void setUp() {
		FrameBlock setUpData = TestUtils.generateRandomFrameBlock(10, new ValueType[]{ValueType.FP32}, 231);

		MultiColumnEncoder encoder = EncoderFactory.createEncoder("{recode:[C1]}", setUpData.getColumnNames(), setUpData.getNumColumns(), null);
		try {
			encoder.encode(setUpData, 1); // Setup encoding with a single thread
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Parameters
	public static Collection<Object[]> data() {
		final ArrayList<Object[]> tests = new ArrayList<>();
		final int[] threads = new int[] {1, 2, 4, 8};
		FrameBlock testData = TestUtils.generateRandomFrameBlock(
				10,
				new ValueType[]{
						ValueType.FP32, ValueType.FP32, ValueType.FP32,
						ValueType.FP32, ValueType.FP32, ValueType.FP32,
						ValueType.FP32, ValueType.FP32, ValueType.FP32,
						ValueType.FP32, ValueType.FP32},
				231
		);
		List<List<String>> specLists = Arrays.asList(
				Arrays.asList(
						"{recode:[C1]}", "{recode:[C2]}",
						"{recode:[C3]}", "{recode:[C4]}",
						"{recode:[C5]}", "{recode:[C6]}",
						"{recode:[C7]}", "{recode:[C8]}",
						"{recode:[C9]}", "{recode:[C10]}"),
				Arrays.asList(
						"{ids:true, bin:[{id:1, method:equi-width, numbins:4}]}",
						"{ids:true, bin:[{id:2, method:equi-width, numbins:4}]}",
						"{ids:true, bin:[{id:3, method:equi-width, numbins:4}]}",
						"{ids:true, bin:[{id:4, method:equi-width, numbins:4}]}",
						"{ids:true, bin:[{id:5, method:equi-width, numbins:4}]}",
						"{ids:true, bin:[{id:6, method:equi-width, numbins:4}]}",
						"{ids:true, bin:[{id:7, method:equi-width, numbins:4}]}",
						"{ids:true, bin:[{id:8, method:equi-width, numbins:4}]}",
						"{ids:true, bin:[{id:9, method:equi-width, numbins:4}]}",
						"{ids:true, bin:[{id:10, method:equi-width, numbins:4}]}")
		);
		List<EncoderType> encoderTypes = Arrays.asList(EncoderType.Recode, EncoderType.Bin);

		for (int index = 0; index < specLists.size(); index++){
			for(int k : threads)
				tests.add(new Object[]{testData, k, specLists.get(index), encoderTypes.get(index)});
		}
		return tests;
	}

	@Test
	public void assertThatMatrixBlockIsEqualForAllThreadNumbers() {
		// Assert that the resulting matrix block is equal independent of the number of threads
		try {
			FrameBlock meta = null;
			List<MultiColumnEncoder> encoders = new ArrayList<>();
			for (String spec: specs) {
				encoders.add(EncoderFactory.createEncoder(spec, data.getColumnNames(), data.getNumColumns(), meta));
			}

			final int[] threads = new int[] {2, 4, 8, 3};

			for (MultiColumnEncoder encoder : encoders) {

				MatrixBlock singleThreadResult = encoder.encode(data, 1);
				for (int k : threads) {
					MatrixBlock multiThreadResult = encoder.encode(data, k);
					compareMatrixBlocks(singleThreadResult, multiThreadResult);
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private void compareMatrixBlocks(MatrixBlock mb1, MatrixBlock mb2) {
		assertEquals("Encoded matrix blocks should be equal", mb1, mb2);
	}
}