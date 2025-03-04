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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;
import java.util.Map.Entry;
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
public class TransformCompressedTestMultiCol {
	protected static final Log LOG = LogFactory.getLog(TransformCompressedTestMultiCol.class.getName());

	private final FrameBlock data;
	private final int k;

	public TransformCompressedTestMultiCol(FrameBlock data, int k) {
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

			ValueType[] kPlusCols = new ValueType[100];

			Arrays.fill(kPlusCols, ValueType.BOOLEAN);

			FrameBlock[] blocks = new FrameBlock[] {//
				TestUtils.generateRandomFrameBlock(16, //
					new ValueType[] {ValueType.UINT4, ValueType.UINT8, ValueType.UINT4}, 231), //
				TestUtils.generateRandomFrameBlock(10, //
					new ValueType[] {ValueType.BOOLEAN, ValueType.UINT8, ValueType.UINT4}, 231), //
				new FrameBlock(new ValueType[] {ValueType.BOOLEAN, ValueType.INT32, ValueType.INT32}, 100), //
				TestUtils.generateRandomFrameBlock(11, //
					new ValueType[] {ValueType.UINT4, ValueType.BOOLEAN, ValueType.FP32}, 231, 0.2),
				TestUtils.generateRandomFrameBlock(432, //
					new ValueType[] {ValueType.UINT4, ValueType.BOOLEAN, ValueType.FP32}, 231, 0.2),
				TestUtils.generateRandomFrameBlock(12, //
					new ValueType[] {ValueType.UINT4, ValueType.BOOLEAN, ValueType.FP32}, 231, 0.9),
				TestUtils.generateRandomFrameBlock(14, //
					new ValueType[] {ValueType.UINT4, ValueType.BOOLEAN, ValueType.FP32}, 231, 0.99),

				TestUtils.generateRandomFrameBlock(5, kPlusCols, 322),
				TestUtils.generateRandomFrameBlock(1020, kPlusCols, 322),
				FrameLibCompress.compress(TestUtils.generateRandomFrameBlock(1030, new ValueType[] {
					ValueType.UINT4, ValueType.BOOLEAN, ValueType.UINT4}, 231, 0.0), 2),
				FrameLibCompress.compress(TestUtils.generateRandomFrameBlock(1030, new ValueType[] {
					ValueType.UINT4, ValueType.BOOLEAN, ValueType.UINT4}, 231, 0.5), 2),
					
	

			};
			blocks[2].ensureAllocatedColumns(20);

			for(FrameBlock block : blocks) {
				for(int k : threads) {
					tests.add(new Object[] {block, k});
				}
			}

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		return tests;
	}

	@Test
	public void testRecode() {
		test("{recode:[C1, C2, C3]}");
	}

	@Test
	public void testDummyCode() {
		test("{dummycode:[C1,C2,C3]}");
	}

	@Test
	public void testDummyCodeV2() {
		test("{ids:true, dummycode:[1,2,3]}");
	}

	@Test
	public void testBin() {
		test(
			"{ids:true, bin:[{id:1, method:equi-width, numbins:4},{id:2, method:equi-width, numbins:16},{id:3, method:equi-width, numbins:32}]}");
	}

	@Test
	public void testBin3() {
		test("{ids:true, bin:[{id:2, method:equi-width, numbins:2},{id:3, method:equi-width, numbins:200}]}");
	}

	@Test
	public void passThrough() {
		test("{ids:true}");
	}

	@Test
	public void testBinToDummy() {
		test(
			"{ids:true, bin:[{id:1, method:equi-height, numbins:10},{id:2, method:equi-height, numbins:10},{id:3, method:equi-height, numbins:40}], dummycode:[1,2,3] }");
	}

	@Test
	public void testHash() {
		test("{ids:true, hash:[1,2,3], K:10}");
	}

	@Test
	public void testHashToDummy() {
		test("{ids:true, hash:[1,2,3], K:10, dummycode:[1,2]}");
	}

	public void test(String spec) {
		try {
			FrameBlock meta = null;
			MultiColumnEncoder encoderCompressed = EncoderFactory.createEncoder(spec, data.getColumnNames(),
				data.getNumColumns(), meta);

			MatrixBlock outCompressed = encoderCompressed.encode(data, k, true);
			MultiColumnEncoder encoderNormal = EncoderFactory.createEncoder(spec, data.getColumnNames(),
				data.getNumColumns(), meta);
			MatrixBlock outNormal = encoderNormal.encode(data, k);


			TestUtils.compareMatrices(outNormal, outCompressed, 0, "Not Equal after apply");
			meta = encoderNormal.getMetaData(meta);
			MultiColumnEncoder ec2 = EncoderFactory.createEncoder(spec, data.getColumnNames(), data.getNumColumns(),
				encoderNormal.getMetaData(null));

			FrameBlock metaBack = ec2.getMetaData(null);
			compareMeta(metaBack, meta);
			MatrixBlock outMeta12 = ec2.apply(data, k);

			TestUtils.compareMatrices(outNormal, outMeta12, 0, "Not Equal after apply2");

			MultiColumnEncoder ec = EncoderFactory.createEncoder(spec, data.getColumnNames(), data.getNumColumns(),
				encoderCompressed.getMetaData(null));

			MatrixBlock outMeta1 = ec.apply(data, k);
			TestUtils.compareMatrices(outNormal, outMeta1, 0, "Not Equal after apply");

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private void compareMeta(FrameBlock e, FrameBlock a){
		try{
			assertEquals(e.getNumRows(), a.getNumRows());
			if(e.getNumRows()>0){
				for(int i = 0; i < e.getNumColumns(); i++){
					Map<?, Integer> em = e.getColumn(i).getRecodeMap();
					Map<?, Integer> am = a.getColumn(i).getRecodeMap();
					for(Entry<?, Integer> eme : em.entrySet()){
							assertTrue(am.containsKey(eme.getKey()));
							assertEquals(eme.getValue(), am.get(eme.getKey()));
					}
				}
			}
		}
		catch(Exception ex){
			throw new RuntimeException(e.toString(), ex);
		}
	}
}
