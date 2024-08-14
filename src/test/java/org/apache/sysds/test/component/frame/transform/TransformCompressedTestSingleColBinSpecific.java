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
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class TransformCompressedTestSingleColBinSpecific {
	protected static final Log LOG = LogFactory.getLog(TransformCompressedTestSingleColBinSpecific.class.getName());

	private final FrameBlock data;
	private final int k;

	public TransformCompressedTestSingleColBinSpecific(FrameBlock data, int k) {
		this.data = data;
		this.k = k;
	}

	@Parameters
	public static Collection<Object[]> data() {
		final ArrayList<Object[]> tests = new ArrayList<>();
		final int[] threads = new int[] {1, 4};
		try {

			FrameBlock data = TestUtils.generateRandomFrameBlock(120, new ValueType[] {ValueType.FP64}, 231);
			data.setSchema(new ValueType[] {ValueType.FP64});
			for(int k : threads) {
				tests.add(new Object[] {data, k});
			}

			FrameBlock data2 = TestUtils.generateRandomFrameBlock(1200, new ValueType[] {ValueType.FP64}, 231);
			data2.setSchema(new ValueType[] {ValueType.FP64});
			for(int k : threads) {
				tests.add(new Object[] {data2, k});
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		return tests;
	}

	@Test
	public void test1() {
		test(1);
	}

	@Test
	public void test2() {
		test(2);
	}

	@Test
	public void test3() {
		test(3);
	}

	@Test
	public void test4() {
		test(4);
	}

	@Test
	public void test30() {
		test(30);
	}

	public void test(int bin) {
		test("{ids:true, bin:[{id:1, method:equi-height, numbins:" + bin + "}], dummycode:[1] }", true);
	}

	@Test
	public void testAP1() {
		testAP(1);
	}

	@Test
	public void testAP2() {
		testAP(2);
	}

	@Test
	public void testAP3() {
		testAP(3);
	}

	@Test
	public void testAP4() {
		testAP(4);
	}

	@Test
	public void testAP30() {
		testAP(30);
	}

	public void testAP(int bin) {
		DMLScript.SEED = 132;
		test("{ids:true, bin:[{id:1, method:equi-height-approx, numbins:" + bin + "}], dummycode:[1] }", false);
	}

	public void test(String spec, boolean EQ) {
		try {

			FrameBlock meta = null;
			MultiColumnEncoder encoderCompressed = EncoderFactory.createEncoder(spec, data.getColumnNames(),
				data.getNumColumns(), meta);

			MatrixBlock outCompressed = encoderCompressed.encode(data, k, true);
			FrameBlock outCompressedMD = encoderCompressed.getMetaData(null);
			MultiColumnEncoder encoderNormal = EncoderFactory.createEncoder(spec, data.getColumnNames(),
				data.getNumColumns(), meta);
			MatrixBlock outNormal = encoderNormal.encode(data, k);
			FrameBlock outNormalMD = encoderNormal.getMetaData(null);
			TestUtils.compareFrames(outNormalMD, outCompressedMD, true);
			TestUtils.compareMatrices(outNormal, outCompressed, 0, "Not Equal after apply");

			if(EQ){
				// Assert that each bucket has the same number of elements
				MatrixBlock colSum = outNormal.colSum();
				for(int i = 0; i < colSum.getNumColumns(); i++)
					assertEquals(colSum.get(0, 0), colSum.get(0, i), 0.001);
			}
		}

		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
