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
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.fail;

@RunWith(value = Parameterized.class)
public class TransformEncodeCacheTest {
	protected static final Log LOG = LogFactory.getLog(TransformEncodeCacheTest.class.getName());

	private final FrameBlock data;
	private final int k;
	private final String spec;

	public TransformEncodeCacheTest(FrameBlock data, int k, String spec) {
		this.data = data;
		this.k = k;
		this.spec = spec;
	}

	@Parameters
	public static Collection<Object[]> data() {
		final ArrayList<Object[]> tests = new ArrayList<>();
		final int k = 1;
		final List<String> specs = Arrays.asList("{recode:[C1]}");
		FrameBlock testData = TestUtils.generateRandomFrameBlock(10, new ValueType[] {ValueType.UINT4}, 231);
		for (String spec : specs) {
			tests.add(new Object[] {testData, k, spec});
		}
		return tests;
	}

	@Test
	public void testCache() {
		test();
	}

	public void test() {
		try {
			FrameBlock meta = null;
			MultiColumnEncoder encoderNormal = EncoderFactory.createEncoder(spec, data.getColumnNames(),
				data.getNumColumns(), meta);
			long st1 = System.nanoTime();
			encoderNormal.encode(data, k);
			long et1 = System.nanoTime();
			long duration1 = et1 - st1;
			long st2 = System.nanoTime();
			encoderNormal.encode(data, k);
			long et2 = System.nanoTime();
			long duration2 = et2 - st2;
			System.out.println("First run (nanoseconds): " + duration1);
			System.out.println("Second run (nanoseconds): " + duration2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
