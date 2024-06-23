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
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

@RunWith(value = Parameterized.class)
public class TransformEncodeCacheTest {
	protected static final Log LOG = LogFactory.getLog(TransformEncodeCacheTest.class.getName());

	private final FrameBlock data;
	private final int k;
	private final List<String> specs;

	public TransformEncodeCacheTest(FrameBlock data, int k, List<String> specs) {
		this.data = data;
		this.k = k;
		this.specs = specs;
	}

	@Parameters
	public static Collection<Object[]> data() {
		final ArrayList<Object[]> tests = new ArrayList<>();
		final int k = 1;
		FrameBlock testData = TestUtils.generateRandomFrameBlock(
				10,
				new ValueType[]{ValueType.FP32, ValueType.FP32, ValueType.FP32, ValueType.FP32},
				231
		);
		List<String> specs = Arrays.asList(
				"{recode:[C1]}", //compiles everything, should be slow
				"{recode:[C2]}", // uses compiled systems but cannot retrieve any data
				"{recode:[C3]}", // uses compiled systems but cannot retrieve any data
				"{recode:[C4]}", // uses compiled systems but cannot retrieve any data
				// Average runtime over the 3 cases
				"{recode:[C2]}", // uses cache entry, should be faster now
				"{recode:[C3]}", // uses cache entry, should be faster now
				"{recode:[C4]}" // uses cache entry, should be faster now
				// Average runtime over the 3 cases
		);
		tests.add(new Object[]{testData, k, specs});
		return tests;
	}

	@Test
	public void testCache() {
		List<Long> durations = new ArrayList<>();
		try {
			FrameBlock meta = null;
			List<MultiColumnEncoder> encoders = new ArrayList<>();
			for (String spec: specs) {
				encoders.add(EncoderFactory.createEncoder(spec, data.getColumnNames(), data.getNumColumns(), meta));
			}
			for (MultiColumnEncoder encoder: encoders) {
				durations.add(measureEncodeTime(encoder, data, k));
			}
			analyzePerformance(durations);

		} catch (Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private long measureEncodeTime(MultiColumnEncoder encoder, FrameBlock data, int k) {
		long startTime = System.nanoTime();
		encoder.encode(data, k);
		long endTime = System.nanoTime();
		return endTime - startTime;
	}

	private void analyzePerformance(List<Long> durations) {
		long firstRun = durations.get(0);
		List<Long> runsWithoutCache = durations.subList(1,4);
		List<Long> runsWithCache = durations.subList(4, durations.size());
		double averageWithoutCache = runsWithoutCache.stream().mapToLong(Long::longValue).average().orElse(Double.NaN);
		double averageWithCache = runsWithCache.stream().mapToLong(Long::longValue).average().orElse(Double.NaN);

		DecimalFormat df = new DecimalFormat("#.#####");
		LOG.info("First run (seconds): " + df.format(firstRun/1_000_000_000.0));
		LOG.info("Average time without cache (seconds): " + df.format(averageWithoutCache/1_000_000_000.0));
		LOG.info("Average time with cache (seconds): " + df.format(averageWithCache/1_000_000_000.0));

		assertTrue("Runs with cache should be faster than runs without cache", averageWithCache < averageWithoutCache);
	}

}
