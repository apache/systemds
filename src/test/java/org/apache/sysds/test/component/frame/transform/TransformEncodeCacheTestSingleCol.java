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
import org.apache.sysds.runtime.transform.encode.EncodeBuildCache;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderType;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.TestUtils;
import org.junit.BeforeClass;
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
public class TransformEncodeCacheTestSingleCol {
	protected static final Log LOG = LogFactory.getLog(TransformEncodeCacheTestSingleCol.class.getName());

	private final FrameBlock data;
	private final int k;
	private final List<String> specs;
	private final EncoderType encoderType;

	public TransformEncodeCacheTestSingleCol(FrameBlock data, int k, List<String> specs, EncoderType encoderType) {
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
			long duration = EncodeCacheTestUtil.measureEncodeTime(encoder, setUpData, 1);
			LOG.info("Setup took " + duration/1_000_000.0 + " milliseconds");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Parameters
	public static Collection<Object[]> data() {
		final ArrayList<Object[]> tests = new ArrayList<>();
		final int k = 1;

		int numColumns = 10;
		int numRows = 10;
		FrameBlock testData = EncodeCacheTestUtil.generateTestData(numColumns, numRows);

		List<List<String>> specLists = Arrays.asList(
				EncodeCacheTestUtil.generateRecodeSpecs(numColumns),
				EncodeCacheTestUtil.generateBinSpecs(numColumns)
		);
		List<EncoderType> encoderTypes = Arrays.asList(EncoderType.Recode, EncoderType.Bin);

		for (int index = 0; index < specLists.size(); index++){
			tests.add(new Object[]{testData, k, specLists.get(index), encoderTypes.get(index)});
		}
		return tests;
	}

	@Test
	public void test() {
		List<Long> durations = new ArrayList<>();
		try {
			FrameBlock meta = null;
			List<MultiColumnEncoder> encoders = new ArrayList<>();
			for (String spec: specs) {
				encoders.add(EncoderFactory.createEncoder(spec, data.getColumnNames(), data.getNumColumns(), meta));
			}
			// encode twice with each encoder
			for (int i = 0; i < 2; i ++){
				for (MultiColumnEncoder encoder: encoders) {
					durations.add(EncodeCacheTestUtil.measureEncodeTime(encoder, data, k));
				}
			}
			analyzePerformance(durations);

		} catch (Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private void analyzePerformance(List<Long> durations) {

		long firstRun = durations.get(0);
		// get the first half of the list which contains the runs without any cache entry
		int halfListSize = durations.size()/2;
		List<Long> runsWithoutCache = durations.subList(0, halfListSize);
		for (long duration: runsWithoutCache){
			LOG.info("duration without cache: " + duration/1_000_000.0);
		}
		List<Long> runsWithCache = durations.subList(halfListSize, durations.size());
		for (long duration: runsWithCache){
			LOG.info("duration with cache: " + duration/1_000_000.0);
		}

		double averageWithoutCache = runsWithoutCache.stream().mapToLong(Long::longValue).average().orElse(Double.NaN);
		double averageWithCache = runsWithCache.stream().mapToLong(Long::longValue).average().orElse(Double.NaN);

		DecimalFormat df = new DecimalFormat("#.#####");
		LOG.info("Runtime for " + encoderType + ":");
		LOG.info("First run: " + df.format(firstRun/1_000_000.0) + " milliseconds");
		LOG.info("Average time without cache: " + df.format(averageWithoutCache/1_000_000.0) + " milliseconds");
		LOG.info("Average time with cache: " + df.format(averageWithCache/1_000_000.0) + " milliseconds");

		assertTrue("Runs with cache should be faster than runs without cache", averageWithCache < averageWithoutCache);
	}

}
