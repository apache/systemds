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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.*;
import org.apache.sysds.test.TestUtils;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import javax.sound.midi.Soundbank;
import java.util.*;
import java.util.stream.Collectors;

import static org.junit.Assert.*;

/*
this class is intended to test the execution time of the encoding process under different threading scenarios
*/
@RunWith(value = Parameterized.class)
public class TransformEncodeCacheMultiThreadTest {
	protected static final Log LOG = LogFactory.getLog(TransformEncodeCacheMultiThreadTest.class.getName());

	private final FrameBlock _data;
	private final int _k;
	private final List<String> _specs;
	private final EncoderType _encoderType;
	protected static LinkedList<EncodeCacheKey> _evicQueue = null;
	protected static Map<EncodeCacheKey, EncodeCacheEntry> _cacheMap = null;

	public TransformEncodeCacheMultiThreadTest(FrameBlock _data, int _k, List<String> _specs, EncoderType _encoderType) {
		this._data = _data;
		this._k = _k;
		this._specs = _specs;
		this._encoderType = _encoderType;
	}

	@BeforeClass
	public static void setUp() {
		EncodeCacheConfig.useCache(true);
		try {
			long st = System.nanoTime();
			EncodeBuildCache.getEncodeBuildCache();
			long et = System.nanoTime();
			double setUpTime = (et - st)/1_000_000.0;

			_evicQueue = EncodeBuildCache.get_evictionQueue();
			System.out.println((String.format("Successfully set up cache in %f milliseconds. " +
					"Size of eviction queue: %d", setUpTime, _evicQueue.size())));

			_cacheMap = EncodeBuildCache.get_cache();

			System.out.println((String.format("Cache limit: %d", EncodeBuildCache.get_cacheLimit())));

		} catch(DMLRuntimeException e){
			LOG.error("Creation of cache failed:" + e.getMessage());
		}
		EncodeBuildCache.clear();
	}

	@Parameters
	public static Collection<Object[]> testParameters() {
		final ArrayList<Object[]> tests = new ArrayList<>();

		final int[] threads = new int[] {1, 2, 4, 8};
		//final int[] threads = new int[] {2, 4, 8};

		int numColumns = 60;
		int numRows = 10000;
		FrameBlock testData = EncodeCacheTestUtil.generateTestData(numColumns, numRows);

		//create a list of recode specs referring to one distinct column each
		List<String> recodeSpecs = EncodeCacheTestUtil.generateRecodeSpecs(numColumns);

		//create a list of bin specs referring to one distinct column each
		List<String> binSpecs = EncodeCacheTestUtil.generateBinSpecs(numColumns);

		List<List<String>> specLists = Arrays.asList(recodeSpecs, binSpecs);
		List<EncoderType> encoderTypes = Arrays.asList(EncoderType.Recode, EncoderType.Bin);

		//create test cases for each combination of recoder type and thread number:
		//2x4 tests running on numColumn distinct encoders
		for (int index = 0; index < specLists.size(); index++){
			for(int k : threads)
				tests.add(new Object[]{testData, k, specLists.get(index), encoderTypes.get(index)});
		}
		return tests;
	}

	@Test
	public void compareCachePerformanceSingleAndMultithreaded(){

		List<Long> durationsWithout = new ArrayList<>();
		List<Long> durationsWith = new ArrayList<>();
		try {
			FrameBlock meta = null;
			List<MultiColumnEncoder> encoders = new ArrayList<>();
			for (String spec: _specs) {
				//create an encoder for each column
				encoders.add(EncoderFactory.createEncoder(spec, _data.getColumnNames(), _data.getNumColumns(), meta));
			}

			// first run, no cache entries present yet
			for (MultiColumnEncoder encoder : encoders) {
				durationsWithout.add(EncodeCacheTestUtil.measureEncodeTime(encoder, _data, _k));
			}

			// second run, there is a cache entry for every spec now
			for (MultiColumnEncoder encoder : encoders) {
				durationsWith.add(EncodeCacheTestUtil.measureEncodeTime(encoder, _data, _k));
			}

			//exclude a large number of runs to make sure not to average over any initialization procedures
			int numExclusions = 10;
			LOG.debug(String.format("Number of runs to exclude: %d%n", numExclusions));
			double avgExecTimeWithout = EncodeCacheTestUtil.analyzePerformance(numExclusions, durationsWithout, false);
			LOG.debug(String.format("Average exec time for %s with %d threads without cache: %f%n", _encoderType, _k, avgExecTimeWithout));

			double avgExecTimeWith = EncodeCacheTestUtil.analyzePerformance(numExclusions, durationsWith, true);
			LOG.debug(String.format("Average exec time for %s with %d threads with cache: %f%n", _encoderType, _k, avgExecTimeWith));

		} catch (Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}