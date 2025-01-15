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

package org.apache.sysds.test.component.frame.array;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.List;
import java.util.Map;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.spi.LoggingEvent;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.test.LoggingUtils;
import org.apache.sysds.test.LoggingUtils.TestAppender;
import org.junit.Test;

public class RecodeMapTest {

	@Test
	public void createRecodeMapLoggingDebug() throws Exception {
		final TestAppender appender = LoggingUtils.overwrite();
		try {
			Logger.getLogger(Array.class).setLevel(Level.DEBUG);
			Array<String> a = ArrayFactory.create(FrameArrayTests.generateRandomStringNUnique(100, 324, 10));

			Map<String, Integer> rcm = a.getRecodeMap(10);
			assertTrue(rcm.size() == 10);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			assertTrue(log.size() >= 1);
		}
		finally {
			LoggingUtils.reinsert(appender);
		}

	}

	@Test
	public void createRecodeMapParallel() throws Exception {
		final TestAppender appender = LoggingUtils.overwrite();
		int tmp = Array.ROW_PARALLELIZATION_THRESHOLD;
		try {
			Array.ROW_PARALLELIZATION_THRESHOLD = 10;
			Logger.getLogger(Array.class).setLevel(Level.DEBUG);
			Array<String> a = ArrayFactory.create(FrameArrayTests.generateRandomStringNUnique(1000, 324, 10));

			Map<String, Integer> rcm = a.getRecodeMap(10, CommonThreadPool.get(10), 10);
			assertTrue(rcm.size() == 10);
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			assertTrue(log.size() >= 1);
		}
		finally {
			LoggingUtils.reinsert(appender);
			Array.ROW_PARALLELIZATION_THRESHOLD = tmp;
		}

	}

	@Test
	public void createRecodeMapParallel2() throws Exception {
		final TestAppender appender = LoggingUtils.overwrite();
		int tmp = Array.ROW_PARALLELIZATION_THRESHOLD;
		try {
			Array.ROW_PARALLELIZATION_THRESHOLD = 10;
			Logger.getLogger(Array.class).setLevel(Level.DEBUG);
			Array<String> a = ArrayFactory.create(FrameArrayTests.generateRandomStringNUnique(1000, 324, 500));

			Map<String, Integer> rcm = a.getRecodeMap(10, CommonThreadPool.get(10), 10);
			Map<String, Integer> rcm2 = a.getRecodeMap(10, null, -1);
			assertTrue(Math.abs(rcm.size() -  500)  < 100);

			assertTrue(rcm.size() == rcm2.size());

			rcm.forEach((k,v) ->{
				assertEquals(rcm.get(k), rcm2.get(k));
			});
			final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
			assertTrue(log.size() >= 1);
		}
		finally {
			LoggingUtils.reinsert(appender);
			Array.ROW_PARALLELIZATION_THRESHOLD = tmp;
		}

	}
}
