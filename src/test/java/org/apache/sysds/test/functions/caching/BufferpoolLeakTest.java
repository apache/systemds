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

package org.apache.sysds.test.functions.caching;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.controlprogram.caching.CacheStatistics;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.LazyWriteBuffer;
import org.apache.sysds.runtime.controlprogram.caching.LazyWriteBuffer.RPolicy;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class BufferpoolLeakTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "BufferpoolLeak";
	private final static String TEST_DIR = "functions/caching/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BufferpoolLeakTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "V" }) ); 
	}
	
	@Test
	public void testLeak1_FIFO() {
		runTestBufferpoolLeak(10000, 15, RPolicy.FIFO, false);
	}
	
	@Test
	public void testLeak1_LRU() {
		runTestBufferpoolLeak(10000, 15, RPolicy.LRU, false);
	}
	
	@Test
	public void testLeak1_FIFO_Async() {
		runTestBufferpoolLeak(10000, 15, RPolicy.FIFO, true);
	}
	
	private void runTestBufferpoolLeak(int rows, int cols, RPolicy policy, boolean asyncSerialize) {
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);
		CacheableData.CACHING_BUFFER_POLICY = policy;
		CacheableData.CACHING_ASYNC_SERIALIZE = asyncSerialize;
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-stats", "-args",
			Integer.toString(rows), Integer.toString(cols)};
		
		//run test and check no evictions
		runTest(true, false, null, -1);
		LazyWriteBuffer.printStatus("tests");
		Assert.assertEquals(0, CacheStatistics.getFSWrites());
	}
}
