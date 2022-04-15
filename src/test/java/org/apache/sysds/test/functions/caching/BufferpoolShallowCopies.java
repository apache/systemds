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
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class BufferpoolShallowCopies extends AutomatedTestBase 
{
	private final static String TEST_NAME = "BufferpoolShallow";
	private final static String TEST_DIR = "functions/caching/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BufferpoolShallowCopies.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "V" }) ); 
	}
	
	@Test
	public void testShallowM() {
		runTestBufferpoolShallow(2000, 1500, "M");
	}
	
	//TODO implement shallow copy for frame replace
//	@Test
//	public void testShallowF() {
//		runTestBufferpoolShallow(2000, 1500, "F");
//	}
	
	private void runTestBufferpoolShallow(int rows, int cols, String type) {
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-stats", "-args",
			Integer.toString(rows), Integer.toString(cols), type};
		
		//run test and check no buffer pool writes (lineage and same object)
		runTest(true, false, null, -1);
		Assert.assertEquals(0, CacheStatistics.getFSBuffWrites());
	}
}
