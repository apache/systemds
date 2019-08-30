/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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
 
package org.tugraz.sysds.test.functions.codegenalg;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.tugraz.sysds.test.applications.ArimaTest;

import java.io.File;

@RunWith(value = Parameterized.class)
public class AlgorithmARIMA extends ArimaTest 
{
	private final static String LOCAL_TEST_DIR = "functions/codegenalg/";
	private final static String TEST_CONF_DEFAULT = "SystemDS-config-codegen.xml";
	private final static File TEST_CONF_FILE_DEFAULT = new File(SCRIPT_DIR + LOCAL_TEST_DIR, TEST_CONF_DEFAULT);
	private final static String TEST_CONF_FUSE_ALL = "SystemDS-config-codegen-fuse-all.xml";
	private final static File TEST_CONF_FILE_FUSE_ALL = new File(SCRIPT_DIR + LOCAL_TEST_DIR, TEST_CONF_FUSE_ALL);
	private final static String TEST_CONF_FUSE_NO_REDUNDANCY = "SystemDS-config-codegen-fuse-no-redundancy.xml";
	private final static File TEST_CONF_FILE_FUSE_NO_REDUNDANCY = new File(SCRIPT_DIR + LOCAL_TEST_DIR,
			TEST_CONF_FUSE_NO_REDUNDANCY);

	private enum TestType { DEFAULT,FUSE_ALL,FUSE_NO_REDUNDANCY }
	private TestType currentTestType = TestType.DEFAULT;

	public AlgorithmARIMA(int m, int p, int d, int q, int P, int D, int Q, int s, int include_mean, int useJacobi) {
		super(m, p, d, q, P, D, Q, s, include_mean, useJacobi);
		TEST_CLASS_DIR = TEST_DIR + AlgorithmARIMA.class.getSimpleName() + "/";
	}

	@Test
	public void testArimaDml() {
		testArima(TestType.DEFAULT);
	}

	@Test
	public void testArimaDmlFuseAll() {
		testArima(TestType.FUSE_ALL);
	}

	@Test
	public void testArimaDmlFuseNoRedundancy() {
		testArima(TestType.FUSE_NO_REDUNDANCY);
	}

	private void testArima(TestType testType){
		currentTestType = testType;
		testArima();
	}
	
	@Override
	protected File getConfigTemplateFile() {
		String message = "This test case overrides default configuration with ";
		if(currentTestType == TestType.FUSE_ALL){
			System.out.println(message + TEST_CONF_FILE_FUSE_ALL.getPath());
			return TEST_CONF_FILE_FUSE_ALL;
		} else if(currentTestType == TestType.FUSE_NO_REDUNDANCY){
			System.out.println(message + TEST_CONF_FILE_FUSE_NO_REDUNDANCY.getPath());
			return TEST_CONF_FILE_FUSE_NO_REDUNDANCY;
		} else {
			System.out.println(message + TEST_CONF_FILE_DEFAULT.getPath());
			return TEST_CONF_FILE_DEFAULT;
		}
	}
}
