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

package org.apache.sysds.test.functions.codegenalg.partone;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.apache.sysds.test.applications.MDABivariateStatsTest;

import java.io.File;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class AlgorithmMDABivar extends MDABivariateStatsTest 
{
	private final static String LOCAL_TEST_DIR = "functions/codegenalg/";

	private CodegenTestType currentTestType = CodegenTestType.DEFAULT;
	
	public AlgorithmMDABivar(int n, int m, int li, int lml) {
		super(n, m, li, lml);
		TEST_CLASS_DIR = TEST_DIR + AlgorithmMDABivar.class.getSimpleName() + "/";
	}
	
	@Test
	public void testMDABivariateStatsDml() {
		testMDABivariateStats(CodegenTestType.DEFAULT);
	}

	@Test
	public void testMDABivariateStatsDmlFuseAll() {
		testMDABivariateStats(CodegenTestType.FUSE_ALL);
	}

	@Test
	public void testMDABivariateStatsDmlFuseNoRedundancy() {
		testMDABivariateStats(CodegenTestType.FUSE_NO_REDUNDANCY);
	}

	private void testMDABivariateStats(CodegenTestType CodegenTestType) {
		currentTestType = CodegenTestType;
		testMDABivariateStats();
	}
	
	@Override
	protected File getConfigTemplateFile() {
		return getCodegenConfigFile(SCRIPT_DIR + LOCAL_TEST_DIR, currentTestType);
	}
}
