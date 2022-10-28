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

package org.apache.sysds.test.functions.countDistinct;

import org.apache.sysds.common.Types.ExecType;
import org.junit.Test;

public class CountDistinctRowColParameterized extends CountDistinctRowColBase {

	public String TEST_NAME = "countDistinctRowColParameterized";
	public String TEST_DIR = "functions/countDistinct/";
	public String TEST_CLASS_DIR = TEST_DIR + CountDistinctRowColParameterized.class.getSimpleName() + "/";

	protected String getTestClassDir() {
		return TEST_CLASS_DIR;
	}

	protected String getTestName() {
		return TEST_NAME;
	}

	protected String getTestDir() {
		return TEST_DIR;
	}

	@Override
	public void setUp() {
		super.addTestConfiguration();
		super.percentTolerance = 0.0;
	}

	@Test
	public void testSimple1by1() {
		// test simple 1 by 1.
		ExecType ex = ExecType.CP;
		countDistinctScalarTest(1, 1, 1, 1.0, ex, 0.00001);
	}
}
