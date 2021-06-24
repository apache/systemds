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

package org.apache.sysds.test.applications;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.AutomatedTestBase;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class GANTest extends AutomatedTestBase
{
	protected final static String TEST_DIR = "applications/GAN/";
	protected final static String TEST_NAME = "GAN_mnist";
	protected String TEST_CLASS_DIR = TEST_DIR + GANTest.class.getSimpleName() + "/";

	protected String _type;

	public GANTest(String type) {
		_type = type;
	}

	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] { 
			// {"cnn"}, //TODO investigate what's taking so long, and why are there spark instructions in hybrid?
			{"simple"}};
		return Arrays.asList(data);
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"accuracy.scalar"}));
	}

	@Test
	public void testGAN() {
		System.out.println("Running GAN test");
		getAndLoadTestConfiguration(TEST_NAME);
		ExecMode modeOld = setExecMode(ExecMode.SINGLE_NODE);
		try {
			fullDMLScriptName = getScript();
			List<String> proArgs = new ArrayList<>();
			proArgs.add("-args");
			proArgs.add(_type);
			proArgs.add(output(""));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
	
			writeExpectedScalar("accuracy", 0.95); //0.5 w/ 5000 instead of 1000
			compareResults(0.15);
		}
		finally {
			resetExecMode(modeOld);
		}
	}
}
