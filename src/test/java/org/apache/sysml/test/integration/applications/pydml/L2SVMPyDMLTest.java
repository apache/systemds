/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.test.integration.applications.pydml;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import org.apache.sysml.test.integration.applications.L2SVMTest;

@RunWith(value = Parameterized.class)
public class L2SVMPyDMLTest extends L2SVMTest {

	public L2SVMPyDMLTest(int rows, int cols, double sp, boolean intercept) {
		super(rows, cols, sp, intercept);
		TEST_CLASS_DIR = TEST_DIR + L2SVMPyDMLTest.class.getSimpleName() + "/";
	}

	@Test
	public void testL2SVMPyDml() {
		testL2SVM(ScriptType.PYDML);
	}

}
