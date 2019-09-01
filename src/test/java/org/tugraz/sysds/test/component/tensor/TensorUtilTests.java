/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.test.component.tensor;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.runtime.util.UtilFunctions;


public class TensorUtilTests {
	@Test
	public void testBlockNumber() {
		Assert.assertEquals(5, UtilFunctions.computeBlockNumber(new int[]{3, 2, 1}, new long[]{4000, 133, 1}, 128));
	}

	@Test
	public void testBlockNumberBegin() {
		Assert.assertEquals(0, UtilFunctions.computeBlockNumber(new int[]{1, 1, 1}, new long[]{4000, 133000, 9}, 128));
	}

	@Test
	public void testBlockNumberLast() {
		Assert.assertEquals(2 * 8 - 1, UtilFunctions.computeBlockNumber(new int[]{1, 2, 8}, new long[]{128, 128 + 1, 8 * 128}, 128));
	}
}
