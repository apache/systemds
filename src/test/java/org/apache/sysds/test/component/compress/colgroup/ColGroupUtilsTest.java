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

package org.apache.sysds.test.component.compress.colgroup;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils;
import org.junit.Test;

public class ColGroupUtilsTest {

	@Test
	public void containsNan_false() {
		assertFalse(ColGroupUtils.containsInfOrNan(Double.NaN, new double[10]));
	}

	@Test
	public void containsInf_false() {
		assertFalse(ColGroupUtils.containsInfOrNan(Double.POSITIVE_INFINITY, new double[10]));
	}

	@Test
	public void containsInf_True() {
		assertTrue(ColGroupUtils.containsInfOrNan(Double.POSITIVE_INFINITY,
			new double[] {0, 0, 0, 0, Double.POSITIVE_INFINITY, 0, 0, 0}));
	}

	@Test
	public void containsNan_True() {
		assertTrue(ColGroupUtils.containsInfOrNan(Double.NaN, new double[] {0, 0, 0, 0, Double.NaN, 0, 0, 0}));
	}

}
