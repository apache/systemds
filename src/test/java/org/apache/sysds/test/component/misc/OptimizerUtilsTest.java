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

package org.apache.sysds.test.component.misc;

import static org.junit.Assert.assertTrue;

import org.apache.sysds.hops.OptimizerUtils;
import org.junit.Test;

public class OptimizerUtilsTest {

	@Test
	public void estimateFrameSize() {
		Long size = OptimizerUtils.estimateSizeExactFrame(10, 10);
		assertTrue(size > 10 * 10);
	}

	@Test
	public void estimateFrameSizeMoreRowsThanInt() {
		// Currently we do not support frames larger than INT. Therefore we estimate their size to be extremely large.
		// The large size force spark operations
		Long size = OptimizerUtils.estimateSizeExactFrame(Integer.MAX_VALUE + 1L, 10);

		assertTrue(size == Long.MAX_VALUE);
	}
}
