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

package org.apache.sysds.test.component.compress.offset;

import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.junit.Test;

public class OffsetSingleTests {

	@Test(expected = RuntimeException.class)
	public void testInvalidSize_01() {
		OffsetFactory.estimateInMemorySize(-1, 100);
	}

	@Test(expected = RuntimeException.class)
	public void testInvalidSize_02() {
		OffsetFactory.estimateInMemorySize(10, -1);
	}

	@Test(expected = RuntimeException.class)
	public void testInvalidCreation() {
		OffsetFactory.create(new int[] {1, 2, 3, -1});
	}
}
