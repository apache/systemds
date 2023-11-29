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

package org.apache.sysds.test.component.matrix;

import static org.junit.Assert.assertEquals;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.junit.Test;

public class SparseFactory {
	protected static final Log LOG = LogFactory.getLog(SparseFactory.class.getName());

	@Test
	public void testCreateFromArray() {
		double[] dense = new double[] {0, 0, 0, 1, 1, 1, 0, 0, 0};
		SparseBlock sb = SparseBlockFactory.createFromArray(dense, 3, 3);
		
		assertEquals(0, sb.get(0, 0), 0.0);
		assertEquals(0, sb.get(1, 1), 1.0);
		assertEquals(0, sb.get(2, 2), 0.0);
	}
}
