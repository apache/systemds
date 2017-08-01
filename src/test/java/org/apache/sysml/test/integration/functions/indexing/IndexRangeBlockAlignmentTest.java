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

package org.apache.sysml.test.integration.functions.indexing;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.test.integration.AutomatedTestBase;


public class IndexRangeBlockAlignmentTest extends AutomatedTestBase
{
	private static final int BRLEN = 1000;
	private static final int BCLEN = 1000;
	
	@Override
	public void setUp() {
		
	}
	
	@Test
	public void testRowBlockFirstColumn() {
		Assert.assertEquals(Boolean.TRUE,
				OptimizerUtils.isIndexingRangeBlockAligned(2001, 4000, 1, 1736, BRLEN, BCLEN));
	}
	
	@Test
	public void testRowBlockColBlock() {
		Assert.assertEquals(Boolean.TRUE,
				OptimizerUtils.isIndexingRangeBlockAligned(2001, 4000, 7001, 9000, BRLEN, BCLEN));
	}

	@Test
	public void testSingleRowBlockFirstColumn() {
		Assert.assertEquals(Boolean.TRUE,
				OptimizerUtils.isIndexingRangeBlockAligned(2500, 2600, 1, 1736, BRLEN, BCLEN));
	}
	
	@Test
	public void testSingleRowBlockColBlock() {
		Assert.assertEquals(Boolean.TRUE,
				OptimizerUtils.isIndexingRangeBlockAligned(2500, 2600, 7001, 9000, BRLEN, BCLEN));
	}
	
	@Test
	public void testRowBlockFirstColumnNeg() {
		Assert.assertEquals(Boolean.FALSE,
				OptimizerUtils.isIndexingRangeBlockAligned(2501, 4500, 1, 1736, BRLEN, BCLEN));
	}
	
	@Test
	public void testRowBlockColBlockNeg() {
		Assert.assertEquals(Boolean.FALSE,
				OptimizerUtils.isIndexingRangeBlockAligned(2501, 4500, 7001, 9000, BRLEN, BCLEN));
	}

	@Test
	public void testSingleRowBlockFirstColumnNeg() {
		Assert.assertEquals(Boolean.FALSE,
				OptimizerUtils.isIndexingRangeBlockAligned(2500, 3001, 1, 1736, BRLEN, BCLEN));
	}
	
	@Test
	public void testSingleRowBlockColBlockNeg() {
		Assert.assertEquals(Boolean.FALSE,
				OptimizerUtils.isIndexingRangeBlockAligned(2500, 3001, 7001, 9000, BRLEN, BCLEN));
	}
}
