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

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.junit.Test;

public class OffsetSingleTests {

	@Test
	public void testEmptyEstimateMemory() {
		assertTrue(OffsetFactory.estimateInMemorySize(0, 10000) < 10);
	}


	@Test(expected = DMLCompressionException.class)
	public void testInvalidCreate(){
		OffsetFactory.createOffset(null, 10, 0);
	}


	@Test
	public void equivalentToOtherConstructor(){
		int[] offset = new int[]{1,2,3,4,5,8};
		AOffset a = OffsetFactory.createOffset(offset);
		IntArrayList offsetB = new IntArrayList(offset);
		AOffset b = OffsetFactory.createOffset(offsetB);
		assertTrue(a.equals(b));
	}

	@Test
	public void notEquivalentOnLast(){
		int[] offset = new int[]{1,2,3,4,5,8};
		AOffset a = OffsetFactory.createOffset(offset);
		IntArrayList offsetB = new IntArrayList(new int[]{1,9});
		AOffset b = OffsetFactory.createOffset(offsetB);
		assertFalse(a.equals(b));
	}

	@Test
	public void notEquivalentOnFirst(){
		int[] offset = new int[]{1,2,3,4,5,8};
		AOffset a = OffsetFactory.createOffset(offset);
		IntArrayList offsetB = new IntArrayList(new int[]{0,8});
		AOffset b = OffsetFactory.createOffset(offsetB);
		assertFalse(a.equals(b));
	}

	@Test
	public void notEquivalentInside(){
		int[] offset = new int[]{1,2,3,4,5,8};
		AOffset a = OffsetFactory.createOffset(offset);
		IntArrayList offsetB = new IntArrayList(new int[]{1,2,3,4,8});
		AOffset b = OffsetFactory.createOffset(offsetB);
		assertFalse(a.equals(b));
	}
}
