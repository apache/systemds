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

package org.apache.sysds.test.component.codegen;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;

public class CPlanModIntdivPrimitives 
{
	@Test
	public void testINT32Mod2() {
		Double val = LibSpoofPrimitives.mod(4, 2);
		Assert.assertEquals(val, new Double(0));
	}
	
	@Test
	public void testFP64Mod2() {
		Double val = LibSpoofPrimitives.mod(4.3, 2);
		Assert.assertEquals(val, new Double(0.3), 10-8);
	}
	
	@Test
	public void testINT32Intdiv2() {
		Double val = LibSpoofPrimitives.intDiv(4, 2);
		Assert.assertEquals(val, new Double(2));
	}
	
	@Test
	public void testFP64Intdiv2() {
		Double val = LibSpoofPrimitives.intDiv(4.3, 2);
		Assert.assertEquals(val, new Double(2));
	}
}
