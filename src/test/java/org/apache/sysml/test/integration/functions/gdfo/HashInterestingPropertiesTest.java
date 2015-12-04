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

package org.apache.sysml.test.integration.functions.gdfo;


import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.hops.globalopt.InterestingProperties;
import org.apache.sysml.hops.globalopt.InterestingProperties.Format;
import org.apache.sysml.hops.globalopt.InterestingProperties.Location;
import org.apache.sysml.hops.globalopt.InterestingProperties.Partitioning;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;

public class HashInterestingPropertiesTest extends AutomatedTestBase
{
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testCopyEquivalence()
	{
		InterestingProperties ips1 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.MEM, Partitioning.NONE, 1, true);
		InterestingProperties ips2 = new InterestingProperties(ips1);
		InterestingProperties ips3 = new InterestingProperties(ips2);
		
		//test hash functions
		Assert.assertEquals(ips1.hashCode(), ips2.hashCode());
		Assert.assertEquals(ips2.hashCode(), ips3.hashCode());
		
		//test equals (used inside hash maps)
		Assert.assertTrue(ips1.equals(ips2));
		Assert.assertTrue(ips2.equals(ips3));
	}
	
	@Test
	public void testValueEquivalence()
	{
		InterestingProperties ips1 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.MEM, Partitioning.NONE, 1, true);
		InterestingProperties ips2 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.MEM, Partitioning.NONE, 1, true);
		InterestingProperties ips3 = new InterestingProperties(2000, Format.BINARY_BLOCK, Location.MEM, Partitioning.NONE, 1, true);
		InterestingProperties ips4 = new InterestingProperties(1000, Format.TEXT_CELL, Location.MEM, Partitioning.NONE, 1, true);
		InterestingProperties ips5 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.HDFS, Partitioning.NONE, 1, true);
		InterestingProperties ips6 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.MEM, Partitioning.COL_WISE, 1, true);
		InterestingProperties ips7 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.MEM, Partitioning.NONE, 1, false);
		
		//test hash functions
		Assert.assertEquals(ips1.hashCode(), ips2.hashCode());
		
		//test equals (used inside hash maps)
		Assert.assertTrue(ips1.equals(ips2));
		Assert.assertFalse(ips2.equals(ips3));
		Assert.assertFalse(ips2.equals(ips4));
		Assert.assertFalse(ips2.equals(ips5));
		Assert.assertFalse(ips2.equals(ips6));
		Assert.assertFalse(ips2.equals(ips7));
	}
	
	@Test
	public void testValueEquivalenceWithNull()
	{
		InterestingProperties ips1 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.MEM, Partitioning.NONE, 1, true);
		InterestingProperties ips2 = new InterestingProperties(1000, null, Location.MEM, Partitioning.NONE, 1, true);
		InterestingProperties ips3 = new InterestingProperties(1000, Format.BINARY_BLOCK, null, Partitioning.NONE, 1, true);
		InterestingProperties ips4 = new InterestingProperties(1000, Format.BINARY_BLOCK, Location.MEM, null, 1, true);
		InterestingProperties ips5 = new InterestingProperties(1000, Format.ANY, Location.MEM, Partitioning.NONE, 1, true);
		
		//test equals (used inside hash maps)
		Assert.assertFalse(ips1.equals(ips2));
		Assert.assertFalse(ips1.equals(ips3));
		Assert.assertFalse(ips1.equals(ips4));
		Assert.assertFalse(ips1.equals(ips5));
	}
}
