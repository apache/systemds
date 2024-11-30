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

package org.apache.sysds.test.component.parfor;

import java.net.SocketException;
import java.net.UnknownHostException;

import org.apache.sysds.runtime.controlprogram.parfor.util.IDHandler;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.junit.Assert;
import org.junit.Test;

public class ParForIDHandling {

	@Test
	public void testExtractIntID() {
		Assert.assertEquals(2000009, IDHandler.extractIntID("task_local_0002_m_000009"));
		Assert.assertEquals(898000001, IDHandler.extractIntID("task_201203111647_0898_m_000001"));
	}
	
	@Test
	public void testDistributedUniqueID() {
		Assert.assertTrue(IDHandler.createDistributedUniqueID().contains("_"));
	}
	
	@Test
	public void testProcessID() {
		Assert.assertTrue(IDHandler.getProcessID() < Long.MAX_VALUE);
	}
	
	@Test
	public void testIntConcatenation() {
		long tmp = IDHandler.concatIntIDsToLong(3, 7);
		Assert.assertEquals(3, IDHandler.extractIntIDFromLong(tmp, 1));
		Assert.assertEquals(7, IDHandler.extractIntIDFromLong(tmp, 2));
		Assert.assertEquals(-1, IDHandler.extractIntIDFromLong(tmp, 3));
	}
	
	@Test
	public void testIPAddress() throws SocketException, UnknownHostException {
		Assert.assertNotEquals(null, IDHandler.getIPAddress(false));
		Assert.assertNotEquals(null, IDHandler.getIPAddress(true));
	}
	
	@Test
	public void testCyclicIDSequence() {
		testIDSequence(true);
	}
	
	@Test
	public void testNonCyclicIDSequence() {
		testIDSequence(false);
	}
	
	public void testIDSequence(boolean cyclic) {
		IDSequence seq = new IDSequence(cyclic, 2);
		Assert.assertEquals(-1, seq.getCurrentID());
		Assert.assertEquals(0, seq.getNextID());
		Assert.assertEquals(1, seq.getNextID());
		try {
			Assert.assertEquals(2, seq.getNextID());
			Assert.assertEquals(0, seq.getNextID());
			if( !cyclic ) // should have raised exception
				Assert.fail();
		}
		catch(Exception ex) {
			Assert.assertEquals(2, seq.getCurrentID());
		}
	}
}
