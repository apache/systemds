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

package org.apache.sysds.test.component.paramserv;

import java.io.IOException;
import java.util.Arrays;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.paramserv.rpc.PSRpcCall;
import org.apache.sysds.runtime.controlprogram.paramserv.rpc.PSRpcObject;
import org.apache.sysds.runtime.controlprogram.paramserv.rpc.PSRpcResponse;
import org.apache.sysds.runtime.instructions.cp.ListObject;

public class RpcObjectTest {

	private static ListObject generateData() {
		MatrixObject mo1 = SerializationTest.generateDummyMatrix(10);
		MatrixObject mo2 = SerializationTest.generateDummyMatrix(20);
		return new ListObject(Arrays.asList(mo1, mo2));
	}

	@Test
	public void testPSRpcCall() throws IOException {
		PSRpcCall expected = new PSRpcCall(PSRpcObject.PUSH, 1, generateData());
		PSRpcCall actual = new PSRpcCall(expected.serialize());
		Assert.assertTrue(Arrays.equals(
			new PSRpcCall(PSRpcObject.PUSH, 1, generateData()).serialize().array(),
			actual.serialize().array()));
	}

	@Test
	public void testPSRpcResponse() throws IOException {
		PSRpcResponse expected = new PSRpcResponse(PSRpcResponse.Type.SUCCESS, generateData());
		PSRpcResponse actual = new PSRpcResponse(expected.serialize());
		Assert.assertTrue(Arrays.equals(
			new PSRpcResponse(PSRpcResponse.Type.SUCCESS, generateData()).serialize().array(),
			actual.serialize().array()));
	}
}
