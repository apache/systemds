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

package org.apache.sysml.test.integration.functions.paramserv;

import java.io.IOException;
import java.util.Arrays;

import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.paramserv.spark.rpc.PSRpcCall;
import org.apache.sysml.runtime.controlprogram.paramserv.spark.rpc.PSRpcObject;
import org.apache.sysml.runtime.controlprogram.paramserv.spark.rpc.PSRpcResponse;
import org.apache.sysml.runtime.instructions.cp.ListObject;
import org.junit.Assert;
import org.junit.Test;

public class RpcObjectTest {

	@Test
	public void testPSRpcCall() throws IOException {
		MatrixObject mo1 = SerializationTest.generateDummyMatrix(10);
		MatrixObject mo2 = SerializationTest.generateDummyMatrix(20);
		ListObject lo = new ListObject(Arrays.asList(mo1, mo2));
		PSRpcCall expected = new PSRpcCall(PSRpcObject.PUSH, 1, lo);
		PSRpcCall actual = new PSRpcCall(expected.serialize());
		Assert.assertEquals(new String(expected.serialize().array()), new String(actual.serialize().array()));
	}

	@Test
	public void testPSRpcResponse() throws IOException {
		MatrixObject mo1 = SerializationTest.generateDummyMatrix(10);
		MatrixObject mo2 = SerializationTest.generateDummyMatrix(20);
		ListObject lo = new ListObject(Arrays.asList(mo1, mo2));
		PSRpcResponse expected = new PSRpcResponse(PSRpcResponse.SUCCESS, lo);
		PSRpcResponse actual = new PSRpcResponse(expected.serialize());
		Assert.assertEquals(new String(expected.serialize().array()), new String(actual.serialize().array()));
	}
}
