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

package org.apache.sysds.test.component.frame.array;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;

import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.ArrayWrapper;
import org.junit.Test;

public class ArrayWrapperTest {

	@Test
	public void testSerializationWrapper() throws Exception {
		Array<?> a = ArrayFactory.create(new double[]{1.0, 2.0, 3.0});
		ArrayWrapper wrap = new ArrayWrapper(null);
		wrap._a = a; // assign the a array to the wrapper.
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream fos = new DataOutputStream(bos);
		wrap.write(fos); // written out a.
		wrap._a = null; // reset wrapper.
		DataInputStream fis = new DataInputStream(new ByteArrayInputStream(bos.toByteArray()));
		wrap.readFields(fis);
		FrameArrayTests.compare(a, wrap._a);
	}

}
