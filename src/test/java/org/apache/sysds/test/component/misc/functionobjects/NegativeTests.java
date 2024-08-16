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

package org.apache.sysds.test.component.misc.functionobjects;

import static org.junit.Assert.assertNull;

import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.junit.Test;

public class NegativeTests {

	@Test(expected = Exception.class)
	public void t1() {
		Builtin.getBuiltinFnObject(BuiltinCode.ISNAN).execute(0, 1);
	}

	@Test(expected = Exception.class)
	public void t2() {
		Builtin.getBuiltinFnObject(BuiltinCode.ISNAN).execute(0.0, 1.0);
	}

	@Test(expected = Exception.class)
	public void t3() {
		Builtin.getBuiltinFnObject(BuiltinCode.ISNAN).execute("something something...");
	}

	@Test(expected = Exception.class)
	public void t4() {
		Builtin.getBuiltinFnObject(BuiltinCode.MAXINDEX).execute(1);
	}

	@Test(expected = Exception.class)
	public void t5() {
		Builtin.getBuiltinFnObject(BuiltinCode.MAXINDEX).execute(1.0);
	}

	@Test
	public void t6() {
		assertNull(Builtin.getBuiltinFnObject((BuiltinCode) null));
	}

	@Test(expected = Exception.class)
	public void t7() {
		Builtin.getBuiltinFnObject(BuiltinCode.STOP).execute("Something");
	}

}
