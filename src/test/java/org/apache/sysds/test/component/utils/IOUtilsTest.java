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

package org.apache.sysds.test.component.utils;

import static org.junit.Assert.assertEquals;

import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.junit.Test;

public class IOUtilsTest {

	@Test
	public void getTo() {
		String in = ",\"yyy\"·4,";
		assertEquals(0, getTo(in, 0, ","));
		assertEquals(8, getTo(in, 1, ","));
		assertEquals("\"yyy\"·4", in.substring(1, getTo(in, 1, ",")));
	}

	@Test
	public void getTo2() {
		String in = ",y,";
		assertEquals(0, getTo(in, 0, ","));
		assertEquals(2, getTo(in, 1, ","));
	}

	@Test
	public void getTo3() {
		String in = "a,b,c";
		assertEquals("a", in.substring(0, getTo(in, 0, ",")));
		assertEquals("b", in.substring(2, getTo(in, 2, ",")));
		assertEquals("c", in.substring(4, getTo(in, 4, ",")));
	}

	@Test
	public void getTo4() {
		String in = "a,\",\",c";
		assertEquals("a", in.substring(0, getTo(in, 0, ",")));
		assertEquals("\",\"", in.substring(2, getTo(in, 2, ",")));
	}

	private int getTo(String in, int from, String delim) {
		return IOUtilFunctions.getTo(in, from, ",", in.length(), 1);
	}
}
