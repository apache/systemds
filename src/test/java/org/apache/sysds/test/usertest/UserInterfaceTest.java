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

package org.apache.sysds.test.usertest;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.commons.lang3.tuple.Pair;
import org.junit.Test;

public class UserInterfaceTest extends Base {

	@Test
	public void testHelloWorld(){
		Pair<String,String> res = runThread("helloWorld.dml");
		assertEquals("", res.getRight());
		assertTrue(res.getLeft().contains("Hello, World!"));
	}

	@Test
	public void testStop(){
		Pair<String,String> res = runThread("Stop.dml");
		assertEquals("",res.getRight());
		assertTrue(res.getLeft().contains("An Error Occured :"));
		assertTrue(res.getLeft().contains("DMLScriptException -- Stop Message!"));
	}

	@Test
	public void SyntaxError(){
		Pair<String,String> res = runThread("SyntaxError.dml");
		assertEquals("",res.getRight());
		assertTrue(res.getLeft().contains("An Error Occured :"));
		assertTrue(res.getLeft().contains("[Syntax error]"));
		assertTrue(res.getLeft().contains("ParseException --"));
	}
}
