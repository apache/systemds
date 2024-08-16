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

package org.apache.sysds.test.component.misc;

import java.io.IOException;
import java.net.URL;
import java.util.Enumeration;

import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.OpOp4;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.OpOpDnn;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.common.Types.ReOrgOp;
import org.junit.Assert;
import org.junit.Test;

public class OpTypeTest {

	@Test
	public void testOpOp1() {
		OpOp1[] ops = OpOp1.values();
		for(int i=0; i<ops.length; i++) {
			Assert.assertEquals(i, ops[i].ordinal());
			Assert.assertEquals(OpOp1.valueOf(ops[i].name()), ops[i]);
			Assert.assertEquals(OpOp1.valueOfByOpcode(ops[i].toString()), ops[i]);
		}
	}
	
	@Test
	public void testOpOp2() {
		OpOp2[] ops = OpOp2.values();
		for(int i=0; i<ops.length; i++) {
			Assert.assertEquals(i, ops[i].ordinal());
			Assert.assertEquals(OpOp2.valueOf(ops[i].name()), ops[i]);
			Assert.assertEquals(OpOp2.valueOfByOpcode(ops[i].toString()), ops[i]);
		}
	}
	
	@Test
	public void testOpOp3() {
		OpOp3[] ops = OpOp3.values();
		for(int i=0; i<ops.length; i++) {
			Assert.assertEquals(i, ops[i].ordinal());
			Assert.assertEquals(OpOp3.valueOf(ops[i].name()), ops[i]);
			Assert.assertEquals(OpOp3.valueOfByOpcode(ops[i].toString()), ops[i]);
		}
	}
	
	@Test
	public void testOpOp4() {
		OpOp4[] ops = OpOp4.values();
		for(int i=0; i<ops.length; i++) {
			Assert.assertEquals(i, ops[i].ordinal());
			Assert.assertEquals(OpOp4.valueOf(ops[i].name()), ops[i]);
			Assert.assertEquals(OpOp4.valueOf(ops[i].toString().toUpperCase()), ops[i]);
		}
	}
	
	@Test
	public void testOpOpN() {
		OpOpN[] ops = OpOpN.values();
		for(int i=0; i<ops.length; i++) {
			Assert.assertEquals(i, ops[i].ordinal());
			Assert.assertEquals(OpOpN.valueOf(ops[i].name()), ops[i]);
			if(ops[1]==OpOpN.MAX||ops[1]==OpOpN.MIN||ops[1]==OpOpN.PLUS)
				Assert.assertTrue(ops[1].isCellOp());
		}
	}
	
	@Test
	public void testReOrgOp() {
		ReOrgOp[] ops = ReOrgOp.values();
		for(int i=0; i<ops.length; i++) {
			Assert.assertEquals(i, ops[i].ordinal());
			Assert.assertEquals(ReOrgOp.valueOf(ops[i].name()), ops[i]);
			Assert.assertEquals(ReOrgOp.valueOfByOpcode(ops[i].toString()), ops[i]);
		}
	}
	
	@Test
	public void testParamBuiltinOp() {
		ParamBuiltinOp[] ops = ParamBuiltinOp.values();
		for(int i=0; i<ops.length; i++) {
			Assert.assertEquals(i, ops[i].ordinal());
			Assert.assertEquals(ParamBuiltinOp.valueOf(ops[i].name()), ops[i]);
		}
	}
	
	@Test
	public void testOpOpDnn() {
		OpOpDnn[] ops = OpOpDnn.values();
		for(int i=0; i<ops.length; i++) {
			Assert.assertEquals(i, ops[i].ordinal());
			Assert.assertEquals(OpOpDnn.valueOf(ops[i].name()), ops[i]);
		}
	}
	
	@Test
	public void testOpOpDG() {
		OpOpDG[] ops = OpOpDG.values();
		for(int i=0; i<ops.length; i++) {
			Assert.assertEquals(i, ops[i].ordinal());
			Assert.assertEquals(OpOpDG.valueOf(ops[i].name()), ops[i]);
		}
	}

	@Test
	public void testOpOpData() {
		OpOpData[] ops = OpOpData.values();
		for(int i=0; i<ops.length; i++) {
			Assert.assertEquals(i, ops[i].ordinal());
			Assert.assertEquals(OpOpData.valueOf(ops[i].name()), ops[i]);
		}
	}
}
