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

import static org.junit.Assert.assertEquals;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.junit.Test;

public class RewriteUtilsTest 
{
	@Test
	public void testUnaryValueTypes() {
		Hop input = new LiteralOp("true");
		
		assertEquals(ValueType.FP64,
			HopRewriteUtils.createUnary(input, OpOp1.CAST_AS_DOUBLE).getValueType());
		assertEquals(ValueType.INT64,
			HopRewriteUtils.createUnary(input, OpOp1.CAST_AS_INT).getValueType());
		assertEquals(ValueType.BOOLEAN, 
			HopRewriteUtils.createUnary(input, OpOp1.CAST_AS_BOOLEAN).getValueType());
	}
	
	@Test
	public void testBinaryValueTypes1() {
		Hop input1 = new LiteralOp(7d);
		Hop input2 = new DataOp("tmp", DataType.MATRIX, ValueType.INT64,
			OpOpData.TRANSIENTREAD, null, 3, 7, 21, 1000);
		assertEquals(ValueType.FP64,
			HopRewriteUtils.createBinary(input1, input2, OpOp2.MULT).getValueType());
	}
	
	@Test
	public void testBinaryValueTypes2() {
		Hop input1 = new LiteralOp(7);
		Hop input2 = new DataOp("tmp", DataType.MATRIX, ValueType.INT64,
			OpOpData.TRANSIENTREAD, null, 3, 7, 21, 1000);
		assertEquals(ValueType.FP64,
			HopRewriteUtils.createBinary(input1, input2, OpOp2.MULT).getValueType());
	}
	
	@Test
	public void testBinaryValueTypes3() {
		Hop input1 = new LiteralOp(7);
		Hop input2 = new LiteralOp(3);
		assertEquals(ValueType.INT64,
			HopRewriteUtils.createBinary(input1, input2, OpOp2.MULT).getValueType());
	}
}
