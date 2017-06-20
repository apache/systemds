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

package org.apache.sysml.test.integration.functions.codegen;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.DataOpTypes;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.codegen.cplan.CNode;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary;
import org.apache.sysml.hops.codegen.cplan.CNodeData;
import org.apache.sysml.hops.codegen.cplan.CNodeTernary;
import org.apache.sysml.hops.codegen.cplan.CNodeTernary.TernaryType;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary;
import org.apache.sysml.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;

/**
 * A CPlan is the internal representation of code generation plans
 * and consists of a DAG of CNodes and a surrounding template. These 
 * plans implements equals and hashCode to efficient match equivalent
 * plans and subexpressions. Since this was a frequent source of issues
 * in the past, this testsuite aims to explicitly check various scenarios.
 * 
 */
public class CPlanComparisonTest extends AutomatedTestBase 
{
	private IDSequence _seq = new IDSequence();
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}
	
	@Test
	public void testEqualLiteral() {
		CNodeData c1 = new CNodeData(new LiteralOp(7), 0, 0, DataType.SCALAR);
		CNodeData c2 = new CNodeData(new LiteralOp(7), 0, 0, DataType.SCALAR);
		Assert.assertEquals(c1.hashCode(), c2.hashCode());
		Assert.assertEquals(c1, c2);
		c1.setLiteral(true);
		c2.setLiteral(true);
		Assert.assertEquals(c1.hashCode(), c2.hashCode());
		Assert.assertEquals(c1, c2);
		c1.setStrictEquals(true);
		c2.setStrictEquals(true);
		Assert.assertEquals(c1.hashCode(), c2.hashCode());
		Assert.assertEquals(c1, c2);
	}
	
	@Test
	public void testNotEqualLiteral() {
		CNodeData c1 = new CNodeData(new LiteralOp(7), 0, 0, DataType.SCALAR);
		CNodeData c2 = new CNodeData(new LiteralOp(3), 0, 0, DataType.SCALAR);
		Assert.assertNotEquals(c1.hashCode(), c2.hashCode());
		Assert.assertNotEquals(c1, c2);
		c1.setLiteral(true);
		c2.setLiteral(true);
		Assert.assertNotEquals(c1.hashCode(), c2.hashCode());
		Assert.assertNotEquals(c1, c2);
		c1.setStrictEquals(true);
		c2.setStrictEquals(true);
		Assert.assertNotEquals(c1.hashCode(), c2.hashCode());
		Assert.assertNotEquals(c1, c2);
	}
	
	@Test
	public void testEqualMatrixDataNode() {
		Hop data = createDataOp(DataType.MATRIX);
		CNode c1 = new CNodeData(data);
		CNode c2 = new CNodeData(data);
		Assert.assertEquals(c1.hashCode(), c2.hashCode());
		Assert.assertEquals(c1, c2);
	}
	
	@Test
	public void testNotEqualDataTypeDataNode() {
		Assert.assertNotEquals(
			createCNodeData(DataType.MATRIX),
			createCNodeData(DataType.SCALAR));
	}
	
	@Test
	public void testEqualUnaryNodes() {
		CNode c0 = createCNodeData(DataType.MATRIX);
		CNode c1 = new CNodeUnary(c0, UnaryType.EXP);
		CNode c2 = new CNodeUnary(c0, UnaryType.EXP);
		Assert.assertEquals(c1.hashCode(), c2.hashCode());
		Assert.assertEquals(c1, c2);
	}
	
	@Test
	public void testNotEqualUnaryNodes() {
		CNode c0 = createCNodeData(DataType.MATRIX);
		CNode c1 = new CNodeUnary(c0, UnaryType.EXP);
		CNode c2 = new CNodeUnary(c0, UnaryType.LOG);
		Assert.assertNotEquals(c1, c2);
	}
	
	@Test
	public void testEqualBinaryNodes() {
		CNode c1 = createCNodeData(DataType.MATRIX);
		CNode c2 = createCNodeData(DataType.SCALAR);
		CNode bin1 = new CNodeBinary(c1, c2, BinType.PLUS);
		CNode bin2 = new CNodeBinary(c1, c2, BinType.PLUS);
		Assert.assertEquals(bin1.hashCode(), bin2.hashCode());
		Assert.assertEquals(bin1, bin2);
	}
	
	@Test
	public void testNotEqualBinaryNodes() {
		CNode c1 = createCNodeData(DataType.MATRIX);
		CNode c2 = createCNodeData(DataType.SCALAR);
		Assert.assertNotEquals(
			new CNodeBinary(c1, c2, BinType.PLUS),
			new CNodeBinary(c1, c2, BinType.MULT));
	}
	
	@Test
	public void testEqualTernaryNodes() {
		CNode c1 = createCNodeData(DataType.MATRIX);
		CNode c2 = createCNodeData(DataType.SCALAR);
		CNode c3 = createCNodeData(DataType.MATRIX);
		CNode ter1 = new CNodeTernary(c1, c2, c3, TernaryType.MINUS_MULT);
		CNode ter2 = new CNodeTernary(c1, c2, c3, TernaryType.MINUS_MULT);
		Assert.assertEquals(ter1.hashCode(), ter2.hashCode());
		Assert.assertEquals(ter1, ter2);
	}
	
	@Test
	public void testNotEqualTernaryNodes() {
		CNode c1 = createCNodeData(DataType.MATRIX);
		CNode c2 = createCNodeData(DataType.SCALAR);
		CNode c3 = createCNodeData(DataType.MATRIX);
		CNode ter1 = new CNodeTernary(c1, c2, c3, TernaryType.MINUS_MULT);
		CNode ter2 = new CNodeTernary(c1, c2, c3, TernaryType.PLUS_MULT);
		Assert.assertNotEquals(ter1, ter2);
	}

	@Test
	public void testNotEqualUnaryBinaryNodes() {
		CNode c1 = createCNodeData(DataType.MATRIX);
		CNode c2 = createCNodeData(DataType.SCALAR);
		CNode un1 = new CNodeUnary(c1, UnaryType.ABS);
		CNode bin2 = new CNodeBinary(c1, c2, BinType.DIV);
		Assert.assertNotEquals(un1, bin2);
	}
	
	@Test
	public void testNotEqualUnaryTernaryNodes() {
		CNode c1 = createCNodeData(DataType.MATRIX);
		CNode c2 = createCNodeData(DataType.SCALAR);
		CNode c3 = createCNodeData(DataType.MATRIX);
		CNode un1 = new CNodeUnary(c1, UnaryType.ABS);
		CNode ter2 = new CNodeTernary(c1, c2, c3, TernaryType.PLUS_MULT);
		Assert.assertNotEquals(un1, ter2);
	}
	
	@Test
	public void testNotEqualBinaryTernaryNodes() {
		CNode c1 = createCNodeData(DataType.MATRIX);
		CNode c2 = createCNodeData(DataType.SCALAR);
		CNode c3 = createCNodeData(DataType.MATRIX);
		CNode un1 = new CNodeBinary(c1, c2, BinType.EQUAL);
		CNode ter2 = new CNodeTernary(c1, c2, c3, TernaryType.PLUS_MULT);
		Assert.assertNotEquals(un1, ter2);
	}
	
	@Test
	public void testNotEqualBinaryDAG1() {
		CNode c1 = createCNodeData(DataType.MATRIX);
		CNode c2 = createCNodeData(DataType.MATRIX);
		CNode c3 = createCNodeData(DataType.SCALAR);
		//DAG 1a: (c1*c2)*c3
		CNode b1a = new CNodeBinary(c1, c2, BinType.MULT);
		CNode b2a = new CNodeBinary(b1a, c3, BinType.MULT);
		//DAG 1b: (c1*c2)*c1
		CNode b1b = new CNodeBinary(c1, c2, BinType.MULT);
		CNode b2b = new CNodeBinary(b1b, c1, BinType.MULT);
		Assert.assertNotEquals(b2a, b2b);
	}
	
	@Test
	public void testNotEqualBinaryDAG2() {
		CNode c1 = createCNodeData(DataType.MATRIX);
		CNode c2 = createCNodeData(DataType.MATRIX);
		CNode c3 = createCNodeData(DataType.MATRIX);
		//DAG 2a: (c1*c2)*c3
		CNode b1a = new CNodeBinary(c1, c2, BinType.MULT);
		CNode b2a = new CNodeBinary(b1a, c3, BinType.MULT);
		//DAG 2b: (c1*c2)*c1
		CNode b1b = new CNodeBinary(c1, c2, BinType.MULT);
		CNode b2b = new CNodeBinary(b1b, c1, BinType.MULT);
		Assert.assertNotEquals(b2a, b2b);
	}
	
	@Test
	public void testNotEqualBinaryDAG3() {
		CNode c1 = createCNodeData(DataType.MATRIX);
		CNode c2 = createCNodeData(DataType.MATRIX);
		CNode c3 = createCNodeData(DataType.MATRIX);
		//DAG 3a: (c1+c3)*(c2+c3)
		CNode b1a = new CNodeBinary(c1, c3, BinType.PLUS);
		CNode b2a = new CNodeBinary(c2, c3, BinType.PLUS);
		CNode b3a = new CNodeBinary(b1a, b2a, BinType.MULT);
		//DAG 3b: (c1+c2)*(c3+c3)
		CNode b1b = new CNodeBinary(c1, c2, BinType.PLUS);
		CNode b2b = new CNodeBinary(c3, c3, BinType.PLUS);
		CNode b3b = new CNodeBinary(b1b, b2b, BinType.MULT);
		Assert.assertNotEquals(b3a, b3b);
	}
	
	private CNode createCNodeData(DataType dt) {
		return new CNodeData(createDataOp("tmp"+_seq.getNextID(), dt));
	}
	
	private Hop createDataOp(DataType dt) {
		return new DataOp("tmp"+_seq.getNextID(), dt, ValueType.DOUBLE, 
			DataOpTypes.TRANSIENTREAD, "tmp", 77L, 7L, -1L, 1000L, 1000L);
	}
	
	private Hop createDataOp(String name, DataType dt) {
		return new DataOp(name, dt, ValueType.DOUBLE, 
			DataOpTypes.TRANSIENTREAD, "tmp", 77L, 7L, -1L, 1000L, 1000L);
	}
}
