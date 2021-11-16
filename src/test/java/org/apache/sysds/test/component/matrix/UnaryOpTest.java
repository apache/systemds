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
package org.apache.sysds.test.component.matrix;

import static org.junit.Assert.assertTrue;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.junit.Test;

public class UnaryOpTest {
	protected static final Log LOG = LogFactory.getLog(UnaryOpTest.class.getName());
	static final MatrixBlock m = new MatrixBlock(100, 100, false);
	static final UnaryOperator op = new UnaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.ROUND));
	
	static {
		m.setValue(3, 3, 4.2);
	}

	@Test
	public void testBasic() {
		assertTrue(m.getValue(3, 3) == 4.2);
	}

	@Test
	public void testDirectUnaryOp() {
		MatrixBlock mr = m.unaryOperations(op, null);
		assertTrue(mr.getValue(3, 3) == 4);
	}

	@Test
	public void testFromSparseCSRUnaryOp() {
		MatrixBlock sb = new MatrixBlock(1, 1, false);
		sb.copy(m);
		sb.setSparseBlock(new SparseBlockCSR(sb.getSparseBlock()));
		MatrixBlock mr = sb.unaryOperations(op, null);
		assertTrue(mr.getValue(3, 3) == 4);
	}

	@Test
	public void testFromSparseMCSRUnaryOp() {
		MatrixBlock sb = new MatrixBlock(1, 1, false);
		sb.copy(m);
		MatrixBlock mr = sb.unaryOperations(op, null);
		assertTrue(mr.getValue(3, 3) == 4);
	}
}
