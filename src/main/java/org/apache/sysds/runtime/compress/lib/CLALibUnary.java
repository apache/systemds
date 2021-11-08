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
package org.apache.sysds.runtime.compress.lib;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public class CLALibUnary {
	private static final Log LOG = LogFactory.getLog(CLALibUnary.class.getName());
	
	public static MatrixBlock unaryOperations(CompressedMatrixBlock m, UnaryOperator op, MatrixValue result) {
		final boolean overlapping = m.isOverlapping();
		final int r = m.getNumRows();
		final int c = m.getNumColumns();
		
		// early aborts:
		if(m.isEmpty())
			return new MatrixBlock(r, c, 0).unaryOperations(op, result);
		else if(overlapping) {
			// when in overlapping state it is guaranteed that there is no infinites, NA, or NANs.
			if(Builtin.isBuiltinCode(op.fn, BuiltinCode.ISINF, BuiltinCode.ISNA, BuiltinCode.ISNAN))
				return new MatrixBlock(r, c, 0);
		}
		else if(Builtin.isBuiltinCode(op.fn, BuiltinCode.ISINF, BuiltinCode.ISNAN, BuiltinCode.ISNA) && !m.containsValue(op.getPattern()))
			return new MatrixBlock(r, c, 0); // avoid unnecessary allocation
		

		if(op.isInplace()) {
			LOG.warn("Compressed ops forcing unaryOperator not to be inplace.");
			op = new UnaryOperator(op.fn, op.getNumThreads(), false);
		}

		MatrixBlock uc = m.getUncompressed("unaryOperations " + op.fn.toString());
		MatrixBlock ret = uc.unaryOperations(op, result);
		return ret;
	}
}
