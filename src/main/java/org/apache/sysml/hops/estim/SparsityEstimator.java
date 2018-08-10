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

package org.apache.sysml.hops.estim;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public abstract class SparsityEstimator 
{
	protected static final Log LOG = LogFactory.getLog(SparsityEstimator.class.getName());
	
	//internal configuration
	public static boolean MULTI_THREADED_BUILD = false;
	public static boolean MULTI_THREADED_ESTIM = false;
	public static final int MIN_PAR_THRESHOLD = 10 * 1024;
	
	private static OpCode[] EXACT_META_DATA_OPS = new OpCode[] {
		OpCode.EQZERO, OpCode.NEQZERO, OpCode.CBIND,
		OpCode.RBIND, OpCode.TRANS, OpCode.DIAG, OpCode.RESHAPE
	};
	
	public static enum OpCode {
		MM, 
		MULT, PLUS, EQZERO, NEQZERO,
		CBIND, RBIND, 
		TRANS, DIAG, RESHAPE;
	}
	
	/**
	 * Estimates the output sparsity of a DAG of matrix multiplications
	 * for the given operator graph of a single root node.
	 * 
	 * @param root
	 * @return
	 */
	public abstract MatrixCharacteristics estim(MMNode root);
	
	
	/**
	 * Estimates the output sparsity for a single matrix multiplication.
	 * 
	 * @param m1 left-hand-side operand
	 * @param m2 right-hand-side operand
	 * @return sparsity
	 */
	public abstract double estim(MatrixBlock m1, MatrixBlock m2);
	
	/**
	 * Estimates the output sparsity for a given binary operation.
	 * 
	 * @param m1 left-hand-side operand
	 * @param m2 right-hand-side operand
	 * @param op operator code
	 * @return sparsity
	 */
	public abstract double estim(MatrixBlock m1, MatrixBlock m2, OpCode op);
	
	/**
	 * Estimates the output sparsity for a given unary operation.
	 * 
	 * @param m1 left-hand-side operand
	 * @param op operator code
	 * @return sparsity
	 */
	public abstract double estim(MatrixBlock m, OpCode op);
	
	protected boolean isExactMetadataOp(OpCode op) {
		return ArrayUtils.contains(EXACT_META_DATA_OPS, op);
	}
	
	protected MatrixCharacteristics estimExactMetaData(MatrixCharacteristics mc1, MatrixCharacteristics mc2, OpCode op) {
		switch( op ) {
			case EQZERO:
				return new MatrixCharacteristics(mc1.getRows(), mc1.getCols(),
					(long) mc1.getRows() * mc1.getCols() - mc1.getNonZeros());
			case DIAG:
				return (mc1.getCols() == 1) ?
					new MatrixCharacteristics(mc1.getRows(), mc1.getRows(), mc1.getNonZeros()) :
					new MatrixCharacteristics(mc1.getRows(), 1, Math.min(mc1.getRows(), mc1.getNonZeros()));
			// binary operations that preserve sparsity exactly
			case CBIND:
				return new MatrixCharacteristics(mc1.getRows(), 
					mc1.getCols() + mc2.getCols(), mc1.getNonZeros() + mc2.getNonZeros());
			case RBIND:
				return new MatrixCharacteristics(mc1.getRows() + mc2.getRows(), 
					mc1.getCols(), mc1.getNonZeros() + mc2.getNonZeros());
			// unary operation that preserve sparsity exactly
			case NEQZERO:
			case TRANS:
			case RESHAPE:
				return mc1;
			default:
				throw new HopsException("Opcode is not an exact meta data operation: "+op.name());
		}
	}
}
