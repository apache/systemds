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


package org.apache.sysds.runtime.matrix.operators;

import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.And;
import org.apache.sysds.runtime.functionobjects.BitwAnd;
import org.apache.sysds.runtime.functionobjects.BitwOr;
import org.apache.sysds.runtime.functionobjects.BitwShiftL;
import org.apache.sysds.runtime.functionobjects.BitwShiftR;
import org.apache.sysds.runtime.functionobjects.BitwXor;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Equals;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysds.runtime.functionobjects.IntegerDivide;
import org.apache.sysds.runtime.functionobjects.LessThan;
import org.apache.sysds.runtime.functionobjects.LessThanEquals;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Minus1Multiply;
import org.apache.sysds.runtime.functionobjects.MinusMultiply;
import org.apache.sysds.runtime.functionobjects.MinusNz;
import org.apache.sysds.runtime.functionobjects.Modulus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.NotEquals;
import org.apache.sysds.runtime.functionobjects.Or;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.PlusMultiply;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.functionobjects.Xor;

/**
 * BinaryOperator class for operations that have two inputs.
 * 
 * For instance
 * 
 * <pre>
 *BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject());
 *double r = op.execute(5.0, 8.2)
 * </pre>
 */
public class BinaryOperator extends MultiThreadedOperator {
	private static final long serialVersionUID = -2547950181558989209L;

	public final ValueFunction fn;
	public final boolean commutative;

	public BinaryOperator(ValueFunction p) {
		this(p, 1);
	}

	public BinaryOperator(ValueFunction p, int k) {
		// binaryop is sparse-safe iff (0 op 0) == 0
		super(p instanceof Plus || p instanceof Multiply || p instanceof Minus || p instanceof PlusMultiply ||
			p instanceof MinusMultiply || p instanceof And || p instanceof Or || p instanceof Xor ||
			p instanceof BitwAnd || p instanceof BitwOr || p instanceof BitwXor || p instanceof BitwShiftL ||
			p instanceof BitwShiftR);
		fn = p;
		commutative = p instanceof Plus || p instanceof Multiply || p instanceof And || p instanceof Or ||
			p instanceof Xor || p instanceof Minus1Multiply;
		_numThreads = k;
	}

	public BinaryOperator(ValueFunction p, boolean sparseSafe){
		// enforce desired sparse safety
		super(sparseSafe);
		fn = p;
		commutative = p instanceof Plus || p instanceof Multiply || p instanceof And || p instanceof Or ||
				p instanceof Xor || p instanceof Minus1Multiply;
		_numThreads = 1;
	}

	/**
	 * Method for getting the hop binary operator type for a given function object.
	 * This is used in order to use a common code path for consistency between 
	 * compiler and runtime.
	 * 
	 * @return binary operator type for a function object
	 */
	public OpOp2 getBinaryOperatorOpOp2() {
		if( fn instanceof Plus )                   return OpOp2.PLUS;
		else if( fn instanceof Minus )             return OpOp2.MINUS;
		else if( fn instanceof Multiply )          return OpOp2.MULT;
		else if( fn instanceof Divide )            return OpOp2.DIV;
		else if( fn instanceof Modulus )           return OpOp2.MODULUS;
		else if( fn instanceof IntegerDivide )     return OpOp2.INTDIV;
		else if( fn instanceof LessThan )          return OpOp2.LESS;
		else if( fn instanceof LessThanEquals )    return OpOp2.LESSEQUAL;
		else if( fn instanceof GreaterThan )       return OpOp2.GREATER;
		else if( fn instanceof GreaterThanEquals ) return OpOp2.GREATEREQUAL;
		else if( fn instanceof Equals )            return OpOp2.EQUAL;
		else if( fn instanceof NotEquals )         return OpOp2.NOTEQUAL;
		else if( fn instanceof And )               return OpOp2.AND;
		else if( fn instanceof Or )                return OpOp2.OR;
		else if( fn instanceof Xor )               return OpOp2.XOR;
		else if( fn instanceof BitwAnd )           return OpOp2.BITWAND;
		else if( fn instanceof BitwOr )            return OpOp2.BITWOR;
		else if( fn instanceof BitwXor )           return OpOp2.BITWXOR;
		else if( fn instanceof BitwShiftL )        return OpOp2.BITWSHIFTL;
		else if( fn instanceof BitwShiftR )        return OpOp2.BITWSHIFTR;
		else if( fn instanceof Power )             return OpOp2.POW;
		else if( fn instanceof MinusNz )           return OpOp2.MINUS_NZ;
		else if( fn instanceof Builtin ) {
			BuiltinCode bfc = ((Builtin) fn).getBuiltinCode();
			if( bfc == BuiltinCode.MIN )           return OpOp2.MIN;
			else if( bfc == BuiltinCode.MAX )      return OpOp2.MAX;
			else if( bfc == BuiltinCode.LOG )      return OpOp2.LOG;
			else if( bfc == BuiltinCode.LOG_NZ )   return OpOp2.LOG_NZ;
		}
		
		//non-supported ops (not required for sparsity estimates):
		//PRINT, CONCAT, QUANTILE, INTERQUANTILE, IQM, 
		//CENTRALMOMENT, COVARIANCE, APPEND, SOLVE, MEDIAN,
		return null;
	}
	
	public boolean isCommutative() {
		return commutative;
	}

	/**
	 * Check if the operation returns zeros if the input zero.
	 * 
	 * @param row The values to check
	 * @return If the output is always zero if other value is zero
	 */
	public boolean isRowSafeLeft(double[] row){
		for(double v : row)
			 if(0 !=  fn.execute(v, 0))
			 	return false;
		return true;
	}

	/**
	 * Check if the operation returns zeros if the input zero.
	 * 
	 * @param row The values to check
	 * @return If the output is always zero if other value is zero
	 */
	public boolean isRowSafeLeft(MatrixBlock row) {
		if(row.isEmpty())
			return 0 == fn.execute(0.0, 0.0);
		else if(row.isInSparseFormat()) {
			if(0 != fn.execute(0.0, 0.0))
				return false;
			SparseBlock sb = row.getSparseBlock();
			if(sb.isEmpty(0))
				return true;
			return isRowSafeLeft(sb.values(0));
		}
		else
			return isRowSafeLeft(row.getDenseBlockValues());
	}
	
	/**
	 * Check if the operation returns zeros if the input is contained in row.
	 * 
	 * @param row The values to check
	 * @return If the output contains zeros
	 */
	public boolean isIntroducingZerosLeft(MatrixBlock row) {
		if(row.isEmpty())
			return introduceZeroLeft(0.0);
		else if(row.isInSparseFormat()) {
			if(introduceZeroLeft(0.0))
				return true;
			SparseBlock sb = row.getSparseBlock();
			if(sb.isEmpty(0))
				return false;
			return isIntroducingZerosLeft(sb.values(0));
		}
		else
			return isIntroducingZerosLeft(row.getDenseBlockValues());
	}

	/**
	 * Check if the operation returns zeros if the input is contained in row.
	 * 
	 * @param row The values to check
	 * @return If the output contains zeros
	 */
	public boolean isIntroducingZerosLeft(double[] row) {
		for(double v : row)
			if(introduceZeroLeft(v))
				return true;
		return false;
	}

	/**
	 * Check if zero is returned at arbitrary input. The verification is done via two different values that hopefully do
	 * not return 0 in both instances unless the operation really have a tendency to return zero.
	 * 
	 * @param v The value to check if returns zero
	 * @return if the evaluation return zero
	 */
	private boolean introduceZeroLeft(double v) {
		return 0 == fn.execute(v, 11.42) && 0 == fn.execute(v, -11.22);
	}

	/**
	 * Check if the operation returns zeros if the input zero.
	 * 
	 * @param row The values to check
	 * @return If the output is always zero if other value is zero
	 */
	public boolean isRowSafeRight(double[] row){
		for(double v : row)
			 if(0 !=  fn.execute(0, v))
			 	return false;
		return true;
	}
	
	/**
	 * Check if the operation returns zeros if the input zero.
	 * 
	 * @param row The values to check
	 * @return If the output is always zero if other value is zero
	 */
	public boolean isRowSafeRight(MatrixBlock row) {
		if(row.isEmpty())
			return 0 == fn.execute(0.0, 0.0);
		else if(row.isInSparseFormat()) {
			if(0 != fn.execute(0.0, 0.0))
				return false;
			SparseBlock sb = row.getSparseBlock();
			if(sb.isEmpty(0))
				return true;
			return isRowSafeRight(sb.values(0));
		}
		else
			return isRowSafeRight(row.getDenseBlockValues());
	}

	/**
	 * Check if the operation returns zeros if the input is contained in row.
	 * 
	 * @param row The values to check
	 * @return If the output contains zeros
	 */
	public boolean isIntroducingZerosRight(MatrixBlock row){
		if(row.isEmpty())
			return  introduceZeroRight(0.0);
		else if(row.isInSparseFormat()){
			if (introduceZeroRight(0.0))
				return true;
			SparseBlock sb = row.getSparseBlock();
			if(sb.isEmpty(0))
				return false;
			return isIntroducingZerosRight(sb.values(0));	
		}
		else 
			return isIntroducingZerosRight(row.getDenseBlockValues());
	}

	/**
	 * Check if the operation returns zeros if the input is contained in row.
	 * 
	 * @param row The values to check
	 * @return If the output contains zeros
	 */
	public boolean isIntroducingZerosRight(double[] row){
		for(double v : row)
			if( introduceZeroRight(v))
				return true;

		return false;
	}

	/**
	 * Check if zero is returned at arbitrary input. The verification is done via two different values that hopefully do
	 * not return 0 in both instances unless the operation really have a tendency to return zero.
	 * 
	 * @param v The value to check if returns zero
	 * @return if the evaluation return zero
	 */
	private boolean introduceZeroRight(double v) {
		return 0 == fn.execute(11.42, v) && 0 == fn.execute(-11.22, v);
	}

	@Override
	public String toString() {
		return "BinaryOperator("+fn.getClass().getSimpleName()+")";
	}
}
