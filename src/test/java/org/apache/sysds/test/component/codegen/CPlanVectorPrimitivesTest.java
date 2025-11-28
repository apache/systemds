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

package org.apache.sysds.test.component.codegen;

import java.lang.reflect.Method;

import org.apache.sysds.runtime.data.SparseRowVector;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.apache.hadoop.util.StringUtils;

public class CPlanVectorPrimitivesTest extends AutomatedTestBase 
{
	private static final int m = 3;
	private static final int n = 132;
	private static final double sparsity1 = 0.9;
	private static final double sparsity2 = 0.09;
	private static final double eps = 1e-10;
	
	private enum InputType {
		SCALAR,
		VECTOR_DENSE,
		VECTOR_SPARSE,
	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}
	
	//support aggregate vector primitives
	
	@Test
	public void testVectorSumDense() {
		testVectorAggPrimitive(UnaryType.ROW_SUMS, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorSumSparse() {
		testVectorAggPrimitive(UnaryType.ROW_SUMS, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorMinDense() {
		testVectorAggPrimitive(UnaryType.ROW_MINS, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorMinSparse() {
		testVectorAggPrimitive(UnaryType.ROW_MINS, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorMaxDense() {
		testVectorAggPrimitive(UnaryType.ROW_MAXS, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorMaxSparse() {
		testVectorAggPrimitive(UnaryType.ROW_MAXS, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorMeansDense() {
		testVectorAggPrimitive(UnaryType.ROW_MEANS, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorMeansSparse() {
		testVectorAggPrimitive(UnaryType.ROW_MEANS, InputType.VECTOR_SPARSE);
	}
	
	//support unary vector primitives (pow2/mult2 current excluded because not unary)
	
	@Test
	public void testVectorExpDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_EXP, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorExpSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_EXP, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorSqrtDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_SQRT, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorSqrtSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_SQRT, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorLogDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_LOG, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorLogSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_LOG, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorAbsDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_ABS, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorAbsSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_ABS, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorRoundDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_ROUND, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorRoundSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_ROUND, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorCeilDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_CEIL, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorCeilSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_CEIL, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorFloorDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_FLOOR, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorFloorSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_FLOOR, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorSinDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_SIN, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorSinSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_SIN, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorCosDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_COS, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorCosSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_COS, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorTanDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_TAN, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorTanSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_TAN, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorASinDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_ASIN, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorASinSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_ASIN, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorACosDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_ACOS, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorACosSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_ACOS, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorATanDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_ATAN, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorATanSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_ATAN, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorSinHDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_SINH, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorSinHSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_SINH, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorCosHDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_COSH, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorCosHSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_COSH, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorTanHDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_TANH, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorTanHSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_TANH, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorSignDense() {
		testVectorUnaryPrimitive(UnaryType.VECT_SIGN, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorSignSparse() {
		testVectorUnaryPrimitive(UnaryType.VECT_SIGN, InputType.VECTOR_SPARSE);
	}
	
	//support unary vector primitives (pow currently only on vector-scalars)
	
	@Test
	public void testVectorScalarMultDense() {
		testVectorBinaryPrimitive(BinType.VECT_MULT_SCALAR, InputType.VECTOR_DENSE, InputType.SCALAR);
	}
	
	@Test
	public void testVectorScalarMultSparse() {
		testVectorBinaryPrimitive(BinType.VECT_MULT_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}
	
	@Test
	public void testScalarVectorMultDense() {
		testVectorBinaryPrimitive(BinType.VECT_MULT_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testScalarVectorMultSparse() {
		testVectorBinaryPrimitive(BinType.VECT_MULT_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorVectorMultDenseDense() {
		testVectorBinaryPrimitive(BinType.VECT_MULT, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorVectorMultSparseDense() {
		testVectorBinaryPrimitive(BinType.VECT_MULT, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorScalarDivDense() {
		testVectorBinaryPrimitive(BinType.VECT_DIV_SCALAR, InputType.VECTOR_DENSE, InputType.SCALAR);
	}
	
	@Test
	public void testVectorScalarDivSparse() {
		testVectorBinaryPrimitive(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}
	
	@Test
	public void testScalarVectorDivDense() {
		testVectorBinaryPrimitive(BinType.VECT_DIV_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testScalarVectorDivSparse() {
		testVectorBinaryPrimitive(BinType.VECT_DIV_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorVectorDivDenseDense() {
		testVectorBinaryPrimitive(BinType.VECT_DIV, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorVectorDivSparseDense() {
		testVectorBinaryPrimitive(BinType.VECT_DIV, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorScalarPlusDense() {
		testVectorBinaryPrimitive(BinType.VECT_PLUS_SCALAR, InputType.VECTOR_DENSE, InputType.SCALAR);
	}
	
	@Test
	public void testVectorScalarPlusSparse() {
		testVectorBinaryPrimitive(BinType.VECT_PLUS_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}
	
	@Test
	public void testScalarVectorPlusDense() {
		testVectorBinaryPrimitive(BinType.VECT_PLUS_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testScalarVectorPlusSparse() {
		testVectorBinaryPrimitive(BinType.VECT_PLUS_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorVectorPlusDenseDense() {
		testVectorBinaryPrimitive(BinType.VECT_PLUS, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorVectorPlusSparseDense() {
		testVectorBinaryPrimitive(BinType.VECT_PLUS, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorScalarMinusDense() {
		testVectorBinaryPrimitive(BinType.VECT_MINUS_SCALAR, InputType.VECTOR_DENSE, InputType.SCALAR);
	}
	
	@Test
	public void testVectorScalarMinusSparse() {
		testVectorBinaryPrimitive(BinType.VECT_MINUS_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}
	
	@Test
	public void testScalarVectorMinusDense() {
		testVectorBinaryPrimitive(BinType.VECT_MINUS_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testScalarVectorMinusSparse() {
		testVectorBinaryPrimitive(BinType.VECT_MINUS_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorVectorMinusDenseDense() {
		testVectorBinaryPrimitive(BinType.VECT_MINUS, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorVectorMinusSparseDense() {
		testVectorBinaryPrimitive(BinType.VECT_MINUS, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorScalarMinDense() {
		testVectorBinaryPrimitive(BinType.VECT_MIN_SCALAR, InputType.VECTOR_DENSE, InputType.SCALAR);
	}
	
	@Test
	public void testVectorScalarMinSparse() {
		testVectorBinaryPrimitive(BinType.VECT_MIN_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}
	
	@Test
	public void testScalarVectorMinDense() {
		testVectorBinaryPrimitive(BinType.VECT_MIN_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testScalarVectorMinSparse() {
		testVectorBinaryPrimitive(BinType.VECT_MIN_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorVectorMinDenseDense() {
		testVectorBinaryPrimitive(BinType.VECT_MIN, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorVectorMinSparseDense() {
		testVectorBinaryPrimitive(BinType.VECT_MIN, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorScalarMaxDense() {
		testVectorBinaryPrimitive(BinType.VECT_MAX_SCALAR, InputType.VECTOR_DENSE, InputType.SCALAR);
	}
	
	@Test
	public void testVectorScalarMaxSparse() {
		testVectorBinaryPrimitive(BinType.VECT_MAX_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}
	
	@Test
	public void testScalarVectorMaxDense() {
		testVectorBinaryPrimitive(BinType.VECT_MAX_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testScalarVectorMaxSparse() {
		testVectorBinaryPrimitive(BinType.VECT_MAX_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorVectorMaxDenseDense() {
		testVectorBinaryPrimitive(BinType.VECT_MAX, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorVectorMaxSparseDense() {
		testVectorBinaryPrimitive(BinType.VECT_MAX, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorScalarPowDense() {
		testVectorBinaryPrimitive(BinType.VECT_POW_SCALAR, InputType.VECTOR_DENSE, InputType.SCALAR);
	}
	
	@Test
	public void testVectorScalarPowSparse() {
		testVectorBinaryPrimitive(BinType.VECT_POW_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}
	
	@Test
	public void testScalarVectorPowDense() {
		testVectorBinaryPrimitive(BinType.VECT_POW_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testScalarVectorPowSparse() {
		testVectorBinaryPrimitive(BinType.VECT_POW_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}
	
	//TODO pow vector-vector operations 
	
	@Test
	public void testVectorScalarEqualDense() {
		testVectorBinaryPrimitive(BinType.VECT_EQUAL_SCALAR, InputType.VECTOR_DENSE, InputType.SCALAR);
	}
	
	@Test
	public void testVectorScalarEqualSparse() {
		testVectorBinaryPrimitive(BinType.VECT_EQUAL_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}
	
	@Test
	public void testScalarVectorEqualDense() {
		testVectorBinaryPrimitive(BinType.VECT_EQUAL_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testScalarVectorEqualSparse() {
		testVectorBinaryPrimitive(BinType.VECT_EQUAL_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorVectorEqualDenseDense() {
		testVectorBinaryPrimitive(BinType.VECT_EQUAL, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorVectorEqualSparseDense() {
		testVectorBinaryPrimitive(BinType.VECT_EQUAL, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorScalarNotEqualDense() {
		testVectorBinaryPrimitive(BinType.VECT_NOTEQUAL_SCALAR, InputType.VECTOR_DENSE, InputType.SCALAR);
	}
	
	@Test
	public void testVectorScalarNotEqualSparse() {
		testVectorBinaryPrimitive(BinType.VECT_NOTEQUAL_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}
	
	@Test
	public void testScalarVectorNotEqualDense() {
		testVectorBinaryPrimitive(BinType.VECT_NOTEQUAL_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testScalarVectorNotEqualSparse() {
		testVectorBinaryPrimitive(BinType.VECT_NOTEQUAL_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorVectorNotEqualDenseDense() {
		testVectorBinaryPrimitive(BinType.VECT_NOTEQUAL, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorVectorNotEqualSparseDense() {
		testVectorBinaryPrimitive(BinType.VECT_NOTEQUAL, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorScalarLessDense() {
		testVectorBinaryPrimitive(BinType.VECT_LESS_SCALAR, InputType.VECTOR_DENSE, InputType.SCALAR);
	}
	
	@Test
	public void testVectorScalarLessSparse() {
		testVectorBinaryPrimitive(BinType.VECT_LESS_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}
	
	@Test
	public void testScalarVectorLessDense() {
		testVectorBinaryPrimitive(BinType.VECT_LESS_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testScalarVectorLessSparse() {
		testVectorBinaryPrimitive(BinType.VECT_LESS_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorVectorLessDenseDense() {
		testVectorBinaryPrimitive(BinType.VECT_LESS, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorVectorLessSparseDense() {
		testVectorBinaryPrimitive(BinType.VECT_LESS, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorScalarLessEqualDense() {
		testVectorBinaryPrimitive(BinType.VECT_LESSEQUAL_SCALAR, InputType.VECTOR_DENSE, InputType.SCALAR);
	}
	
	@Test
	public void testVectorScalarLessEqualSparse() {
		testVectorBinaryPrimitive(BinType.VECT_LESSEQUAL_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}
	
	@Test
	public void testScalarVectorLessEqualDense() {
		testVectorBinaryPrimitive(BinType.VECT_LESSEQUAL_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testScalarVectorLessEqualSparse() {
		testVectorBinaryPrimitive(BinType.VECT_LESSEQUAL_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorVectorLessEqualDenseDense() {
		testVectorBinaryPrimitive(BinType.VECT_LESSEQUAL, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorVectorLessEqualSparseDense() {
		testVectorBinaryPrimitive(BinType.VECT_LESSEQUAL, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorScalarGreaterDense() {
		testVectorBinaryPrimitive(BinType.VECT_GREATER_SCALAR, InputType.VECTOR_DENSE, InputType.SCALAR);
	}
	
	@Test
	public void testVectorScalarGreaterSparse() {
		testVectorBinaryPrimitive(BinType.VECT_GREATER_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}
	
	@Test
	public void testScalarVectorGreaterDense() {
		testVectorBinaryPrimitive(BinType.VECT_GREATER_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testScalarVectorGreaterSparse() {
		testVectorBinaryPrimitive(BinType.VECT_GREATER_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorVectorGreaterDenseDense() {
		testVectorBinaryPrimitive(BinType.VECT_GREATER, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorVectorGreaterSparseDense() {
		testVectorBinaryPrimitive(BinType.VECT_GREATER, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorScalarGreaterEqualDense() {
		testVectorBinaryPrimitive(BinType.VECT_GREATEREQUAL_SCALAR, InputType.VECTOR_DENSE, InputType.SCALAR);
	}
	
	@Test
	public void testVectorScalarGreaterEqualSparse() {
		testVectorBinaryPrimitive(BinType.VECT_GREATEREQUAL_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}
	
	@Test
	public void testScalarVectorGreaterEqualDense() {
		testVectorBinaryPrimitive(BinType.VECT_GREATEREQUAL_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testScalarVectorGreaterEqualSparse() {
		testVectorBinaryPrimitive(BinType.VECT_GREATEREQUAL_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}
	
	@Test
	public void testVectorVectorGreaterEqualDenseDense() {
		testVectorBinaryPrimitive(BinType.VECT_GREATEREQUAL, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}
	
	@Test
	public void testVectorVectorGreaterEqualSparseDense() {
		testVectorBinaryPrimitive(BinType.VECT_GREATEREQUAL, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE);
	}

	@Test
	public void testScalarVectorXorDense() {
		testVectorBinaryPrimitive(BinType.VECT_XOR_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}

	@Test
	public void testVectorScalarXorDense() {
		testVectorBinaryPrimitive(BinType.VECT_XOR_SCALAR, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}

	@Test
	public void testVectorVectorXorDenseDense() {
		testVectorBinaryPrimitive(BinType.VECT_XOR, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}

	@Test
	public void testVectorScalarXorSparse() {
		testVectorBinaryPrimitive(BinType.VECT_XOR_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	@Test
	public void testScalarVectorXorSparse() {
		testVectorBinaryPrimitive(BinType.VECT_XOR_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorVectorXorSparseDense() {
		testVectorBinaryPrimitive(BinType.VECT_XOR, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE);
	}

	//***************** Logical Bitwise Operators ********************//

	@Test //1.
	public void testVectorScalarBitwAndDense() {
		testVectorBinaryPrimitive(BinType.VECT_BITWAND_SCALAR, InputType.VECTOR_DENSE, InputType.SCALAR);
	}

	@Test //2.
	public void testScalarVectorBitwAndDense() {
		testVectorBinaryPrimitive(BinType.VECT_BITWAND_SCALAR, InputType.SCALAR, InputType.VECTOR_DENSE);
	}

	@Test //3.
	public void testVectorVectorBitwAndDenseDense() {
		testVectorBinaryPrimitive(BinType.VECT_BITWAND, InputType.VECTOR_DENSE, InputType.VECTOR_DENSE);
	}

	@Test //4.
	public void testVectorScalarBitwAndSparse() {
		testVectorBinaryPrimitive(BinType.VECT_BITWAND, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	@Test //5.
	public void testScalarVectorBitwAndSparse() {
		testVectorBinaryPrimitive(BinType.VECT_BITWAND, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}

	@Test //6.
	public void testVectorVectorBitwAndSparseDense() {
		testVectorBinaryPrimitive(BinType.VECT_BITWAND, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE);
	}

	//********************testing with sparse intermediates********************//
	//vector - scalar

	@Test
	public void testVectorScalarMultSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_MULT_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	@Test
	public void testVectorScalarDivSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	@Test
	public void testVectorScalarMinSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_MIN_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	@Test
	public void testVectorScalarMaxSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_MAX_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	@Test
	public void testVectorScalarPowSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_POW_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	@Test
	public void testVectorScalarEqualSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_EQUAL_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	@Test
	public void testVectorScalarNotEqualSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_NOTEQUAL_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	@Test
	public void testVectorScalarLessSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_LESS_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	@Test
	public void testVectorScalarLessEqualSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_LESSEQUAL_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	@Test
	public void testVectorScalarGreaterSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_GREATER_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	@Test
	public void testVectorScalarGreaterEqualSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_GREATEREQUAL_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	//todo: this only makes sense, when bval is 0
	@Test
	public void testVectorScalarXorSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_XOR_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	@Test
	public void testVectorScalarBitwAndSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_BITWAND_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
	}

	//scalar - vector

	@Test
	public void testScalarVectorMultSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_MULT_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testScalarVectorDivSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_DIV_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testScalarVectorMinSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_MIN_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testScalarVectorMaxSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_MAX_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testScalarVectorEqualSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_EQUAL_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testScalarVectorNotEqualSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_NOTEQUAL_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testScalarVectorLessSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_LESS_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testScalarVectorLessEqualSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_LESSEQUAL_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testScalarVectorGreaterSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_GREATER_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testScalarVectorGreaterEqualSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_GREATEREQUAL_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testScalarVectorXorSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_XOR_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testScalarVectorBitwAndSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_BITWAND_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE);
	}

	//special binary

	//	@Test
	//	public void testVectorPow2SparseToSparse() {
	//		testVectorUnarySparsePrimitive(UnaryType.VECT_POW2, InputType.VECTOR_SPARSE);
	//	}
	//
	//	@Test
	//	public void testVectorMult2SparseToSparse() {
	//		testVectorUnarySparsePrimitive(UnaryType.VECT_MULT2, InputType.VECTOR_SPARSE);
	//	}

	//vector - vector

	@Test
	public void testVectorVectorMultSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_MULT, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorVectorDivSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_DIV, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorVectorPlusSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_PLUS, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorVectorMinusSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_MINUS, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorVectorMinSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_MIN, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	}
	@Test
	public void testVectorVectorMaxSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_MAX, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorVectorEqualSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_EQUAL, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorVectorNotEqualSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_NOTEQUAL, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorVectorLessSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_LESS, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorVectorLessEqualSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_LESSEQUAL, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorVectorGreaterSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_GREATER, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorVectorGreaterEqualSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_GREATEREQUAL, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorVectorXorSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_XOR, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorVectorBitwAndSparseToSparse() {
		testVectorBinarySparsePrimitive(BinType.VECT_BITWAND, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	}

	//	@Test
	//	public void testVectorVectorMatrixMultSparseToSparse() {
	//		testVectorBinarySparsePrimitive(BinType.VECT_MATRIXMULT, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE);
	//	}

	//unary primitives with sparse intermediates

	@Test
	public void testVectorSqrtSparseToSparse() {
		testVectorUnarySparsePrimitive(UnaryType.VECT_SQRT, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorAbsSparseToSparse() {
		testVectorUnarySparsePrimitive(UnaryType.VECT_ABS, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorRoundSparseToSparse() {
		testVectorUnarySparsePrimitive(UnaryType.VECT_ROUND, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorCeilSparseToSparse() {
		testVectorUnarySparsePrimitive(UnaryType.VECT_CEIL, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorFloorSparseToSparse() {
		testVectorUnarySparsePrimitive(UnaryType.VECT_FLOOR, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorSinSparseToSparse() {
		testVectorUnarySparsePrimitive(UnaryType.VECT_SIN, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorTanSparseToSparse() {
		testVectorUnarySparsePrimitive(UnaryType.VECT_TAN, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorAsinSparseToSparse() {
		testVectorUnarySparsePrimitive(UnaryType.VECT_ASIN, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorAtanSparseToSparse() {
		testVectorUnarySparsePrimitive(UnaryType.VECT_ATAN, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorSinhSparseToSparse() {
		testVectorUnarySparsePrimitive(UnaryType.VECT_SINH, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorTanhSparseToSparse() {
		testVectorUnarySparsePrimitive(UnaryType.VECT_TANH, InputType.VECTOR_SPARSE);
	}

	@Test
	public void testVectorSignSparseToSparse() {
		testVectorUnarySparsePrimitive(UnaryType.VECT_SIGN, InputType.VECTOR_SPARSE);
	}

	private static void testVectorAggPrimitive(UnaryType aggtype, InputType type1)
	{
		try {
			//generate input data
			double sparsity = (type1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock in = MatrixBlock.randOperations(m, n, sparsity, -1, 1, "uniform", 7);
			
			//get vector primitive via reflection
			String tmp = StringUtils.camelize(aggtype.name().split("_")[1]);
			String meName = "vect"+tmp.substring(0, tmp.length()-1);
			Method me = (type1 == InputType.VECTOR_DENSE) ? 
				LibSpoofPrimitives.class.getMethod(meName, new Class[]{double[].class, int.class, int.class}) : 
				LibSpoofPrimitives.class.getMethod(meName, new Class[]{double[].class, int[].class, int.class, int.class, int.class});
			
			for( int i=0; i<m; i++ ) {
				//execute vector primitive via reflection
				Double ret1 = (Double) ((type1 == InputType.VECTOR_DENSE) ? 
					me.invoke(null, in.getDenseBlockValues(), i*n, n) : 
					me.invoke(null, in.getSparseBlock().values(i), in.getSparseBlock().indexes(i), 
						in.getSparseBlock().pos(i), in.getSparseBlock().size(i), n));
				
				//execute comparison operation
				MatrixBlock in2 = in.slice(i, i, 0, n-1, new MatrixBlock());
				Double ret2 = -1d;
				switch( aggtype ) {
					case ROW_SUMS:  ret2 = in2.sum(); break;
					case ROW_MAXS:  ret2 = in2.max(); break;
					case ROW_MINS:  ret2 = in2.min(); break;
					case ROW_MEANS: ret2 = in2.mean(); break;
				}
				
				//compare results
				Assert.assertEquals(ret1, ret2, eps);
			}
		} 
		catch( Exception ex ) {
			throw new RuntimeException(ex);
		}
	}
	
	private static void testVectorUnaryPrimitive(UnaryType utype, InputType type1)
	{
		try {
			//generate input data
			double sparsity = (type1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock in = MatrixBlock.randOperations(m, n, sparsity, -1, 1, "uniform", 7);
			
			//get vector primitive via reflection
			String meName = "vect"+StringUtils.camelize(utype.name().split("_")[1])+"Write";
			Method me = (type1 == InputType.VECTOR_DENSE) ? 
				LibSpoofPrimitives.class.getMethod(meName, new Class[]{double[].class, int.class, int.class}) : 
				LibSpoofPrimitives.class.getMethod(meName, new Class[]{double[].class, int[].class, int.class, int.class, int.class});
			
			for( int i=0; i<m; i++ ) {
				//execute vector primitive via reflection
				double[] ret1 = (double[]) ((type1 == InputType.VECTOR_DENSE) ? 
					me.invoke(null, in.getDenseBlockValues(), i*n, n) : 
					me.invoke(null, in.getSparseBlock().values(i), in.getSparseBlock().indexes(i), 
						in.getSparseBlock().pos(i), in.getSparseBlock().size(i), n));
				
				//execute comparison operation
				String opcode = utype.name().split("_")[1].toLowerCase();
				UnaryOperator uop = new UnaryOperator(Builtin.getBuiltinFnObject(opcode));
				double[] ret2 = DataConverter.convertToDoubleVector(
					in.slice(i, i, 0, n-1, new MatrixBlock())
					.unaryOperations(uop, new MatrixBlock()), false);
				
				//compare results
				TestUtils.compareMatrices(ret1, ret2, eps);
			}
		} 
		catch( Exception ex ) {
			throw new RuntimeException(ex);
		}
	}

	private static void testVectorUnarySparsePrimitive(UnaryType utype, InputType type1)
	{
		try {
			//generate input data
			double sparsity = (type1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock in = MatrixBlock.randOperations(m, n, sparsity, -1, 1, "uniform", 7);

			//get vector primitive via reflection
			String meName = "vect"+StringUtils.camelize(utype.name().split("_")[1])+"Write";
			Method me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{int.class, double[].class, int[].class, int.class, int.class});


			for( int i=0; i<m; i++ ) {
				double[] ret1 = new double[n];
				//execute vector primitive via reflection
				SparseRowVector retX = (SparseRowVector) me.invoke(null, n, in.getSparseBlock().values(i), in.getSparseBlock().indexes(i),
					in.getSparseBlock().pos(i), in.getSparseBlock().size(i));

				int[] indexes = retX.indexes();
				for (int j = 0; j < retX.size(); j++) {
					ret1[indexes[j]] = retX.get(indexes[j]);
				}

				//execute comparison operation
				String opcode = utype.name().split("_")[1].toLowerCase();
				UnaryOperator uop = new UnaryOperator(Builtin.getBuiltinFnObject(opcode));
				double[] ret2 = DataConverter.convertToDoubleVector(
					in.slice(i, i, 0, n-1, new MatrixBlock())
						.unaryOperations(uop, new MatrixBlock()), false);

				//compare results
				TestUtils.compareMatrices(ret1, ret2, eps);
			}
		}
		catch( Exception ex ) {
			throw new RuntimeException(ex);
		}
	}
	
	private static void testVectorBinaryPrimitive(BinType bintype, InputType type1, InputType type2)
	{
		try {
			//generate input data (scalar later derived if needed)
			double sparsityA = (type1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock inA = MatrixBlock.randOperations(m, n, sparsityA, -5, 5, "uniform", 3);
			double sparsityB = (type2 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock inB = MatrixBlock.randOperations(m, n, sparsityB, -5, 5, "uniform", 7);
			
			//get vector primitive via reflection
			String meName = "vect"+StringUtils.camelize(bintype.name().split("_")[1])+"Write";
			final Method me ;
			if( type1==InputType.SCALAR && type2==InputType.VECTOR_DENSE )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{double.class, double[].class, int.class, int.class});
			else if( type1==InputType.VECTOR_DENSE && type2==InputType.SCALAR )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{double[].class, double.class, int.class, int.class});
			else if( type1==InputType.VECTOR_DENSE && type2==InputType.VECTOR_DENSE )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{double[].class, double[].class, int.class, int.class, int.class});
			else if( type1==InputType.VECTOR_SPARSE && type2==InputType.SCALAR )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{double[].class, double.class, int[].class, int.class, int.class, int.class});
			else if( type1==InputType.SCALAR && type2==InputType.VECTOR_SPARSE )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{double.class, double[].class, int[].class, int.class, int.class, int.class});
			else // if( type1==InputType.VECTOR_SPARSE && type2==InputType.VECTOR_DENSE )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{double[].class, double[].class, int[].class, int.class, int.class, int.class, int.class});
			
			for( int i=0; i<m; i++ ) {
				//execute vector primitive via reflection
				double[] ret1 = null;
				if( type1==InputType.SCALAR && type2==InputType.VECTOR_DENSE )
					ret1 = (double[]) me.invoke(null, inA.max(), inB.getDenseBlockValues(), i*n, n);
				else if( type1==InputType.VECTOR_DENSE && type2==InputType.SCALAR )
					ret1 = (double[]) me.invoke(null, inA.getDenseBlockValues(), inB.max(), i*n, n);
				else if( type1==InputType.VECTOR_DENSE && type2==InputType.VECTOR_DENSE )
					ret1 = (double[]) me.invoke(null, inA.getDenseBlockValues(), inB.getDenseBlockValues(), i*n, i*n, n);
				else if( type1==InputType.VECTOR_SPARSE && type2==InputType.SCALAR )
					ret1 = (double[]) me.invoke(null, inA.getSparseBlock().values(i), inB.max(), inA.getSparseBlock().indexes(i), 
						inA.getSparseBlock().pos(i), inA.getSparseBlock().size(i), n);
				else if( type1==InputType.SCALAR && type2==InputType.VECTOR_SPARSE )
					ret1 = (double[]) me.invoke(null, inA.max(), inB.getSparseBlock().values(i), 
						inB.getSparseBlock().indexes(i), inB.getSparseBlock().pos(i), inB.getSparseBlock().size(i), n);
				else if( type1==InputType.VECTOR_SPARSE && type2==InputType.VECTOR_DENSE )
					ret1 = (double[]) me.invoke(null, inA.getSparseBlock().values(i), inB.getDenseBlockValues(), 
						inA.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), i*n, inA.getSparseBlock().size(i), n);
				
				//execute comparison operation
				String opcode = OpOp2.valueOf(bintype.name().split("_")[1]).toString();
				MatrixBlock in1 = inA.slice(i, i, 0, n-1, new MatrixBlock());
				MatrixBlock in2 = inB.slice(i, i, 0, n-1, new MatrixBlock());
				double[] ret2 = null;
				if( type1 == InputType.SCALAR ) {
					ScalarOperator bop = InstructionUtils.parseScalarBinaryOperator(opcode, true);
					bop = bop.setConstant(inA.max());
					ret2 = DataConverter.convertToDoubleVector(
						in2.scalarOperations(bop, new MatrixBlock()), false);
				}
				else if( type2 == InputType.SCALAR ) {
					ScalarOperator bop = InstructionUtils.parseScalarBinaryOperator(opcode, false);
					bop = bop.setConstant(inB.max());
					ret2 = DataConverter.convertToDoubleVector(
						in1.scalarOperations(bop, new MatrixBlock()), false);
				}
				else { //vector-vector
					BinaryOperator bop = InstructionUtils.parseBinaryOperator(opcode);
					ret2 = DataConverter.convertToDoubleVector(
						in1.binaryOperations(bop, in2, new MatrixBlock()), false);
				}
				
				//compare results
				TestUtils.compareMatrices(ret1, ret2, eps);
			}
		} 
		catch( Exception ex ) {
			throw new RuntimeException(ex);
		}
	}

	private static void testVectorBinarySparsePrimitive(BinType bintype, InputType type1, InputType type2) {
		try {
			//generate input data (scalar later derived if needed)
			double sparsityA = (type1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock inA = MatrixBlock.randOperations(m, n, sparsityA, -5, 5, "uniform", 3);
			double sparsityB = (type2 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
			MatrixBlock inB = MatrixBlock.randOperations(m, n, sparsityB, -5, 5, "uniform", 7);

			boolean sparse = getOutputType(bintype);
			int testType = getTestType(bintype);

			//get vector primitive via reflection
			String meName = "vect"+StringUtils.camelize(bintype.name().split("_")[1])+"Write";
			final Method me ;
			if( type1==InputType.VECTOR_SPARSE && type2==InputType.SCALAR )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{int.class, double[].class, double.class, int[].class, int.class, int.class});
			else if( type1==InputType.SCALAR && type2==InputType.VECTOR_SPARSE )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{int.class, double.class, double[].class, int[].class, int.class, int.class});
			else //if( type1==InputType.VECTOR_SPARSE && type2==InputType.VECTOR_SPARSE )
				me = LibSpoofPrimitives.class.getMethod(meName, new Class[]{int.class, double[].class, double[].class, int[].class, int[].class, int.class, int.class, int.class, int.class});

			for( int i=0; i<m; i++ ) {
				//execute vector primitive via reflection
				double[] ret1 = new double[n];
				SparseRowVector retX = null;
				if( type1==InputType.VECTOR_SPARSE && type2==InputType.SCALAR )
					if(testType >= 0 && i == m-1) {
						if(testType == 0) {
							retX = (SparseRowVector) me.invoke(null, n, inA.getSparseBlock().values(i), 0, inA.getSparseBlock().indexes(i),
								inA.getSparseBlock().pos(i), inA.getSparseBlock().size(i));
						} else {
							retX = (SparseRowVector) me.invoke(null, n, inA.getSparseBlock().values(i), inB.min(), inA.getSparseBlock().indexes(i),
								inA.getSparseBlock().pos(i), inA.getSparseBlock().size(i));
						}
					} else {
						retX = (SparseRowVector) me.invoke(null, n, inA.getSparseBlock().values(i), inB.max(), inA.getSparseBlock().indexes(i),
							inA.getSparseBlock().pos(i), inA.getSparseBlock().size(i));
					}
				else if( type1==InputType.SCALAR && type2==InputType.VECTOR_SPARSE )
					if(testType >= 0 && i == m-1) {
						if(testType == 0) {
							retX = (SparseRowVector) me.invoke(null, n, 0, inB.getSparseBlock().values(i),
								inB.getSparseBlock().indexes(i), inB.getSparseBlock().pos(i), inB.getSparseBlock().size(i));
						} else {
							retX = (SparseRowVector) me.invoke(null, n, inA.min(), inB.getSparseBlock().values(i),
								inB.getSparseBlock().indexes(i), inB.getSparseBlock().pos(i), inB.getSparseBlock().size(i));
						}
					} else {
						retX = (SparseRowVector) me.invoke(null, n, inA.max(), inB.getSparseBlock().values(i),
							inB.getSparseBlock().indexes(i), inB.getSparseBlock().pos(i), inB.getSparseBlock().size(i));
					}
				else if( type1==InputType.VECTOR_SPARSE && type2==InputType.VECTOR_SPARSE )
					if(sparse)
						retX = (SparseRowVector) me.invoke(null, n, inA.getSparseBlock().values(i), inB.getSparseBlock().values(i),
							inA.getSparseBlock().indexes(i), inB.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), inB.getSparseBlock().pos(i), inA.getSparseBlock().size(i), inB.getSparseBlock().size(i));
					else
						ret1 = (double[]) me.invoke(null, n, inA.getSparseBlock().values(i), inB.getSparseBlock().values(i),
							inA.getSparseBlock().indexes(i), inB.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), inB.getSparseBlock().pos(i), inA.getSparseBlock().size(i), inB.getSparseBlock().size(i));

				if(sparse) {
					int[] indexes = retX.indexes();
					for (int j = 0; j < retX.size(); j++) {
						ret1[indexes[j]] = retX.get(indexes[j]);
					}
				}

				//execute comparison operation
				String opcode = OpOp2.valueOf(bintype.name().split("_")[1]).toString();
				MatrixBlock in1 = inA.slice(i, i, 0, n-1, new MatrixBlock());
				MatrixBlock in2 = inB.slice(i, i, 0, n-1, new MatrixBlock());
				double[] ret2 = null;
				if( type1 == InputType.SCALAR ) {
					ScalarOperator bop = InstructionUtils.parseScalarBinaryOperator(opcode, true);
					bop = bop.setConstant(testType >= 0 && i == m-1 ? testType == 0 ? 0 : inA.min() : inA.max());
					ret2 = DataConverter.convertToDoubleVector(
						in2.scalarOperations(bop, new MatrixBlock()), false);
				}
				else if( type2 == InputType.SCALAR ) {
					ScalarOperator bop = InstructionUtils.parseScalarBinaryOperator(opcode, false);
					bop = bop.setConstant(testType >= 0 && i == m-1 ? testType == 0 ? 0 : inB.min() : inB.max());
					ret2 = DataConverter.convertToDoubleVector(
						in1.scalarOperations(bop, new MatrixBlock()), false);
				}
				else { //vector-vector
					BinaryOperator bop = InstructionUtils.parseBinaryOperator(opcode);
					ret2 = DataConverter.convertToDoubleVector(
						in1.binaryOperations(bop, in2, new MatrixBlock()), false);
				}

				//compare results
				TestUtils.compareMatrices(ret2, ret1, eps);
			}
		}
		catch( Exception ex ) {
			throw new RuntimeException(ex);
		}
	}


	/**
	 * @param type
	 * @return {@code true}, when the matching method has a sparse return
	 */
	private static boolean getOutputType(BinType type) {
		switch(type) {
			case VECT_EQUAL:
			case VECT_LESSEQUAL:
			case VECT_GREATEREQUAL:
				return false;
			default:
				return true;
		}
	}

	/**
	 * @param type
	 * @return returns {@code -1}, for normal testing;<br>
	 *         returns {@code 0}, for testing with 0 and non-zeros;<br>
	 *         returns {@code 1}, for testing with negative and positive numbers;
	 */
	private static int getTestType(BinType type) {
		switch(type) {
			case VECT_DIV_SCALAR:
			case VECT_EQUAL_SCALAR:
			case VECT_NOTEQUAL_SCALAR:
			case VECT_XOR_SCALAR:
				//			case VECT_POW_SCALAR:
				return 0;
			case VECT_GREATER_SCALAR:
			case VECT_GREATEREQUAL_SCALAR:
			case VECT_LESS_SCALAR:
			case VECT_LESSEQUAL_SCALAR:
			case VECT_MIN_SCALAR:
			case VECT_MAX_SCALAR:
				return 1;
			default:
				return -1;
		}
	}
}
