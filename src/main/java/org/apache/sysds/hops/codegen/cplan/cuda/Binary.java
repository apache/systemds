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

package org.apache.sysds.hops.codegen.cplan.cuda;

import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CodeTemplate;

import static org.apache.sysds.runtime.matrix.data.LibMatrixNative.isSinglePrecision;

public class Binary extends CodeTemplate
{
	@Override
	public String getTemplate(CNodeBinary.BinType type, boolean sparseLhs, boolean sparseRhs,
		boolean scalarVector, boolean scalarInput, boolean vectorVector)
	{
		if(type == CNodeBinary.BinType.VECT_CBIND) {
			if(scalarInput)
				return "\t\tVector<T>& %TMP% = vectCbindWrite(%IN1%, %IN2%, this);\n";
			else if (!vectorVector)
				return sparseLhs ? 
					"\t\tVector<T>& %TMP% = vectCbindWrite(%IN1v%, %IN2%, %IN1i%, %POS1%, alen, %LEN%, this);\n" :
					"\t\tVector<T>& %TMP% = vectCbindWrite(%IN1%, %IN2%, %POS1%, %LEN%, this);\n";
			else //vect/vect
				return sparseLhs ?
					"\t\tVector<T>& %TMP% = vectCbindWrite(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, alen, %LEN1%, %LEN2%, this);\n" :
					"\t\tVector<T>& %TMP% = vectCbindWrite(%IN1%, %IN2%, %POS1%, %POS2%, %LEN1%, %LEN2%, this);\n";
		}
		
		if(isSinglePrecision()) {
			switch(type) {
				case DOT_PRODUCT:
					return sparseLhs ? "	T %TMP% = LibSpoofPrimitives.dotProduct(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, alen);\n" : "	T %TMP% = LibSpoofPrimitives.dotProduct(%IN1%, %IN2%, %POS1%, %POS2%, %LEN%);\n";
				case VECT_MATRIXMULT:
					return sparseLhs ? "	T[] %TMP% = LibSpoofPrimitives.vectMatrixMult(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, alen, len);\n" : "	T[] %TMP% = LibSpoofPrimitives.vectMatrixMult(%IN1%, %IN2%, %POS1%, %POS2%, %LEN%);\n";
				case VECT_OUTERMULT_ADD:
					return sparseLhs ? "	LibSpoofPrimitives.vectOuterMultAdd(%IN1v%, %IN2%, %OUT%, %IN1i%, %POS1%, %POS2%, %POSOUT%, alen, %LEN1%, %LEN2%);\n" : sparseRhs ? "	LibSpoofPrimitives.vectOuterMultAdd(%IN1%, %IN2v%, %OUT%, %POS1%, %IN2i%, %POS2%, %POSOUT%, alen, %LEN1%, %LEN2%);\n" : "	LibSpoofPrimitives.vectOuterMultAdd(%IN1%, %IN2%, %OUT%, %POS1%, %POS2%, %POSOUT%, %LEN1%, %LEN2%);\n";

				//vector-scalar-add operations
				case VECT_MULT_ADD:
				case VECT_DIV_ADD:
				case VECT_MINUS_ADD:
				case VECT_PLUS_ADD:
				case VECT_POW_ADD:
				case VECT_XOR_ADD:
				case VECT_MIN_ADD:
				case VECT_MAX_ADD:
				case VECT_EQUAL_ADD:
				case VECT_NOTEQUAL_ADD:
				case VECT_LESS_ADD:
				case VECT_LESSEQUAL_ADD:
				case VECT_GREATER_ADD:
				case VECT_GREATEREQUAL_ADD:
				case VECT_CBIND_ADD: {
					String vectName = type.getVectorPrimitiveName();
					if(scalarVector)
						return sparseLhs ? "	LibSpoofPrimitives.vect" + vectName + "Add(%IN1%, %IN2v%, %OUT%, %IN2i%, %POS2%, %POSOUT%, alen, %LEN%);\n" : "	LibSpoofPrimitives.vect" + vectName + "Add(%IN1%, %IN2%, %OUT%, %POS2%, %POSOUT%, %LEN%);\n";
					else
						return sparseLhs ? "	LibSpoofPrimitives.vect" + vectName + "Add(%IN1v%, %IN2%, %OUT%, %IN1i%, %POS1%, %POSOUT%, alen, %LEN%);\n" : "	LibSpoofPrimitives.vect" + vectName + "Add(%IN1%, %IN2%, %OUT%, %POS1%, %POSOUT%, %LEN%);\n";
				}

				//vector-scalar operations
				case VECT_MULT_SCALAR:
				case VECT_DIV_SCALAR:
				case VECT_MINUS_SCALAR:
				case VECT_PLUS_SCALAR:
				case VECT_POW_SCALAR:
				case VECT_XOR_SCALAR:
				case VECT_BITWAND_SCALAR:
				case VECT_MIN_SCALAR:
				case VECT_MAX_SCALAR:
				case VECT_EQUAL_SCALAR:
				case VECT_NOTEQUAL_SCALAR:
				case VECT_LESS_SCALAR:
				case VECT_LESSEQUAL_SCALAR:
				case VECT_GREATER_SCALAR:
				case VECT_GREATEREQUAL_SCALAR: {
					String vectName = type.getVectorPrimitiveName();
					if(scalarVector)
						return sparseRhs ? "	T[] %TMP% = LibSpoofPrimitives.vect" + vectName + "Write(%IN1%, %IN2v%, %IN2i%, %POS2%, alen, %LEN%);\n" : "	T[] %TMP% = LibSpoofPrimitives.vect" + vectName + "Write(%IN1%, %IN2%, %POS2%, %LEN%);\n";
					else
						return sparseLhs ? "	T[] %TMP% = LibSpoofPrimitives.vect" + vectName + "Write(%IN1v%, %IN2%, %IN1i%, %POS1%, alen, %LEN%);\n" : "	T[] %TMP% = LibSpoofPrimitives.vect" + vectName + "Write(%IN1%, %IN2%, %POS1%, %LEN%);\n";
				}
				//vector-vector operations
				case VECT_MULT:
				case VECT_DIV:
				case VECT_MINUS:
				case VECT_PLUS:
				case VECT_XOR:
				case VECT_BITWAND:
				case VECT_BIASADD:
				case VECT_BIASMULT:
				case VECT_MIN:
				case VECT_MAX:
				case VECT_EQUAL:
				case VECT_NOTEQUAL:
				case VECT_LESS:
				case VECT_LESSEQUAL:
				case VECT_GREATER:
				case VECT_GREATEREQUAL: {
					String vectName = type.getVectorPrimitiveName();
					return sparseLhs ? "	T[] %TMP% = LibSpoofPrimitives.vect" + vectName + "Write(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, alen, %LEN%);\n" : sparseRhs ? "	T[] %TMP% = LibSpoofPrimitives.vect" + vectName + "Write(%IN1%, %IN2v%, %POS1%, %IN2i%, %POS2%, alen, %LEN%);\n" : "	T[] %TMP% = LibSpoofPrimitives.vect" + vectName + "Write(%IN1%, %IN2%, %POS1%, %POS2%, %LEN%);\n";
				}

				//scalar-scalar operations
				case MULT:
					return "	T %TMP% = %IN1% * %IN2%;\n";
				case DIV:
					return "	T %TMP% = %IN1% / %IN2%;\n";
				case PLUS:
					return "	T %TMP% = %IN1% + %IN2%;\n";
				case MINUS:
					return "	T %TMP% = %IN1% - %IN2%;\n";
				case MODULUS:
					return "	T %TMP% = modulus(%IN1%, %IN2%);\n";
				case INTDIV:
					return "	T %TMP% = intDiv(%IN1%, %IN2%);\n";
				case LESS:
					return "	T %TMP% = (%IN1% < %IN2%) ? 1 : 0;\n";
				case LESSEQUAL:
					return "	T %TMP% = (%IN1% <= %IN2%) ? 1 : 0;\n";
				case GREATER:
					return "	T %TMP% = (%IN1% > %IN2%) ? 1 : 0;\n";
				case GREATEREQUAL:
					return "	T %TMP% = (%IN1% >= %IN2%) ? 1 : 0;\n";
				case EQUAL:
					return "	T %TMP% = (%IN1% == %IN2%) ? 1 : 0;\n";
				case NOTEQUAL:
					return "	T %TMP% = (%IN1% != %IN2%) ? 1 : 0;\n";

				case MIN:
					return "	T %TMP% = fminf(%IN1%, %IN2%);\n";
				case MAX:
					return "	T %TMP% = fmaxf(%IN1%, %IN2%);\n";
				case LOG:
					return "	T %TMP% = logf(%IN1%)/Math.log(%IN2%);\n";
				case LOG_NZ:
					return "	T %TMP% = (%IN1% == 0) ? 0 : logf(%IN1%) / logf(%IN2%);\n";
				case POW:
					return "	T %TMP% = powf(%IN1%, %IN2%);\n";
				case MINUS1_MULT:
					return "	T %TMP% = 1 - %IN1% * %IN2%;\n";
				case MINUS_NZ:
					return "	T %TMP% = (%IN1% != 0) ? %IN1% - %IN2% : 0;\n";
				case XOR:
					return "	T %TMP% = ( (%IN1% != 0) != (%IN2% != 0) ) ? 1.0f : 0.0f;\n";
				case BITWAND:
					return "	T %TMP% = bwAnd(%IN1%, %IN2%);\n";
				case SEQ_RIX:
					return "	T %TMP% = %IN1% + grix * %IN2%;\n"; //0-based global rix

				default:
					throw new RuntimeException("Invalid binary type: " + this.toString());
			}
		}
		else {
			switch(type) {
				case DOT_PRODUCT:
//					return sparseLhs ? "	T %TMP% = LibSpoofPrimitives.dotProduct(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, alen);\n" : "	T %TMP% = LibSpoofPrimitives.dotProduct(%IN1%, %IN2%, %POS1%, %POS2%, %LEN%);\n";
//					return sparseLhs ? "		T %TMP% = dotProduct(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, alen);\n" : "		T %TMP% = dotProduct(%IN1%, %IN2%, %POS1%, %POS2%, %LEN%);\n	printf(\"dot=%f, bid=%d, tid=%d\\n\",TMP7,blockIdx.x, threadIdx.x);\n	__syncthreads();\n";
					return sparseLhs ? "		T %TMP% = dotProduct(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, alen);\n" : "		T %TMP% = dotProduct(%IN1%, %IN2%, %POS1%, static_cast<uint32_t>(%POS2%), %LEN%);\n";
				
				case VECT_MATRIXMULT:
					return sparseLhs ? "	T[] %TMP% = vectMatrixMult(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, alen, len);\n" : "		Vector<T>& %TMP% = vectMatrixMult(%IN1%, %IN2%, %POS1%, static_cast<uint32_t>(%POS2%), %LEN%, this);\n";
				case VECT_OUTERMULT_ADD:
					return sparseLhs ? "	LibSpoofPrimitives.vectOuterMultAdd(%IN1v%, %IN2%, %OUT%, %IN1i%, %POS1%, %POS2%, %POSOUT%, alen, %LEN1%, %LEN2%);\n" : sparseRhs ? "	LibSpoofPrimitives.vectOuterMultAdd(%IN1%, %IN2v%, %OUT%, %POS1%, %IN2i%, %POS2%, %POSOUT%, alen, %LEN1%, %LEN2%);\n" : "\t\tvectOuterMultAdd(%IN1%, %IN2%, %OUT%, %POS1%, %POS2%, %POSOUT%, %LEN1%, %LEN2%);\n";

				//vector-scalar-add operations
				case VECT_MULT_ADD:
				case VECT_DIV_ADD:
				case VECT_MINUS_ADD:
				case VECT_PLUS_ADD:
				case VECT_POW_ADD:
				case VECT_XOR_ADD:
				case VECT_MIN_ADD:
				case VECT_MAX_ADD:
				case VECT_EQUAL_ADD:
				case VECT_NOTEQUAL_ADD:
				case VECT_LESS_ADD:
				case VECT_LESSEQUAL_ADD:
				case VECT_GREATER_ADD:
				case VECT_GREATEREQUAL_ADD:
				case VECT_CBIND_ADD: {
					String vectName = type.getVectorPrimitiveName();
					if(scalarVector)
						return sparseLhs ? "\t\tvect" + vectName + "Add(%IN1%, %IN2v%, %OUT%, %IN2i%, %POS2%, %POSOUT%, alen, %LEN%);\n" : "\t\tvect" + vectName + "Add(%IN1%, %IN2%, %OUT%, %POS2%, %POSOUT%, %LEN%);\n";
					else
						return sparseLhs ? "\t\tvect" + vectName + "Add(%IN1v%, %IN2%, %OUT%, %IN1i%, %POS1%, %POSOUT%, alen, %LEN%);\n" : "\t\tvect" + vectName + "Add(%IN1%, %IN2%, %OUT%, %POS1%, static_cast<uint32_t>(%POSOUT%), %LEN%);\n";
				}

				//vector-scalar operations
				case VECT_MULT_SCALAR:
				case VECT_DIV_SCALAR:
				case VECT_MINUS_SCALAR:
				case VECT_PLUS_SCALAR:
				case VECT_POW_SCALAR:
				case VECT_XOR_SCALAR:
				case VECT_BITWAND_SCALAR:
				case VECT_MIN_SCALAR:
				case VECT_MAX_SCALAR:
				case VECT_EQUAL_SCALAR:
				case VECT_NOTEQUAL_SCALAR:
				case VECT_LESS_SCALAR:
				case VECT_LESSEQUAL_SCALAR:
				case VECT_GREATER_SCALAR:
				case VECT_GREATEREQUAL_SCALAR: {
					String vectName = type.getVectorPrimitiveName();
					if(scalarVector)
						return sparseRhs ? "	T[] %TMP% = LibSpoofPrimitives.vect" + vectName + "Write(%IN1%, %IN2v%, %IN2i%, %POS2%, alen, %LEN%);\n" : "		Vector<T>& %TMP% = vect" + vectName + "Write(%IN1%, %IN2%, %POS2%, %LEN%, this);\n";
					else
//						return sparseLhs ? "	T[] %TMP% = LibSpoofPrimitives.vect" + vectName + "Write(%IN1v%, %IN2%, %IN1i%, %POS1%, alen, %LEN%);\n" : "	T[] %TMP% = LibSpoofPrimitives.vect" + vectName + "Write(%IN1%, %IN2%, %POS1%, %LEN%);\n";
						return sparseLhs ? "		Vector<T>& %TMP% = vect" + vectName + "Write(%IN1v%, %IN2%, %IN1i%, %POS1%, alen, %LEN%, this);\n" : "		Vector<T>& %TMP% = vect" + vectName + "Write(%IN1%, %IN2%, static_cast<uint32_t>(%POS1%), %LEN%, this);\n";
				}
					
				//vector-vector operations
				case VECT_MULT:
				case VECT_DIV:
				case VECT_MINUS:
				case VECT_PLUS:
				case VECT_XOR:
				case VECT_BITWAND:
				case VECT_BIASADD:
				case VECT_BIASMULT:
				case VECT_MIN:
				case VECT_MAX:
				case VECT_EQUAL:
				case VECT_NOTEQUAL:
				case VECT_LESS:
				case VECT_LESSEQUAL:
				case VECT_GREATER:
				case VECT_GREATEREQUAL: {
					String vectName = type.getVectorPrimitiveName();
					return sparseLhs ? "		Vector<T>& %TMP% = vect" + vectName + "Write(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, " +
							"alen, %LEN%);\n" : sparseRhs ? "		Vector<T>& %TMP% = vect" + vectName + "Write(%IN1%, %IN2v%, " +
							"%POS1%, %IN2i%, %POS2%, alen, %LEN%);\n" : "		Vector<T>& %TMP% = vect" + vectName + 
						"Write(%IN1%, %IN2%, static_cast<uint32_t>(%POS1%), static_cast<uint32_t>(%POS2%), %LEN%, this);\n";
				}

				//scalar-scalar operations
				case MULT:
					return "		T %TMP% = %IN1% * %IN2%;\n";
				case DIV:
					return "	T %TMP% = %IN1% / %IN2%;\n";
				case PLUS:
					return "		T %TMP% = %IN1% + %IN2%;\n";
				case MINUS:
					return "	T %TMP% = %IN1% - %IN2%;\n";
				case MODULUS:
					return "	T %TMP% = modulus(%IN1%, %IN2%);\n";
				case INTDIV:
					return "	T %TMP% = intDiv(%IN1%, %IN2%);\n";
				case LESS:
					return "	T %TMP% = (%IN1% < %IN2%) ? 1.0 : 0.0;\n";
				case LESSEQUAL:
					return "	T %TMP% = (%IN1% <= %IN2%) ? 1.0 : 0.0;\n";
				case GREATER:
					return "	T %TMP% = (%IN1% > (%IN2% + EPSILON)) ? 1.0 : 0.0;\n";
				case GREATEREQUAL:
					return "	T %TMP% = (%IN1% >= %IN2%) ? 1.0 : 0.0;\n";
				case EQUAL:
					return "	T %TMP% = (%IN1% == %IN2%) ? 1.0 : 0.0;\n";
				case NOTEQUAL:
					return "	T %TMP% = (%IN1% != %IN2%) ? 1.0 : 0.0;\n";

				case MIN:
					return "	T %TMP% = min(%IN1%, %IN2%);\n";
				case MAX:
					return "	T %TMP% = max(%IN1%, %IN2%);\n";
				case LOG:
					return "	T %TMP% = log(%IN1%)/Math.log(%IN2%);\n";
				case LOG_NZ:
					return "	T %TMP% = (%IN1% == 0) ? 0 : log(%IN1%) / log(%IN2%);\n";
				case POW:
					return "	T %TMP% = pow(%IN1%, %IN2%);\n";
				case MINUS1_MULT:
					return "	T %TMP% = 1 - %IN1% * %IN2%;\n";
				case MINUS_NZ:
					return "	T %TMP% = (%IN1% != 0) ? %IN1% - %IN2% : 0;\n";
				case XOR:
//					return "	T %TMP% = ( (%IN1% != 0.0) != (%IN2% != 0.0) ) ? 1.0 : 0.0;\n";
					return "	T %TMP% = ( (%IN1% < EPSILON) != (%IN2% < EPSILON) ) ? 1.0 : 0.0;\n";
				case BITWAND:
					return "	T %TMP% = bwAnd(%IN1%, %IN2%);\n";
				case SEQ_RIX:
					return "		T %TMP% = %IN1% + grix * %IN2%;\n"; //0-based global rix

				default:
					throw new RuntimeException("Invalid binary type: " + this.toString());
			}
		}
	}
}
