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

package org.apache.sysds.hops.codegen.cplan.java;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.hops.codegen.cplan.CodeTemplate;

public class Binary extends CodeTemplate {

	public String getTemplate(BinType type, boolean sparseLhs, boolean sparseRhs,
		boolean scalarVector, boolean scalarInput, boolean vectorVector)
	{
		switch (type) {
			case ROWMAXS_VECTMULT:
				return sparseLhs ? "\tdouble %TMP% = LibSpoofPrimitives.rowMaxsVectMult(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, alen);\n" :
						"\tdouble %TMP% = LibSpoofPrimitives.rowMaxsVectMult(%IN1%, %IN2%, %POS1%, %POS2%, %LEN%);\n";
			case DOT_PRODUCT:
				return sparseLhs ? "    double %TMP% = LibSpoofPrimitives.dotProduct(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, alen);\n" :
						"    double %TMP% = LibSpoofPrimitives.dotProduct(%IN1%, %IN2%, %POS1%, %POS2%, %LEN%);\n";
			case VECT_MATRIXMULT:
				return sparseLhs ? "	double[] %TMP% = LibSpoofPrimitives.vectMatrixMult(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, alen, len);\n" :
						"    double[] %TMP% = LibSpoofPrimitives.vectMatrixMult(%IN1%, %IN2%, %POS1%, %POS2%, %LEN%);\n";
			case VECT_OUTERMULT_ADD:
				return  sparseLhs ? "    LibSpoofPrimitives.vectOuterMultAdd(%IN1v%, %IN2%, %OUT%, %IN1i%, %POS1%, %POS2%, %POSOUT%, alen, %LEN1%, %LEN2%);\n" :
						sparseRhs ? "    LibSpoofPrimitives.vectOuterMultAdd(%IN1%, %IN2v%, %OUT%, %POS1%, %IN2i%, %POS2%, %POSOUT%, alen, %LEN1%, %LEN2%);\n" :
								"    LibSpoofPrimitives.vectOuterMultAdd(%IN1%, %IN2%, %OUT%, %POS1%, %POS2%, %POSOUT%, %LEN1%, %LEN2%);\n";

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
				if( scalarVector )
					return sparseLhs ? "    LibSpoofPrimitives.vect"+vectName+"Add(%IN1%, %IN2v%, %OUT%, %IN2i%, %POS2%, %POSOUT%, alen, %LEN%);\n" :
							"    LibSpoofPrimitives.vect"+vectName+"Add(%IN1%, %IN2%, %OUT%, %POS2%, %POSOUT%, %LEN%);\n";
				else
					return sparseLhs ? "    LibSpoofPrimitives.vect"+vectName+"Add(%IN1v%, %IN2%, %OUT%, %IN1i%, %POS1%, %POSOUT%, alen, %LEN%);\n" :
							"    LibSpoofPrimitives.vect"+vectName+"Add(%IN1%, %IN2%, %OUT%, %POS1%, %POSOUT%, %LEN%);\n";
			}

			//vector-scalar operations
			case VECT_MULT_SCALAR:
			case VECT_POW_SCALAR: {
				String vectName = type.getVectorPrimitiveName();
				if( scalarVector )
					return sparseRhs ? "    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2v%, %IN2i%, %POS2%, alen, %LEN%);\n" :
						"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2%, %POS2%, %LEN%);\n";
				else if(DMLScript.SPARSE_INTERMEDIATE) {
					return sparseLhs ? "    SparseRowVector %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%LEN%, %IN1v%, %IN2%, %IN1i%, %POS1%, %LEN1%);\n" :
						"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2%, %POS1%, %LEN%);\n";
				} else {
					return sparseLhs ? "    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1v%, %IN2%, %IN1i%, %POS1%, alen, %LEN%);\n" :
						"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2%, %POS1%, %LEN%);\n";
				}
			}
			case VECT_DIV_SCALAR:
			case VECT_XOR_SCALAR:
			case VECT_MIN_SCALAR:
			case VECT_MAX_SCALAR:
			case VECT_EQUAL_SCALAR:
			case VECT_NOTEQUAL_SCALAR:
			case VECT_LESS_SCALAR:
			case VECT_LESSEQUAL_SCALAR:
			case VECT_GREATER_SCALAR:
			case VECT_GREATEREQUAL_SCALAR:
			case VECT_BITWAND_SCALAR: {
				String vectName = type.getVectorPrimitiveName();
				if(scalarVector) {
					if(sparseRhs)
						return DMLScript.SPARSE_INTERMEDIATE ? "    SparseRowVector %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%LEN%, %IN1%, %IN2v%, %IN2i%, %POS2%, %LEN1%);\n" :
							"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2v%, %IN2i%, %POS2%, alen, %LEN%);\n";
				} else {
					if(sparseLhs)
						return DMLScript.SPARSE_INTERMEDIATE ? "    SparseRowVector %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%LEN%, %IN1v%, %IN2%, %IN1i%, %POS1%, %LEN1%);\n" :
							"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1v%, %IN2%, %IN1i%, %POS1%, alen, %LEN%);\n";
				}
				return 	"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2%, %POS1%, %LEN%);\n";
			}
			case VECT_MINUS_SCALAR:
			case VECT_PLUS_SCALAR: {
				String vectName = type.getVectorPrimitiveName();
				if( scalarVector )
					return sparseRhs ? "    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2v%, %IN2i%, %POS2%, alen, %LEN%);\n" :
							"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2%, %POS2%, %LEN%);\n";
				else
					return sparseLhs ? "    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1v%, %IN2%, %IN1i%, %POS1%, alen, %LEN%);\n" :
							"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2%, %POS1%, %LEN%);\n";
			}

			case VECT_CBIND:
				if( scalarInput )
					return  "    double[] %TMP% = LibSpoofPrimitives.vectCbindWrite(%IN1%, %IN2%);\n";
				else if( !vectorVector )
					return sparseLhs ?
							"    double[] %TMP% = LibSpoofPrimitives.vectCbindWrite(%IN1v%, %IN2%, %IN1i%, %POS1%, alen, %LEN%);\n" :
							"    double[] %TMP% = LibSpoofPrimitives.vectCbindWrite(%IN1%, %IN2%, %POS1%, %LEN%);\n";
				else //vect/vect
					return sparseLhs ?
						"    double[] %TMP% = LibSpoofPrimitives.vectCbindWrite(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, alen, %LEN1%, %LEN2%);\n" :
						"    double[] %TMP% = LibSpoofPrimitives.vectCbindWrite(%IN1%, %IN2%, %POS1%, %POS2%, %LEN1%, %LEN2%);\n";

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
			case VECT_NOTEQUAL:
			case VECT_LESS:
			case VECT_GREATER:{
				String vectName = type.getVectorPrimitiveName();
				if(DMLScript.SPARSE_INTERMEDIATE && sparseLhs && sparseRhs) {
					return "    SparseRowVector %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%LEN%, %IN1v%, %IN2v%, %IN1i%, %IN2i%, %POS1%, %POS2%, %SLEN1%, %SLEN2%);\n";
				} else {
					return sparseLhs ?
						"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, alen, %LEN%);\n" :
						sparseRhs ?
							"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2v%, %POS1%, %IN2i%, %POS2%, alen, %LEN%);\n" :
							"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2%, %POS1%, %POS2%, %LEN%);\n";
				}
			}
			case VECT_EQUAL:
			case VECT_LESSEQUAL:
			case VECT_GREATEREQUAL: {
				String vectName = type.getVectorPrimitiveName();
				if(DMLScript.SPARSE_INTERMEDIATE && sparseLhs && sparseRhs) {
					return "    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%LEN%, %IN1v%, %IN2v%, %IN1i%, %IN2i%, %POS1%, %POS2%, %SLEN1%, %SLEN2%);\n";
				} else {
					return sparseLhs ?
						"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, alen, %LEN%);\n" :
						sparseRhs ?
							"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2v%, %POS1%, %IN2i%, %POS2%, alen, %LEN%);\n" :
							"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2%, %POS1%, %POS2%, %LEN%);\n";
				}
			}
			//scalar-scalar operations
			case MULT:
				return "    double %TMP% = %IN1% * %IN2%;\n";

			case DIV:
				return "    double %TMP% = %IN1% / %IN2%;\n";
			case PLUS:
				return "    double %TMP% = %IN1% + %IN2%;\n";
			case MINUS:
				return "    double %TMP% = %IN1% - %IN2%;\n";
			case MODULUS:
				return "    double %TMP% = LibSpoofPrimitives.mod(%IN1%, %IN2%);\n";
			case INTDIV:
				return "    double %TMP% = LibSpoofPrimitives.intDiv(%IN1%, %IN2%);\n";
			case LESS:
				return "    double %TMP% = (%IN1% < %IN2%) ? 1 : 0;\n";
			case LESSEQUAL:
				return "	double %TMP% = (%IN1% <= %IN2%) ? 1 : 0;\n";
			case GREATER:
				return "    double %TMP% = (%IN1% > %IN2%) ? 1 : 0;\n";
			case GREATEREQUAL:
				return "    double %TMP% = (%IN1% >= %IN2%) ? 1 : 0;\n";
			case EQUAL:
				return "    double %TMP% = (%IN1% == %IN2%) ? 1 : 0;\n";
			case NOTEQUAL:
				return "    double %TMP% = (%IN1% != %IN2%) ? 1 : 0;\n";

			case MIN:
				return "    double %TMP% = Math.min(%IN1%, %IN2%);\n";
			case MAX:
				return "    double %TMP% = Math.max(%IN1%, %IN2%);\n";
			case LOG:
				return "    double %TMP% = Math.log(%IN1%)/Math.log(%IN2%);\n";
			case LOG_NZ:
				return "    double %TMP% = (%IN1% == 0) ? 0 : Math.log(%IN1%)/Math.log(%IN2%);\n";
			case POW:
				return "    double %TMP% = Math.pow(%IN1%, %IN2%);\n";
			case MINUS1_MULT:
				return "    double %TMP% = 1 - %IN1% * %IN2%;\n";
			case MINUS_NZ:
				return "    double %TMP% = (%IN1% != 0) ? %IN1% - %IN2% : 0;\n";
			case XOR:
				return "    double %TMP% = ( (%IN1% != 0) != (%IN2% != 0) ) ? 1 : 0;\n";
			case BITWAND:
				return "    double %TMP% = LibSpoofPrimitives.bwAnd(%IN1%, %IN2%);\n";
			case SEQ_RIX:
				return "    double %TMP% = %IN1% + grix * %IN2%;\n"; //0-based global rix

			default:
				throw new RuntimeException("Invalid binary type: "+this.toString());
		}
	}
}
