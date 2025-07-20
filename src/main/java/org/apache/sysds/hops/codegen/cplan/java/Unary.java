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

import org.apache.commons.lang3.StringUtils;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysds.hops.codegen.cplan.CodeTemplate;

public class Unary extends CodeTemplate {
	@Override
	public String getTemplate(UnaryType type, boolean sparse) {
		switch( type ) {
			case ROW_SUMS:
			case ROW_SUMSQS:
			case ROW_MINS:
			case ROW_MAXS:
			case ROW_MEANS:
			case ROW_VARS:
			case ROW_COUNTNNZS: {
				String vectName = StringUtils.capitalize(type.name().substring(4, type.name().length()-1).toLowerCase());
				return sparse ? "    double %TMP% = LibSpoofPrimitives.vect"+vectName+"(%IN1v%, %IN1i%, %POS1%, alen, len);\n":
						"    double %TMP% = LibSpoofPrimitives.vect"+vectName+"(%IN1%, %POS1%, %LEN%);\n";
			}

			case VECT_SQRT:
			case VECT_ABS:
			case VECT_ROUND:
			case VECT_CEIL:
			case VECT_FLOOR:
			case VECT_SIN:
			case VECT_TAN:
			case VECT_ASIN:
			case VECT_ATAN:
			case VECT_SINH:
			case VECT_TANH:
			case VECT_SIGN:{
				String vectName = type.getVectorPrimitiveName();
				return sparse ? DMLScript.SPARSE_INTERMEDIATE ?
					"    SparseRowVector %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(len, %IN1v%, %IN1i%, %POS1%, alen);\n" :
					"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write%IN1v%, %IN1i%, %POS1%, alen, len);\n" :
					"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %POS1%, %LEN%);\n";
			}
			case VECT_EXP:
			case VECT_POW2:
			case VECT_MULT2:
			case VECT_LOG:
			case VECT_COS:
			case VECT_ACOS:
			case VECT_COSH:
			case VECT_CUMSUM:
			case VECT_CUMMIN:
			case VECT_CUMMAX:
			case VECT_SPROP:
			case VECT_SIGMOID: {
				String vectName = type.getVectorPrimitiveName();
				return sparse ? "    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1v%, %IN1i%, %POS1%, alen, len);\n" :
						"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %POS1%, %LEN%);\n";
			}

			case EXP:
				return "    double %TMP% = FastMath.exp(%IN1%);\n";
			case LOOKUP_R:
				return sparse ?
						"    double %TMP% = getValue(%IN1v%, %IN1i%, ai, alen, 0);\n" :
						"    double %TMP% = getValue(%IN1%, rix);\n";
			case LOOKUP_C:
				return "    double %TMP% = getValue(%IN1%, n, 0, cix);\n";
			case LOOKUP_RC:
				return "    double %TMP% = getValue(%IN1%, n, rix, cix);\n";
			case LOOKUP0:
				return "    double %TMP% = %IN1%[0];\n";
			case POW2:
				return "    double %TMP% = %IN1% * %IN1%;\n";
			case MULT2:
				return "    double %TMP% = %IN1% + %IN1%;\n";
			case ABS:
				return "    double %TMP% = Math.abs(%IN1%);\n";
			case SIN:
				return "    double %TMP% = FastMath.sin(%IN1%);\n";
			case COS:
				return "    double %TMP% = FastMath.cos(%IN1%);\n";
			case TAN:
				return "    double %TMP% = FastMath.tan(%IN1%);\n";
			case ASIN:
				return "    double %TMP% = FastMath.asin(%IN1%);\n";
			case ACOS:
				return "    double %TMP% = FastMath.acos(%IN1%);\n";
			case ATAN:
				return "    double %TMP% = Math.atan(%IN1%);\n";
			case SINH:
				return "    double %TMP% = FastMath.sinh(%IN1%);\n";
			case COSH:
				return "    double %TMP% = FastMath.cosh(%IN1%);\n";
			case TANH:
				return "    double %TMP% = FastMath.tanh(%IN1%);\n";
			case SIGN:
				return "    double %TMP% = FastMath.signum(%IN1%);\n";
			case SQRT:
				return "    double %TMP% = Math.sqrt(%IN1%);\n";
			case LOG:
				return "    double %TMP% = Math.log(%IN1%);\n";
			case ROUND:
				return "    double %TMP% = Math.round(%IN1%);\n";
			case CEIL:
				return "    double %TMP% = FastMath.ceil(%IN1%);\n";
			case FLOOR:
				return "    double %TMP% = FastMath.floor(%IN1%);\n";
			case SPROP:
				return "    double %TMP% = %IN1% * (1 - %IN1%);\n";
			case SIGMOID:
				return "    double %TMP% = 1 / (1 + FastMath.exp(-%IN1%));\n";
			case LOG_NZ:
				return "    double %TMP% = (%IN1%==0) ? 0 : Math.log(%IN1%);\n";

			default:
				throw new RuntimeException("Invalid unary type: "+this.toString());
		}
	}
}
