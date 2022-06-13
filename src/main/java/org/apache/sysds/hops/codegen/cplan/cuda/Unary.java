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

import org.apache.commons.lang.StringUtils;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CodeTemplate;

import static org.apache.sysds.runtime.matrix.data.LibMatrixNative.isSinglePrecision;

public class Unary extends CodeTemplate {

	@Override
	public String getTemplate(CNodeUnary.UnaryType type, boolean sparse) {
		switch( type ) {
			case ROW_SUMS:
			case ROW_SUMSQS:
			case ROW_MINS:
			case ROW_MAXS:
			case ROW_MEANS:
			case ROW_COUNTNNZS: {
				String vectName = StringUtils.capitalize(type.name().substring(4, type.name().length()-1).toLowerCase());
				return sparse ? "		T %TMP% = vect"+vectName+"(%IN1v%, %IN1i%, %POS1%, alen, %LEN%);\n":
					"		T %TMP% = vect"+vectName+"(%IN1%, static_cast<uint32_t>(%POS1%), %LEN%);\n";

			}

			case VECT_EXP:
			case VECT_POW2:
			case VECT_MULT2:
			case VECT_SQRT:
			case VECT_LOG:
			case VECT_ABS:
			case VECT_ROUND:
			case VECT_CEIL:
			case VECT_FLOOR:
			case VECT_SIGN:
			case VECT_SIN:
			case VECT_COS:
			case VECT_TAN:
			case VECT_ASIN:
			case VECT_ACOS:
			case VECT_ATAN:
			case VECT_SINH:
			case VECT_COSH:
			case VECT_TANH:
			case VECT_CUMSUM:
			case VECT_CUMMIN:
			case VECT_CUMMAX:
			case VECT_SPROP:
			case VECT_SIGMOID: {
				String vectName = type.getVectorPrimitiveName();
				return sparse ? "		Vector<T>& %TMP% = vect"+vectName+"Write(%IN1v%, %IN1i%, %POS1%, alen, %LEN%, this);\n" :
					"		Vector<T>& %TMP% = vect"+vectName+"Write(%IN1%, static_cast<uint32_t>(%POS1%), %LEN%, this);\n";
			}

			case EXP:
				if(isSinglePrecision())
					return "	T %TMP% = expf(%IN1%);\n";
				else
					return "	T %TMP% = exp(%IN1%);\n";
			case LOOKUP_R:
				return sparse ?
					"\t\tT %TMP% = getValue(%IN1v%, %IN1i%, ai, alen, 0);\n" :
//						"		T %TMP% = %IN1%.val(rix);\n";
					"\t\tT %TMP% = getValue(%IN1%, rix);\n";
			case LOOKUP_C:
				return "\t\tT %TMP% = getValue(%IN1%, n, 0, cix);\n";
			case LOOKUP_RC:
				return "\t\tT %TMP% = getValue(%IN1%, n, rix, cix);\n";
			case LOOKUP0:
				return "\t\tT %TMP% = %IN1%[0];\n";
			case POW2:
				return "	T %TMP% = %IN1% * %IN1%;\n";
			case MULT2:
				return "	T %TMP% = %IN1% + %IN1%;\n";
			case ABS:
				if(isSinglePrecision())
					return "	T %TMP% = fabsf(%IN1%);\n";
				else
					return "\t\tT %TMP% = fabs(%IN1%);\n";
			case SIN:
				if(isSinglePrecision())
					return "	T %TMP% = sinf(%IN1%);\n";
				else
					return "	T %TMP% = sin(%IN1%);\n";
			case COS:
				if(isSinglePrecision())
					return "	T %TMP% = cosf(%IN1%);\n";
				else
					return "	T %TMP% = cos(%IN1%);\n";
			case TAN:
				if(isSinglePrecision())
					return "	T %TMP% = tanf(%IN1%);\n";
				else
					return "	T %TMP% = tan(%IN1%);\n";
			case ASIN:
				if(isSinglePrecision())
					return "	T %TMP% = asinf(%IN1%);\n";
				else
					return "	T %TMP% = asin(%IN1%);\n";
			case ACOS:
				if(isSinglePrecision())
					return "	T %TMP% = acosf(%IN1%);\n";
				else
					return "	T %TMP% = acos(%IN1%);\n";
			case ATAN:
				if(isSinglePrecision())
					return "	T %TMP% = atanf(%IN1%);\n";
				else
					return "	T %TMP% = atan(%IN1%);\n";
			case SINH:
				if(isSinglePrecision())
					return "	T %TMP% = sinhf(%IN1%);\n";
				else
					return "	T %TMP% = sinh(%IN1%);\n";
			case COSH:
				if(isSinglePrecision())
					return "	T %TMP% = coshf(%IN1%);\n";
				else
					return "	T %TMP% = cosh(%IN1%);\n";
			case TANH:
				if(isSinglePrecision())
					return "	T %TMP% = tanhf(%IN1%);\n";
				else
					return "	T %TMP% = tanh(%IN1%);\n";
			case SIGN:
				return "	T %TMP% = signbit(%IN1%) == 0 ? 1.0 : -1.0;\n";
			case SQRT:
				if(isSinglePrecision())
					return "	T %TMP% = sqrtf(%IN1%);\n";
				else
					return "	T %TMP% = sqrt(%IN1%);\n";
			case LOG:

				if(isSinglePrecision())
					return "	T %TMP% = logf(%IN1%);\n";
				else
					return "		T %TMP% = log(%IN1%);\n";
			case ROUND:
				if(isSinglePrecision())
					return "	T %TMP% = roundf(%IN1%);\n";
				else
					return "\t\tT %TMP% = round(%IN1%);\n";
			case CEIL:
				if(isSinglePrecision())
					return "	T %TMP% = ceilf(%IN1%);\n";
				else
					return "	T %TMP% = ceil(%IN1%);\n";
			case FLOOR:
				if(isSinglePrecision())
					return "	T %TMP% = floorf(%IN1%);\n";
				else
					return "	T %TMP% = floor(%IN1%);\n";
			case SPROP:
				return "	T %TMP% = %IN1% * (1 - %IN1%);\n";
			case SIGMOID:
				if(isSinglePrecision())
					return "	T %TMP% = 1 / (1 + expf(-%IN1%));\n";
				else
					return "	T %TMP% = 1 / (1 + exp(-%IN1%));\n";
			case LOG_NZ:
				if(isSinglePrecision())
					return "	T %TMP% = (%IN1%==0) ? 0 : logf(%IN1%);\n";
				else
					return "	T %TMP% = (%IN1%==0) ? 0 : log(%IN1%);\n";

			default:
				throw new RuntimeException("Invalid unary type: "+this.toString());
		}
	}
}
