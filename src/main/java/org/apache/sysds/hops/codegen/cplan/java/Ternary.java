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

import org.apache.sysds.hops.codegen.cplan.CNodeTernary;
import org.apache.sysds.hops.codegen.cplan.CodeTemplate;

public class Ternary extends CodeTemplate {

	@Override
	public String getTemplate(CNodeTernary.TernaryType type, boolean sparse) {
		switch (type) {
			case PLUS_MULT:
				return "    double %TMP% = %IN1% + %IN2% * %IN3%;\n";

			case MINUS_MULT:
				return "    double %TMP% = %IN1% - %IN2% * %IN3%;\n";

			case BIASADD:
				return "    double %TMP% = %IN1% + getValue(%IN2%, cix/%IN3%);\n";

			case BIASMULT:
				return "    double %TMP% = %IN1% * getValue(%IN2%, cix/%IN3%);\n";

			case REPLACE:
				return "    double %TMP% = (%IN1% == %IN2% || (Double.isNaN(%IN1%) "
						+ "&& Double.isNaN(%IN2%))) ? %IN3% : %IN1%;\n";

			case REPLACE_NAN:
				return "    double %TMP% = Double.isNaN(%IN1%) ? %IN3% : %IN1%;\n";

			case IFELSE:
				return "    double %TMP% = (%IN1% != 0) ? %IN2% : %IN3%;\n";

			case LOOKUP_RC1:
				return sparse ?
					"    double %TMP% = getValue(%IN1v%, %IN1i%, %POS%, %LEN%, %IN3%-1);\n" :
					"    double %TMP% = getValue(%IN1%, %IN2%, rix, %IN3%-1);\n";

			case LOOKUP_RVECT1:
				return "    double[] %TMP% = getVector(%IN1%, %IN2%, rix, %IN3%-1);\n";

			default:
				throw new RuntimeException("Invalid ternary type: "+this.toString());
		}
	}
}
