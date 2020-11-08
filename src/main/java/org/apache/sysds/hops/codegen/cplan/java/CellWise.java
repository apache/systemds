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

import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeTernary;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CodeTemplate;
import org.apache.sysds.runtime.codegen.SpoofCellwise;

public class CellWise implements CodeTemplate {
	public static final String TEMPLATE =
			"package codegen;\n"
			+ "import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;\n"
			+ "import org.apache.sysds.runtime.codegen.SpoofCellwise;\n"
			+ "import org.apache.sysds.runtime.codegen.SpoofCellwise.AggOp;\n"
			+ "import org.apache.sysds.runtime.codegen.SpoofCellwise.CellType;\n"
			+ "import org.apache.sysds.runtime.codegen.SpoofOperator.SideInput;\n"
			+ "import org.apache.commons.math3.util.FastMath;\n"
			+ "\n"
			+ "public final class %TMP% extends SpoofCellwise {\n"
			+ "  public %TMP%() {\n"
			+ "    super(CellType.%TYPE%, %SPARSE_SAFE%, %SEQ%, %AGG_OP_NAME%);\n"
			+ "  }\n"
			+ "  protected double genexec(double a, SideInput[] b, double[] scalars, int m, int n, long grix, int rix, int cix) { \n"
			+ "%BODY_dense%"
			+ "    return %OUT%;\n"
			+ "  }\n"
			+ "}\n";

	@Override
	public String getTemplate() {
		throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
	}

	@Override
	public String getTemplate(SpoofCellwise.CellType ct) {
		switch(ct) {
			case NO_AGG:
			case FULL_AGG:
			case ROW_AGG:
			case COL_AGG:
			default:
				return TEMPLATE;
		}
	}

	@Override
	public String getTemplate(CNodeUnary.UnaryType type, boolean sparse) {
		throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
	}

	@Override
	public String getTemplate(CNodeBinary.BinType type, boolean sparseLhs, boolean sparseRhs, boolean scalarVector, boolean scalarInput) {
		throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
	}

	@Override
	public String getTemplate(CNodeTernary.TernaryType type, boolean sparse) {
		throw new RuntimeException("Calling wrong getTemplate method on " + getClass().getCanonicalName());
	}
}
