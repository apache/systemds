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

package org.apache.sysml.hops.codegen.cplan;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.sysml.hops.codegen.SpoofFusedOp.SpoofOutputDimsType;
import org.apache.sysml.runtime.codegen.SpoofOuterProduct.OutProdType;


public class CNodeOuterProduct extends CNodeTpl
{	
	private static final String TEMPLATE = 
			  "package codegen;\n"
			+ "import java.util.Arrays;\n"
			+ "import java.util.ArrayList;\n"
			+ "import org.apache.sysml.runtime.codegen.LibSpoofPrimitives;\n"
			+ "import org.apache.sysml.runtime.codegen.SpoofOuterProduct;\n"
			+ "import org.apache.sysml.runtime.codegen.SpoofOuterProduct.OutProdType;\n"
			+ "import org.apache.commons.math3.util.FastMath;\n"
			+ "\n"
			+ "public final class %TMP% extends SpoofOuterProduct { \n"
			+ "  public %TMP%() {\n"
			+ "    _outerProductType = OutProdType.%TYPE%;\n"
			+ "  }\n"
			+ "  protected void genexecDense( double _a, double[] _a1, int _a1i, double[] _a2, int _a2i, double[][] _b, double[] _scalars, double[] _c, int _ci, int _n, int _m, int _k, int _rowIndex, int _colIndex) { \n"
			+ "%BODY_dense%"
			+ "  } \n"
			+ "  protected double genexecCellwise( double _a, double[] _a1, int _a1i, double[] _a2, int _a2i, double[][] _b, double[] _scalars, int _n, int _m, int _k, int _rowIndex, int _colIndex) { \n"
			+ "%BODY_cellwise%"
			+ "    return %OUT_cellwise%;\n"
			+ "  } \n"
			
			+ "}";
	
	private OutProdType _type = null;
	private boolean _transposeOutput = false;
	
	public CNodeOuterProduct(ArrayList<CNode> inputs, CNode output ) {
		super(inputs,output);
	}
	
	@Override
	public String codegen(boolean sparse) {
		// note: ignore sparse flag, generate both
		String tmp = TEMPLATE;
		
		//rename inputs
		rReplaceDataNode(_output, _inputs.get(0), "_a");
		rReplaceDataNode(_output, _inputs.get(1), "_a1"); // u
		rReplaceDataNode(_output, _inputs.get(2), "_a2"); // v
		renameInputs(_inputs, 3);

		//generate dense/sparse bodies
		String tmpDense = _output.codegen(false);
		_output.resetGenerated();

		tmp = tmp.replaceAll("%TMP%", createVarname());

		if(_type == OutProdType.LEFT_OUTER_PRODUCT || _type == OutProdType.RIGHT_OUTER_PRODUCT) {
			tmp = tmp.replaceAll("%BODY_dense%", tmpDense);
			tmp = tmp.replaceAll("%OUT%", "_c");
			tmp = tmp.replaceAll("%BODY_cellwise%", "");
			tmp = tmp.replaceAll("%OUT_cellwise%", "0");
		}
		else {
			tmp = tmp.replaceAll("%BODY_dense%", "");
			tmp = tmp.replaceAll("%BODY_cellwise%", tmpDense);
			tmp = tmp.replaceAll("%OUT_cellwise%", getCurrentVarName());
		}
		//replace size information
		tmp = tmp.replaceAll("%LEN%", "_k");
		
		tmp = tmp.replaceAll("%POSOUT%", "_ci");
		
		tmp = tmp.replaceAll("%TYPE%", _type.toString());

		return tmp;
	}

	public void setOutProdType(OutProdType type) {
		_type = type;
		_hash = 0;
	}
	
	public OutProdType getOutProdType() {
		return _type;
	}

	@Override
	public void setOutputDims() {
		
	}

	public void setTransposeOutput(boolean transposeOutput) {
		_transposeOutput = transposeOutput;
		_hash = 0;
	}

	
	public boolean isTransposeOutput() {
		return _transposeOutput;
	}

	@Override
	public SpoofOutputDimsType getOutputDimType() {
		switch( _type ) {
			case LEFT_OUTER_PRODUCT:
				return SpoofOutputDimsType.COLUMN_RANK_DIMS;
			case RIGHT_OUTER_PRODUCT:
				return SpoofOutputDimsType.ROW_RANK_DIMS;
			case CELLWISE_OUTER_PRODUCT:
				return SpoofOutputDimsType.INPUT_DIMS;
			case AGG_OUTER_PRODUCT:
				return SpoofOutputDimsType.SCALAR;
			default:
				throw new RuntimeException("Unsupported outer product type: "+_type.toString());
		}
	}

	@Override
	public CNodeTpl clone() {
		return new CNodeOuterProduct(_inputs, _output);
	}
	
	@Override
	public int hashCode() {
		if( _hash == 0 ) {
			int h1 = super.hashCode();
			int h2 = _type.hashCode();
			int h3 = Boolean.hashCode(_transposeOutput);
			_hash = Arrays.hashCode(new int[]{h1,h2,h3});
		}
		return _hash;
	}
	
	@Override 
	public boolean equals(Object o) {
		if(!(o instanceof CNodeOuterProduct))
			return false;
		
		CNodeOuterProduct that = (CNodeOuterProduct)o;
		return super.equals(that)
			&& _type == that._type
			&& _transposeOutput == that._transposeOutput;
	}
}
