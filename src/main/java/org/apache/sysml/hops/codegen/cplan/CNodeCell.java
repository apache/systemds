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
import org.apache.sysml.runtime.codegen.SpoofCellwise.CellType;

public class CNodeCell extends CNodeTpl 
{	
	private static final String TEMPLATE = 
			  "package codegen;\n"
			+ "import java.util.Arrays;\n"
			+ "import java.io.Serializable;\n"
			+ "import java.util.ArrayList;\n"
			+ "import org.apache.sysml.runtime.codegen.LibSpoofPrimitives;\n"
			+ "import org.apache.sysml.runtime.codegen.SpoofCellwise;\n"
			+ "import org.apache.sysml.runtime.codegen.SpoofCellwise.CellType;\n"
			+ "import org.apache.commons.math3.util.FastMath;\n"
			+ "\n"
			+ "public final class %TMP% extends SpoofCellwise {\n" 
			+ "  public %TMP%() {\n"
			+ "    _type = CellType.%TYPE%;\n"
			+ "  }\n"
			+ "  protected double genexecDense( double a, double[][] b, double[] scalars, int n, int m, int rowIndex, int colIndex) { \n"
			+ "%BODY_dense%"
			+ "    return %OUT%;\n"
			+ "  } \n"
			+ "}";
	
	private CellType _type = null;
	private boolean _multipleConsumers = false;
	
	public CNodeCell(ArrayList<CNode> inputs, CNode output ) {
		super(inputs,output);
	}
	
	public void setMultipleConsumers(boolean flag) {
		_multipleConsumers = flag;
	}
	
	public boolean hasMultipleConsumers() {
		return _multipleConsumers;
	}
	
	public void setCellType(CellType type) {
		_type = type;
		_hash = 0;
	}
	
	public CellType getCellType() {
		return _type;
	}
	
	@Override
	public String codegen(boolean sparse) {
		String tmp = TEMPLATE;
		
		//rename inputs
		rReplaceDataNode(_output, _inputs.get(0), "a");
		renameInputs(_inputs, 1);
		
		//generate dense/sparse bodies
		String tmpDense = _output.codegen(false);
		_output.resetGenerated();

		tmp = tmp.replaceAll("%TMP%", createVarname());
		tmp = tmp.replaceAll("%BODY_dense%", tmpDense);
		
		//return last TMP
		tmp = tmp.replaceAll("%OUT%", getCurrentVarName());
		
		//replace aggregate information
		tmp = tmp.replaceAll("%TYPE%", getCellType().toString());
		
		return tmp;
	}

	@Override
	public void setOutputDims() {
		
		
	}

	@Override
	public CNodeTpl clone() {
		CNodeCell tmp = new CNodeCell(_inputs, _output);
		tmp.setDataType(getDataType());
		tmp.setCellType(getCellType());
		tmp.setMultipleConsumers(hasMultipleConsumers());
		return tmp;
	}
	
	@Override
	public SpoofOutputDimsType getOutputDimType() {
		switch( _type ) {
			case NO_AGG: return SpoofOutputDimsType.INPUT_DIMS;
			case ROW_AGG: return SpoofOutputDimsType.ROW_DIMS;
			case FULL_AGG: return SpoofOutputDimsType.SCALAR;
			default:
				throw new RuntimeException("Unsupported cell type: "+_type.toString());
		}
	}

	@Override
	public int hashCode() {
		if( _hash == 0 ) {
			int h1 = super.hashCode();
			int h2 = _type.hashCode();
			//note: _multipleConsumers irrelevant for plan comparison
			_hash = Arrays.hashCode(new int[]{h1,h2});
		}
		return _hash;
	}
	
	@Override 
	public boolean equals(Object o) {
		if(!(o instanceof CNodeCell))
			return false;
		
		CNodeCell that = (CNodeCell)o;
		return super.equals(that)
			&& _type == that._type;
	}
	
	@Override
	public String getTemplateInfo() {
		StringBuilder sb = new StringBuilder();
		sb.append("SPOOF CELLWISE [type=");
		sb.append(_type.name());
		sb.append(", mc="+_multipleConsumers);
		sb.append("]");
		return sb.toString();
	}
}
