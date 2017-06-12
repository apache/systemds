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
import org.apache.sysml.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysml.hops.codegen.template.TemplateUtils;
import org.apache.sysml.runtime.codegen.SpoofRowwise.RowType;

public class CNodeRow extends CNodeTpl
{
	private static final String TEMPLATE = 
			  "package codegen;\n"
			+ "import org.apache.sysml.runtime.codegen.LibSpoofPrimitives;\n"
			+ "import org.apache.sysml.runtime.codegen.SpoofRowwise;\n"
			+ "import org.apache.sysml.runtime.codegen.SpoofRowwise.RowType;\n"
			+ "import org.apache.commons.math3.util.FastMath;\n"
			+ "\n"
			+ "public final class %TMP% extends SpoofRowwise { \n"
			+ "  public %TMP%() {\n"
			+ "    super(RowType.%TYPE%, %CBIND0%, %VECT_MEM%);\n"
			+ "  }\n"
			+ "  protected void genexecRowDense( double[] a, int ai, double[][] b, double[] scalars, double[] c, int len, int rowIndex ) { \n"
			+ "%BODY_dense%"
			+ "  }\n"
			+ "  protected void genexecRowSparse( double[] avals, int[] aix, int ai, double[][] b, double[] scalars, double[] c, int len, int rowIndex ) { \n"
			+ "%BODY_sparse%"
			+ "  }\n"			
			+ "}\n";

	private static final String TEMPLATE_ROWAGG_OUT = "    c[rowIndex] = %IN%;\n";
	private static final String TEMPLATE_NOAGG_OUT = "    LibSpoofPrimitives.vectWrite(%IN%, c, rowIndex*len, len);\n";
	
	public CNodeRow(ArrayList<CNode> inputs, CNode output ) {
		super(inputs, output);
	}
	
	private RowType _type = null; //access pattern 
	private int _numVectors = -1; //number of intermediate vectors
	
	public void setNumVectorIntermediates(int num) {
		_numVectors = num;
	}
	
	public int getNumVectorIntermediates() {
		return _numVectors;
	}
	
	public void setRowType(RowType type) {
		_type = type;
		_hash = 0;
	}
	
	public RowType getRowType() {
		return _type;
	}
	
	@Override
	public String codegen(boolean sparse) {
		// note: ignore sparse flag, generate both
		String tmp = TEMPLATE;
		
		//rename inputs
		rReplaceDataNode(_output, _inputs.get(0), "a"); // input matrix
		renameInputs(_inputs, 1);
		
		//generate dense/sparse bodies
		String tmpDense = _output.codegen(false)
			+ getOutputStatement(_output.getVarname());
		_output.resetGenerated();
		String tmpSparse = _output.codegen(true)
			+ getOutputStatement(_output.getVarname());
		tmp = tmp.replaceAll("%TMP%", createVarname());
		tmp = tmp.replaceAll("%BODY_dense%", tmpDense);
		tmp = tmp.replaceAll("%BODY_sparse%", tmpSparse);
		
		//replace outputs 
		tmp = tmp.replaceAll("%OUT%", "c");
		tmp = tmp.replaceAll("%POSOUT%", "0");
		
		//replace size information
		tmp = tmp.replaceAll("%LEN%", "len");
		
		//replace colvector information and number of vector intermediates
		tmp = tmp.replaceAll("%TYPE%", _type.name());
		tmp = tmp.replaceAll("%CBIND0%", String.valueOf(
			TemplateUtils.isUnary(_output, UnaryType.CBIND0)));
		tmp = tmp.replaceAll("%VECT_MEM%", String.valueOf(_numVectors));
		
		return tmp;
	}
	
	private String getOutputStatement(String varName) {
		if( !_type.isColumnAgg() ) {
			String tmp = (_type==RowType.NO_AGG) ?
				TEMPLATE_NOAGG_OUT : TEMPLATE_ROWAGG_OUT;
			return tmp.replace("%IN%", varName);
		}
		return "";
	}

	@Override
	public void setOutputDims() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public SpoofOutputDimsType getOutputDimType() {
		switch( _type ) {
			case NO_AGG: return SpoofOutputDimsType.INPUT_DIMS;
			case ROW_AGG: return TemplateUtils.isUnary(_output, UnaryType.CBIND0) ?
				SpoofOutputDimsType.ROW_DIMS2 : SpoofOutputDimsType.ROW_DIMS;
			case COL_AGG: return SpoofOutputDimsType.COLUMN_DIMS_COLS; //row vector
			case COL_AGG_T: return SpoofOutputDimsType.COLUMN_DIMS_ROWS; //column vector
			default:
				throw new RuntimeException("Unsupported row type: "+_type.toString());
		}
	}
	
	@Override
	public CNodeTpl clone() {
		CNodeRow tmp = new CNodeRow(_inputs, _output);
		tmp.setRowType(_type);
		tmp.setNumVectorIntermediates(_numVectors);
		return tmp;
	}
	
	@Override
	public int hashCode() {
		if( _hash == 0 ) {
			int h1 = super.hashCode();
			int h2 = _type.hashCode();
			int h3 = _numVectors;
			_hash = Arrays.hashCode(new int[]{h1,h2,h3});
		}
		return _hash;
	}
	
	@Override 
	public boolean equals(Object o) {
		if(!(o instanceof CNodeRow))
			return false;
		
		CNodeRow that = (CNodeRow)o;
		return super.equals(o)
			&& _type == that._type
			&& _numVectors == that._numVectors	
			&& equalInputReferences(
				_output, that._output, _inputs, that._inputs);
	}
	
	@Override
	public String getTemplateInfo() {
		StringBuilder sb = new StringBuilder();
		sb.append("SPOOF ROWAGGREGATE [type=");
		sb.append(_type.name());
		sb.append(", reqVectMem=");
		sb.append(_numVectors);
		sb.append("]");
		return sb.toString();
	}
}
