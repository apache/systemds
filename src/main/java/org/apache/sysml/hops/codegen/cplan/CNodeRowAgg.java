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

import org.apache.sysml.hops.codegen.SpoofFusedOp.SpoofOutputDimsType;

public class CNodeRowAgg extends CNodeTpl
{
	private static final String TEMPLATE = 
			  "package codegen;\n"
			+ "import org.apache.sysml.runtime.codegen.LibSpoofPrimitives;\n"
			+ "import org.apache.sysml.runtime.codegen.SpoofRowAggregate;\n"
			+ "\n"
			+ "public final class %TMP% extends SpoofRowAggregate { \n"
			+ "  public %TMP%() {\n"
			+ "    _colVector = %FLAG%;\n"
			+ "  }\n"
			+ "  protected void genexecRowDense( double[] a, int ai, double[][] b, double[] scalars, double[] c, int len, int rowIndex ) { \n"
			+ "%BODY_dense%"
			+ "  }\n"
			+ "  protected void genexecRowSparse( double[] avals, int[] aix, int ai, double[][] b, double[] scalars, double[] c, int len, int rowIndex ) { \n"
			+ "%BODY_sparse%"
			+ "  }\n"			
			+ "}\n";

	public CNodeRowAgg(ArrayList<CNode> inputs, CNode output ) {
		super(inputs, output);
	}
	
	
	@Override
	public String codegen(boolean sparse) {
		// note: ignore sparse flag, generate both
		String tmp = TEMPLATE;
		
		//rename inputs
		rReplaceDataNode(_output, _inputs.get(0), "a"); // input matrix
		renameInputs(_inputs, 1);
		
		//generate dense/sparse bodies
		String tmpDense = _output.codegen(false);
		_output.resetGenerated();
		String tmpSparse = _output.codegen(true);
		tmp = tmp.replaceAll("%TMP%", createVarname());
		tmp = tmp.replaceAll("%BODY_dense%", tmpDense);
		tmp = tmp.replaceAll("%BODY_sparse%", tmpSparse);
		
		//replace outputs 
		tmp = tmp.replaceAll("%OUT%", "c");
		tmp = tmp.replaceAll("%POSOUT%", "0");
		
		//replace size information
		tmp = tmp.replaceAll("%LEN%", "len");
		
		//replace colvector information and start position
		tmp = tmp.replaceAll("%FLAG%", String.valueOf(_output._cols==1));
		tmp = tmp.replaceAll("bi", "0");
		
		return tmp;
	}

	@Override
	public void setOutputDims() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public SpoofOutputDimsType getOutputDimType() {
		return (_output._cols==1) ? 
			SpoofOutputDimsType.COLUMN_DIMS_ROWS : //column vector
			SpoofOutputDimsType.COLUMN_DIMS_COLS;  //row vector
	}
	
	@Override
	public CNodeTpl clone() {
		return new CNodeRowAgg(_inputs, _output);
	}
	
	@Override
	public int hashCode() {
		return super.hashCode();
	}
	
	@Override 
	public boolean equals(Object o) {
		if(!(o instanceof CNodeRowAgg))
			return false;
		
		CNodeRowAgg that = (CNodeRowAgg)o;
		return super.equals(o)
			&& equalInputReferences(
				_output, that._output, _inputs, that._inputs);
	}
	
	@Override
	public String getTemplateInfo() {
		StringBuilder sb = new StringBuilder();
		sb.append("SPOOF ROWAGGREGATE");
		return sb.toString();
	}
}
