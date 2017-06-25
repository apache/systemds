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

import org.apache.commons.collections.CollectionUtils;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.codegen.SpoofFusedOp.SpoofOutputDimsType;
import org.apache.sysml.runtime.util.UtilFunctions;

public class CNodeMultiAgg extends CNodeTpl
{
	private static final String TEMPLATE = 
			  "package codegen;\n"
			+ "import org.apache.sysml.runtime.codegen.LibSpoofPrimitives;\n"
			+ "import org.apache.sysml.runtime.codegen.SpoofCellwise;\n"
			+ "import org.apache.sysml.runtime.codegen.SpoofCellwise.AggOp;\n"
			+ "import org.apache.sysml.runtime.codegen.SpoofMultiAggregate;\n"
			+ "import org.apache.sysml.runtime.codegen.SpoofOperator.SideInput;\n"
			+ "import org.apache.commons.math3.util.FastMath;\n"
			+ "\n"
			+ "public final class %TMP% extends SpoofMultiAggregate { \n"
			+ "  public %TMP%() {\n"
			+ "    super(%AGG_OP%);\n"
			+ "  }\n"
			+ "  protected void genexec(double a, SideInput[] b, double[] scalars, double[] c, "
					+ "int m, int n, int rowIndex, int colIndex) { \n"
			+ "%BODY_dense%"
			+ "  }\n"
			+ "}\n";
	private static final String TEMPLATE_OUT_SUM   = "    c[%IX%] += %IN%;\n";
	private static final String TEMPLATE_OUT_SUMSQ = "    c[%IX%] += %IN% * %IN%;\n";
	private static final String TEMPLATE_OUT_MIN   = "    c[%IX%] = Math.min(c[%IX%], %IN%);\n";
	private static final String TEMPLATE_OUT_MAX   = "    c[%IX%] = Math.max(c[%IX%], %IN%);\n";
	
	private ArrayList<CNode> _outputs = null; 
	private ArrayList<AggOp> _aggOps = null;
	private ArrayList<Hop> _roots = null;
	
	public CNodeMultiAgg(ArrayList<CNode> inputs, ArrayList<CNode> outputs) {
		super(inputs, null);
		_outputs = outputs;
	}
	
	public ArrayList<CNode> getOutputs() {
		return _outputs;
	}
	
	@Override
	public void resetVisitStatusOutputs() {
		for( CNode output : _outputs )
			output.resetVisitStatus();
	}
	
	public void setAggOps(ArrayList<AggOp> aggOps) {
		_aggOps = aggOps;
		_hash = 0;
	}
	
	public ArrayList<AggOp> getAggOps() {
		return _aggOps;
	}
	
	public void setRootNodes(ArrayList<Hop> roots) {
		_roots = roots;
	}
	
	public ArrayList<Hop> getRootNodes() {
		return _roots;
	}
	
	@Override
	public void renameInputs() {
		rRenameDataNode(_outputs, _inputs.get(0), "a"); // input matrix
		renameInputs(_outputs, _inputs, 1);
	}
	
	@Override
	public String codegen(boolean sparse) {
		// note: ignore sparse flag, generate both
		String tmp = TEMPLATE;
		
		//generate dense/sparse bodies
		StringBuilder sb = new StringBuilder();
		for( CNode out : _outputs )
			sb.append(out.codegen(false));
		for( CNode out : _outputs )
			out.resetGenerated();

		//append output assignments
		for( int i=0; i<_outputs.size(); i++ ) {
			CNode out = _outputs.get(i);
			String tmpOut = getAggTemplate(i);
			//get variable name (w/ handling of direct consumption of inputs)
			String varName = (out instanceof CNodeData && ((CNodeData)out).getHopID()==
				((CNodeData)_inputs.get(0)).getHopID()) ? "a" : out.getVarname(); 
			tmpOut = tmpOut.replace("%IN%", varName);
			tmpOut = tmpOut.replace("%IX%", String.valueOf(i));
			sb.append(tmpOut);
		}
			
		//replace class name and body
		tmp = tmp.replace("%TMP%", createVarname());
		tmp = tmp.replace("%BODY_dense%", sb.toString());
	
		//replace meta data information
		String aggList = "";
		for( AggOp aggOp : _aggOps ) {
			aggList += !aggList.isEmpty() ? "," : "";
			aggList += "AggOp."+aggOp.name();
		}
		tmp = tmp.replace("%AGG_OP%", aggList);

		return tmp;
	}

	@Override
	public void setOutputDims() {
		
	}

	@Override
	public SpoofOutputDimsType getOutputDimType() {
		return SpoofOutputDimsType.MULTI_SCALAR;
	}
	
	@Override
	public CNodeTpl clone() {
		CNodeMultiAgg ret = new CNodeMultiAgg(_inputs, _outputs);
		ret.setAggOps(getAggOps());
		return ret;
	}
	
	@Override
	public int hashCode() {
		if( _hash == 0 ) {
			int h = super.hashCode();
			for( int i=0; i<_outputs.size(); i++ ) {
				h = UtilFunctions.intHashCode(h, UtilFunctions.intHashCode(
					_outputs.get(i).hashCode(), _aggOps.get(i).hashCode()));
			}
			_hash = h;
		}
		return _hash;
	}
	
	@Override 
	public boolean equals(Object o) {
		if(!(o instanceof CNodeMultiAgg))
			return false;
		CNodeMultiAgg that = (CNodeMultiAgg)o;
		return super.equals(o)
			&& CollectionUtils.isEqualCollection(_aggOps, that._aggOps)	
			&& equalInputReferences(
				_outputs, that._outputs, _inputs, that._inputs);
	}
	
	@Override
	public String getTemplateInfo() {
		StringBuilder sb = new StringBuilder();
		sb.append("SPOOF MULTIAGG [aggOps=");
		sb.append(Arrays.toString(_aggOps.toArray(new AggOp[0])));
		sb.append("]");
		return sb.toString();
	}
	
	private String getAggTemplate(int pos) {
		switch( _aggOps.get(pos) ) {
			case SUM: return TEMPLATE_OUT_SUM;
			case SUM_SQ: return TEMPLATE_OUT_SUMSQ;
			case MIN: return TEMPLATE_OUT_MIN;
			case MAX: return TEMPLATE_OUT_MAX;
			default:
				return null;
		}
	}
}
