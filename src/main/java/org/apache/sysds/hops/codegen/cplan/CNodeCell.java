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

package org.apache.sysds.hops.codegen.cplan;

import java.util.ArrayList;

import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.codegen.SpoofCompiler.GeneratorAPI;
import org.apache.sysds.hops.codegen.SpoofFusedOp.SpoofOutputDimsType;
import org.apache.sysds.runtime.codegen.SpoofCellwise;
import org.apache.sysds.runtime.codegen.SpoofCellwise.CellType;
import org.apache.sysds.runtime.util.UtilFunctions;

public class CNodeCell extends CNodeTpl 
{
	public static final String JAVA_TEMPLATE =
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
	
	private CellType _type = null;
	private AggOp _aggOp = null;
	private boolean _sparseSafe = false;
	private boolean _containsSeq = true;
	private boolean _requiresCastdtm = false;
	private boolean _multipleConsumers = false;
	
	public CNodeCell(ArrayList<CNode> inputs, CNode output ) {
		super(inputs, output);
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
	
	public void setAggOp(AggOp aggop) {
		_aggOp = aggop;
		_hash = 0;
	}
	
	public AggOp getAggOp() {
		return _aggOp;
	}

	public SpoofCellwise.AggOp getSpoofAggOp() {
		if(_aggOp != null)
			switch(_aggOp) {
				case SUM:
					return SpoofCellwise.AggOp.SUM;
				case SUM_SQ:
					return SpoofCellwise.AggOp.SUM_SQ;
				case MIN:
					return SpoofCellwise.AggOp.MIN;
				case MAX:
					return SpoofCellwise.AggOp.MAX;
				default:
					throw new RuntimeException("Unsupported cell type: "+_type.toString());
		}
		else
			return null;
	}

	public void setSparseSafe(boolean flag) {
		_sparseSafe = flag;
	}
	
	public boolean isSparseSafe() {
		return _sparseSafe;
	}
	
	public void setContainsSeq(boolean flag) {
		_containsSeq = flag;
	}
	
	public boolean containsSeq() {
		return _containsSeq;
	}
	
	public void setRequiresCastDtm(boolean flag) {
		_requiresCastdtm = flag;
		_hash = 0;
	}
	
	public boolean requiredCastDtm() {
		return _requiresCastdtm;
	}
	
	@Override
	public void renameInputs() {
		rRenameDataNode(_output, _inputs.get(0), "a");
		renameInputs(_inputs, 1);
	}

	public String codegen(boolean sparse, GeneratorAPI _api) {
		api = _api;

		String tmp = getLanguageTemplate(this, api);

		//generate dense/sparse bodies
		String tmpDense = _output.codegen(false, api);
		// TODO: workaround to fix name clash of cell and row template
		if(api == GeneratorAPI.CUDA)
			tmpDense = tmpDense.replace("a.vals(0)", "a");
		_output.resetGenerated();
		
		String varName = (getVarname() == null) ?
			createVarname() : getVarname();
		tmp = tmp.replace(api.isJava() ? 
			"%TMP%" : "/*%TMP%*/SPOOF_OP_NAME", varName);
		
		if(tmpDense.contains("grix"))
			tmp = tmp.replace("//%NEED_GRIX%", "\t\tuint32_t grix=_grix + rix;");
		else
			tmp = tmp.replace("//%NEED_GRIX%\n", "");

		// remove empty lines
//		if(!api.isJava())
//			tmpDense = tmpDense.replaceAll("(?m)^[ \t]*\r?\n", "");

		tmp = tmp.replace("%BODY_dense%", tmpDense);
		
		//Return last TMP. Square it for CUDA+SUM_SQ
		tmp = (api.isJava() || _aggOp != AggOp.SUM_SQ) ? tmp.replaceAll("%OUT%", _output.getVarname()) :
			tmp.replaceAll("%OUT%", _output.getVarname() + " * " + _output.getVarname());

		//replace meta data information
		tmp = tmp.replaceAll("%TYPE%", getCellType().name());
		tmp = tmp.replace("%AGG_OP_NAME%", (_aggOp != null) ? "AggOp." + _aggOp.name() : "null");
		tmp = tmp.replace("%SPARSE_SAFE%", String.valueOf(isSparseSafe()));
		tmp = tmp.replace("%SEQ%", String.valueOf(containsSeq()));

		if(api == GeneratorAPI.CUDA) {
			// ToDo: initial_value is misused to pass VT (values per thread) to no_agg operator
			String agg_op = "IdentityOp";
			String initial_value = "(T)1.0";
			if(_aggOp != null)
			switch(_aggOp) {
				case SUM:
				case SUM_SQ:
					agg_op = "SumOp";
					initial_value = "(T)0.0";
					break;
				case MIN:
					agg_op = "MinOp";
					initial_value = "MAX<T>()";
					break;
				case MAX:
					agg_op = "MaxOp";
					initial_value = "-MAX<T>()";
					break;
				default:
					agg_op = "IdentityOp";
					initial_value = "(T)0.0";
			}

			tmp = tmp.replaceAll("%AGG_OP%", agg_op);
			tmp = tmp.replaceAll("%INITIAL_VALUE%", initial_value);
		}
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
			case COL_AGG: return SpoofOutputDimsType.COLUMN_DIMS_COLS;
			case FULL_AGG: return SpoofOutputDimsType.SCALAR;
			default:
				throw new RuntimeException("Unsupported cell type: "+_type.toString());
		}
	}

	@Override
	public int hashCode() {
		if( _hash == 0 ) {
			int h = super.hashCode();
			h = UtilFunctions.intHashCode(h, _type.hashCode());
			h = UtilFunctions.intHashCode(h, (_aggOp!=null) ? _aggOp.hashCode() : 0);
			h = UtilFunctions.intHashCode(h, Boolean.hashCode(_sparseSafe));
			h = UtilFunctions.intHashCode(h, Boolean.hashCode(_requiresCastdtm));
			//note: _multipleConsumers irrelevant for plan comparison
			_hash = h;
		}
		return _hash;
	}
	
	@Override 
	public boolean equals(Object o) {
		if(!(o instanceof CNodeCell))
			return false;
		
		CNodeCell that = (CNodeCell)o;
		return super.equals(that) 
			&& _type == that._type
			&& _aggOp == that._aggOp
			&& _sparseSafe == that._sparseSafe
			&& _requiresCastdtm == that._requiresCastdtm
			&& equalInputReferences(
				_output, that._output, _inputs, that._inputs);
	}
	
	@Override
	public String getTemplateInfo() {
		StringBuilder sb = new StringBuilder();
		sb.append("SPOOF CELLWISE [type=");
		sb.append(_type.name());
		sb.append(", aggOp="+((_aggOp!=null) ? _aggOp.name() : "null"));
		sb.append(", sparseSafe="+_sparseSafe);
		sb.append(", castdtm="+_requiresCastdtm);
		sb.append(", mc="+_multipleConsumers);
		sb.append("]");
		return sb.toString();
	}
	@Override
	public boolean isSupported(GeneratorAPI api) {
		return (api == GeneratorAPI.CUDA || api == GeneratorAPI.JAVA) && _output.isSupported(api);
	}
	
	public int compile(GeneratorAPI api, String src) {
		if(api == GeneratorAPI.CUDA)
			return compile_nvrtc(SpoofCompiler.native_contexts.get(api), _genVar, src, _type.getValue(), 
				_aggOp != null ? _aggOp.ordinal() : 0, _sparseSafe);
		return -1;
	}
	
	private native int compile_nvrtc(long context, String name, String src, int type, int agg_op, boolean sparseSafe);
}
