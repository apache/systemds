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

import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.codegen.SpoofCompiler.GeneratorAPI;
import org.apache.sysds.hops.codegen.SpoofFusedOp.SpoofOutputDimsType;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.hops.codegen.template.TemplateUtils;
import org.apache.sysds.runtime.codegen.SpoofRowwise.RowType;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.util.ArrayList;

public class CNodeRow extends CNodeTpl
{
	protected static final String JAVA_TEMPLATE = 
		  "package codegen;\n"
		+ "import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;\n"
		+ "import org.apache.sysds.runtime.codegen.SpoofOperator.SideInput;\n"
		+ "import org.apache.sysds.runtime.codegen.SpoofRowwise;\n"
		+ "import org.apache.sysds.runtime.codegen.SpoofRowwise.RowType;\n"
	    + "import org.apache.sysds.runtime.data.SparseRowVector;\n"
		+ "import org.apache.commons.math3.util.FastMath;\n"
		+ "\n"
		+ "public final class %TMP% extends SpoofRowwise { \n"
		+ "  public %TMP%() {\n"
		+ "    super(RowType.%TYPE%, %CONST_DIM2%, %TB1%, %VECT_MEM%);\n"
		+ "  }\n"
		+ "  protected void genexec(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len, long grix, int rix) { \n"
		+ "%BODY_dense%"
		+ "  }\n"
		+ "  protected void genexec(double[] avals, int[] aix, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int alen, int len, long grix, int rix) { \n"
		+ "%BODY_sparse%"
		+ "  }\n"
		+ "}\n";
	
	private static final String TEMPLATE_ROWAGG_OUT  = "    c[rix] = %IN%;\n";
	private static final String TEMPLATE_FULLAGG_OUT = "    c[0] += %IN%;\n";
	private static final String TEMPLATE_NOAGG_OUT   = "    LibSpoofPrimitives.vectWrite(%IN%, c, ci, %LEN%);\n";
	private static final String TEMPLATE_NOAGG_CONST_OUT_CUDA   = "\t\tvectWrite(%IN%, c.vals(0), 0, ci, %LEN%);\n";
	private static final String TEMPLATE_NOAGG_OUT_CUDA   = "\t\tvectWrite(%IN%, c.vals(0), 0, ci, %LEN%);\n";
//	private static final String TEMPLATE_ROWAGG_OUT_CUDA  = "\t\tif(threadIdx.x == 0){\n\t\t\t*(c.vals(rix)) = %IN%;\n//printf(\"rix=%d TMP7=%f TMP8=%f %IN%=%f\\n\",rix, TMP7, TMP8,%IN%);\n}\n";
private static final String TEMPLATE_ROWAGG_OUT_CUDA  = "\t\tif(threadIdx.x == 0){\n\t\t\t*(c.vals(rix)) = %IN%;\n\t\t}\n";
//	private static final String TEMPLATE_FULLAGG_OUT_CUDA =
//		"\t\tif(threadIdx.x == 0) {\n\t\t\tT old = atomicAdd(c.vals(0), %IN%);\n//\t\t\tprintf(\"bid=%d full_agg add %f to %f\\n\",blockIdx.x, %IN%, old);\n\t\t}\n";
	private static final String TEMPLATE_FULLAGG_OUT_CUDA =
		"\t\tif(threadIdx.x == 0) {\n\t\tT old = atomicAdd(c.vals(0), %IN%);\n\t\t}\n";


	public CNodeRow(ArrayList<CNode> inputs, CNode output ) {
		super(inputs, output);
	}
	
	private RowType _type = null; //access pattern 
	private long _constDim2 = -1; //constant number of output columns
	private int _numVectors = -1; //number of intermediate vectors
	private boolean _tb1 = false;
	
	public void setRowType(RowType type) {
		_type = type;
		_hash = 0;
	}
	
	public RowType getRowType() {
		return _type;
	}
	
	public void setNumVectorIntermediates(int num) {
		_numVectors = num;
		_hash = 0;
	}
	
	public int getNumVectorIntermediates() {
		return _numVectors;
	}
	
	public void setConstDim2(long dim2) {
		_constDim2 = dim2;
		_hash = 0;
	}
	
	public long getConstDim2() {
		return _constDim2;
	}
	
	@Override
	public void renameInputs() {
		rRenameDataNode(_output, _inputs.get(0), "a"); // input matrix
		renameInputs(_inputs, 1);
	}
	
	@Override
	public String codegen(boolean sparse, GeneratorAPI _api) {
		api = _api;
		
		// note: ignore sparse flag, generate both
		String tmp = getLanguageTemplate(this, api);

		//generate dense/sparse bodies
		String tmpDense = _output.codegen(false, api) + getOutputStatement(_output.getVarname());
		_output.resetGenerated();
		String tmpSparse = _output.codegen(true, api) + getOutputStatement(_output.getVarname());
		_output.resetGenerated();
		String varName = createVarname();
		tmp = tmp.replace(api.isJava()?"%TMP%":"//%TMP%", varName);
		if( !api.isJava() )
			tmp = tmp.replace("/*%TMP%*/SPOOF_OP_NAME", varName);
		String prefix = api.isJava()? "" : "//";
		tmp = tmp.replace(prefix+"%BODY_dense%", tmpDense);
		tmp = tmp.replace(prefix+"%BODY_sparse%", tmpSparse);
		
		//replace outputs 
		tmp = api.isJava() ? tmp.replace("%OUT%", "c") :
			tmp.replace("%OUT%", "c.vals(0)");
		tmp = tmp.replace("%POSOUT%", "0");
		
		//replace size information
		tmp = tmp.replace("%LEN%", "a.cols()");
		
		//replace colvector information and number of vector intermediates
		tmp = tmp.replace("%TYPE%", _type.name());
		tmp = tmp.replace("%CONST_DIM2%", String.valueOf(_constDim2));
		_tb1 = TemplateUtils.containsBinary(_output, BinType.VECT_MATRIXMULT); 
		tmp = tmp.replace("%TB1%", String.valueOf(_tb1));

		if(api == GeneratorAPI.CUDA && _numVectors > 0) {
			tmp = tmp.replace("//%HAS_TEMP_VECT%", ": public TempStorageImpl<T, NUM_TMP_VECT, TMP_VECT_LEN>");
			tmp = tmp.replace("/*%INIT_TEMP_VECT%*/", ", TempStorageImpl<T, NUM_TMP_VECT, TMP_VECT_LEN>(tmp_stor)");
		}
		else {
			tmp = tmp.replace("//%HAS_TEMP_VECT%", "");
			tmp = tmp.replace("/*%INIT_TEMP_VECT%*/", "");
		}
		tmp = tmp.replace("%VECT_MEM%", String.valueOf(_numVectors));

		return tmp;
	}
	
	@SuppressWarnings("fallthrough")
	private String getOutputStatement(String varName) {
		switch( _type ) {
			case NO_AGG:
				if(api == GeneratorAPI.CUDA)
					return TEMPLATE_NOAGG_OUT_CUDA.replace("%IN%", varName + ".vals(0)").replaceAll("%LEN%", _output.getVarname()+".length");
			case NO_AGG_B1:
			case NO_AGG_CONST:
				if(api == GeneratorAPI.JAVA)
					return TEMPLATE_NOAGG_OUT.replace("%IN%", varName.startsWith("STMP")?varName+".values(), "+varName+".indexes()":varName).replace("%LEN%",
						varName.startsWith("STMP") ? varName+".size()" : _output.getVarname()+".length");
				else
					return TEMPLATE_NOAGG_CONST_OUT_CUDA.replace("%IN%", varName + ".vals(0)").replaceAll("%LEN%", _output.getVarname()+".length");
			case FULL_AGG:
				if(api == GeneratorAPI.JAVA)
					return TEMPLATE_FULLAGG_OUT.replace("%IN%", varName);
				else
					return TEMPLATE_FULLAGG_OUT_CUDA.replace("%IN%", varName);
			case ROW_AGG:
				if(api == GeneratorAPI.JAVA)
					return TEMPLATE_ROWAGG_OUT.replace("%IN%", varName);
				else
					return TEMPLATE_ROWAGG_OUT_CUDA.replace("%IN%", varName);
			default:
				return ""; //_type.isColumnAgg()
		}
	}

	@Override
	public void setOutputDims() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public SpoofOutputDimsType getOutputDimType() {
		switch( _type ) {
			case NO_AGG:        return SpoofOutputDimsType.INPUT_DIMS;
			case NO_AGG_B1:     return SpoofOutputDimsType.ROW_RANK_DIMS;
			case NO_AGG_CONST:  return SpoofOutputDimsType.INPUT_DIMS_CONST2; 
			case FULL_AGG:      return SpoofOutputDimsType.SCALAR;
			case ROW_AGG:       return SpoofOutputDimsType.ROW_DIMS;
			case COL_AGG:       return SpoofOutputDimsType.COLUMN_DIMS_COLS; //row vector
			case COL_AGG_T:     return SpoofOutputDimsType.COLUMN_DIMS_ROWS; //column vector
			case COL_AGG_B1:    return SpoofOutputDimsType.COLUMN_RANK_DIMS; 
			case COL_AGG_B1_T:  return SpoofOutputDimsType.COLUMN_RANK_DIMS_T;
			case COL_AGG_B1R:   return SpoofOutputDimsType.RANK_DIMS_COLS;
			case COL_AGG_CONST: return SpoofOutputDimsType.VECT_CONST2;
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
			int h = UtilFunctions.intHashCode(super.hashCode(), _type.hashCode());
			h = UtilFunctions.intHashCode(h, Long.hashCode(_constDim2));
			_hash = UtilFunctions.intHashCode(h, Integer.hashCode(_numVectors));
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
			&& _constDim2 == that._constDim2
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

	@Override
	public boolean isSupported(GeneratorAPI api) {
		return (api == GeneratorAPI.CUDA || api == GeneratorAPI.JAVA) && _output.isSupported(api);
	}

	public int compile(GeneratorAPI api, String src) {
		if(api == GeneratorAPI.CUDA)
			return compile_nvrtc(SpoofCompiler.native_contexts.get(api), _genVar, src, _type.getValue(), _constDim2, 
				_numVectors, _tb1);
		return -1;
	}
	
	private native int compile_nvrtc(long context, String name, String src, int type, long constDim2, int numVectors, 
			boolean TB1);
}
