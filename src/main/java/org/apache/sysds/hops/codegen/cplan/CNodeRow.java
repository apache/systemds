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
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.stream.Collectors;

import org.apache.sysds.hops.codegen.SpoofFusedOp.SpoofOutputDimsType;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.hops.codegen.template.TemplateUtils;
import org.apache.sysds.runtime.codegen.SpoofRowwise.RowType;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.hops.codegen.SpoofCompiler.GeneratorAPI;

import static org.apache.sysds.runtime.matrix.data.LibMatrixNative.isSinglePrecision;

public class CNodeRow extends CNodeTpl
{
	private static final String TEMPLATE_ROWAGG_OUT  = "    c[rix] = %IN%;\n";
	private static final String TEMPLATE_FULLAGG_OUT = "    c[0] += %IN%;\n";
	private static final String TEMPLATE_NOAGG_OUT   = "    LibSpoofPrimitives.vectWrite(%IN%, c, ci, %LEN%);\n";
	private static final String TEMPLATE_NOAGG_CONST_OUT_CUDA   = "\t\tvectWrite(%IN%, c.vals(0), rix * %LEN%, ci, %LEN%);\n";
	private static final String TEMPLATE_NOAGG_OUT_CUDA   = "\t\tvectWrite(%IN%, c.vals(0), 0, ci, %LEN%);\n";
	private static final String TEMPLATE_ROWAGG_OUT_CUDA  = "\t\tif(threadIdx.x == 0){\n\t\t\t*(c.vals(rix)) = %IN%;\n//printf(\"rix=%d TMP7=%f TMP8=%f %IN%=%f\\n\",rix, TMP7, TMP8,%IN%);\n}\n";
	private static final String TEMPLATE_FULLAGG_OUT_CUDA =
		"\t\tif(threadIdx.x == 0) {\n\t\t\tT old = atomicAdd(c.vals(0), %IN%);\n//\t\t\tprintf(\"bid=%d full_agg add %f to %f\\n\",blockIdx.x, %IN%, old);\n\t\t}\n";


	public CNodeRow(ArrayList<CNode> inputs, CNode output ) {
		super(inputs, output);
	}
	
	private RowType _type = null; //access pattern 
	private long _constDim2 = -1; //constant number of output columns
	private int _numVectors = -1; //number of intermediate vectors
	
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
		tmp = tmp.replace("//%TMP%", varName);
		tmp = tmp.replace("/*%TMP%*/SPOOF_OP_NAME", varName);
		tmp = tmp.replace("//%BODY_dense%", tmpDense);
		tmp = tmp.replace("//%BODY_sparse%", tmpSparse);
		
		//replace outputs 
		tmp = api == GeneratorAPI.JAVA ? tmp.replace("%OUT%", "c") :
				tmp.replace("%OUT%", "c.vals(0)");
		tmp = tmp.replace("%POSOUT%", "0");
		
		//replace size information
		tmp = tmp.replace("%LEN%", "a.cols()");
		
		//replace colvector information and number of vector intermediates
		tmp = tmp.replace("%TYPE%", _type.name());
		tmp = tmp.replace("%CONST_DIM2%", String.valueOf(_constDim2));
		tmp = tmp.replace("%TB1%", String.valueOf(
			TemplateUtils.containsBinary(_output, BinType.VECT_MATRIXMULT)));

		if(api == GeneratorAPI.CUDA && _numVectors > 0) {
			StringBuilder tmp_stor_str = new StringBuilder();
			StringBuilder tmp_stor_str_dec = new StringBuilder();

			tmp_stor_str.append("TMP_STORAGE = tmp_stor;\n");
			tmp_stor_str.append("\t\ttmp_row_offset = TMP_VECT_LEN * tmp_count * blockIdx.x;\n");
			tmp_stor_str.append("\t\ttemp_rb.init(tmp_row_offset, TMP_VECT_LEN, tmp_stor);\n");

			tmp_stor_str_dec.append(
					"T* TMP_STORAGE;\n" +
					"\tuint32_t tmp_row_offset;\n" +
					"\tRingBuffer<T,NUM_TMP_VECT> temp_rb;\n");

//			int seq_id = 0;
//			for(int i = 0; i < _numVectors; ++i) {
//				if(tmp.contains("tmp_offset_")) {
//					tmp = tmp.replaceFirst("tmp_offset_", "tmp_offset" + seq_id++);
//					tmp_stor_str.append("\t\ttmp_offset" + i + " = tmp_len*" + i + ";\n");
//					tmp_stor_str_dec.append("\tuint32_t tmp_offset" + i + ";\n");
//				}
//			}
//			_numVectors = seq_id;
			tmp_stor_str_dec.append("\tuint32_t tmp_count = " + _numVectors + ";\n");
//			tmp = tmp.replace("//%TMP_MEM%", tmp_stor_str.toString());
//			tmp = tmp.replace("//%TMP_MEM_DECLARATION%", tmp_stor_str_dec.toString());
			
			String hasTempVectorStorage = ": public TempStorageImpl<T, NUM_TMP_VECT, TMP_VECT_LEN>";
			String initTempVectorStorage = "TempStorageImpl<T, NUM_TMP_VECT, TMP_VECT_LEN>(tmp_stor),";
//			String getTempStorage = "\t__device__ Vector<T>& getTempStorage(uint32_t len) {\n" +
//				"\t\tVector<T>& vec = temp_rb.next();\n" +
//				"\t\tvec.length = len;\n" +
//				"\t\treturn vec;\n" +
//				"\t}\n";
//			tmp = tmp.replace("//%GET_TEMP_STORAGE%", getTempStorage);
			tmp = tmp.replace("//%HAS_TEMP_VECT%", hasTempVectorStorage);
			tmp = tmp.replace("//%INIT_TEMP_VECT%", initTempVectorStorage);
		}
		else {
			tmp = tmp.replace("//%TMP_MEM%", "");
			tmp = tmp.replace("//%TMP_MEM_DECLARATION%", "");
			tmp = tmp.replace("//%GET_TEMP_STORAGE%","");
			tmp = tmp.replace("//%HAS_TEMP_VECT%", "");
			tmp = tmp.replace("//%INIT_TEMP_VECT%", "");
			
			
		}
		tmp = tmp.replace("%VECT_MEM%", String.valueOf(_numVectors));

		//		// replace temp storage occurrences in CUDA code
//		if(api == GeneratorAPI.CUDA) {
//			String dType = isSinglePrecision() ? "float" : "double";
//			StringBuilder declarations = new StringBuilder();
//			Arrays.stream(tmp.split("\\r?\\n")).forEach(line -> {
//				if(line.contains("_STORAGE"))
//					declarations.append("__device__ " + dType + " " +
//						line.substring(line.indexOf("&TMP") + 1, line.indexOf("[0];")) + "[3072];\n");
//				});
//
//			if(!declarations.toString().isEmpty())
//				tmp = tmp.replace("//%TMP_MEM%", declarations.toString());
//			else
//				tmp = tmp.replace("//%TMP_MEM%", "");
//		}
		return tmp;
	}
	
	private String getOutputStatement(String varName) {
		switch( _type ) {
			case NO_AGG:
				if(api == GeneratorAPI.CUDA)
					return TEMPLATE_NOAGG_OUT_CUDA.replace("%IN%", varName + ".vals(0)") .replaceAll("%LEN%", _output.getVarname()+".length");
			case NO_AGG_B1:
			case NO_AGG_CONST:
				if(api == GeneratorAPI.JAVA)
					return TEMPLATE_NOAGG_OUT.replace("%IN%", varName) .replace("%LEN%", _output.getVarname()+".length");
				else
//					return "";
					return TEMPLATE_NOAGG_CONST_OUT_CUDA.replace("%IN%", varName + ".vals(0)") .replaceAll("%LEN%", _output.getVarname()+".length");
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
}
