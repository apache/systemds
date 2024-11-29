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

package org.apache.sysds.hops.codegen;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.codegen.SpoofCompiler.GeneratorAPI;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.MemoTable;
import org.apache.sysds.hops.MultiThreadedHop;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.lops.SpoofFused;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.codegen.SpoofOperator;
import org.apache.sysds.runtime.codegen.SpoofRowwise;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;

public class SpoofFusedOp extends MultiThreadedHop
{
	public enum SpoofOutputDimsType {
		INPUT_DIMS,
		INPUT_DIMS_CONST2,
		ROW_DIMS,
		COLUMN_DIMS_ROWS,
		COLUMN_DIMS_COLS,
		RANK_DIMS_COLS,
		SCALAR,
		MULTI_SCALAR,
		ROW_RANK_DIMS, // right wdivmm, row mm
		COLUMN_RANK_DIMS,  // left wdivmm, row mm
		COLUMN_RANK_DIMS_T,
		VECT_CONST2;
	}
	
	private Class<?> _class = null;
	private boolean _distSupported = false;
	private long _constDim2 = -1;
	private SpoofOutputDimsType _dimsType;
	private GeneratorAPI _api = GeneratorAPI.JAVA;
	private String _genVarName;

	public SpoofFusedOp ( ) {
	
	}
	
	public SpoofFusedOp( String name, DataType dt, ValueType vt, Class<?> cla, GeneratorAPI api,
		String genVarName, boolean dist, SpoofOutputDimsType type ) {
		super(name, dt, vt);
		_class = cla;
		_distSupported = dist;
		_dimsType = type;
		_api = api;
		_genVarName = genVarName;
	}

	@Override
	public boolean allowsAllExecTypes() {
		return _distSupported;
	}
	
	public void setConstDim2(long constDim2) {
		_constDim2 = constDim2;
	}
	
	@Override
	public boolean isGPUEnabled() {
		if(_api == GeneratorAPI.CUDA)
			return true;
		else
			return false;
	}
	
	@Override
	public boolean isMultiThreadedOpType() {
		return true;
	}

	@Override
	protected double computeOutputMemEstimate(long dim1, long dim2, long nnz) {
		// The output estimate influences the ExecType decision as usual, 
		// but for codegen operators also various fusion decisions in both 
		// local and distributed environments. For that reason, we use the 
		// partitioned size as a more conservative estimate - for dense this
		// is almost the same, but for sparse it includes the block indexes
		// and overhead of row arrays per block. In forced singlenode exec
		// mode, the blocksize is however -1 and need appropriate treatment.
		boolean onlyDenseOut = (_api == GeneratorAPI.JAVA
			&& _class.getGenericSuperclass().equals(SpoofRowwise.class));
		int blen = (getBlocksize() > 0) ? getBlocksize() : ConfigurationManager.getBlocksize();
		return onlyDenseOut ?
			OptimizerUtils.estimateSize(dim1, dim2) :
			OptimizerUtils.estimatePartitionedSizeExactSparsity(dim1, dim2, blen, nnz);
	}

	@Override
	protected double computeIntermediateMemEstimate(long dim1, long dim2, long nnz) {
		if( _class.getGenericSuperclass().equals(SpoofRowwise.class) ) {
			long[] cols = new long[getInput().size()];
			Arrays.setAll(cols, i -> getInput(i).getDim2());
			SpoofOperator op = CodegenUtils.createInstance(_class);
			int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
			return ((SpoofRowwise)op).getTmpMemoryReq(k, cols[0], cols);
		}
		return 0;
	}
	
	@Override
	public Lop constructLops() {
		if( getLops() != null )
			return getLops();
		
		ExecType et = optFindExecType();
		
		ArrayList<Lop> inputs = new ArrayList<>();
		for( Hop c : getInput() )
			inputs.add(c.constructLops());
		
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		SpoofFused lop = new SpoofFused(inputs, getDataType(), getValueType(), _class, _api, _genVarName, k, et);
		setOutputDimensions(lop);
		setLineNumbers(lop);
		setLops(lop);
	
		return lop;
	}
	
	@Override
	protected ExecType optFindExecType(boolean transitive) {
		
		checkAndSetForcedPlatform();
		
		if( _etypeForced != null ) {
			_etype = _etypeForced;
		}
		else {
			_etype = findExecTypeByMemEstimate();
			checkAndSetInvalidCPDimsAndSize();
		}
		
		return _etype;
	}

	@Override
	public String getOpString() {
		if(_class != null)
			return "spoof("+_class.getSimpleName()+")";
		else
			return "spoof(" + getName() + ")";	}
	
	public String getClassName() {
		if(_class != null)
			return _class.getName();
		else
			return "spoof" + getName();	}
	
	@Override
	protected DataCharacteristics inferOutputCharacteristics( MemoTable memo )
	{
		//get statistics of main input
		DataCharacteristics mc = memo.getAllInputStats(getInput().get(0));
		DataCharacteristics ret = null;
		
		if( mc.dimsKnown() ) {
			switch(_dimsType)
			{
				case ROW_DIMS:
					ret = new MatrixCharacteristics(mc.getRows(), 1, -1, -1);
					break;
				case COLUMN_DIMS_ROWS:
					ret = new MatrixCharacteristics(mc.getCols(), 1, -1, -1);
					break;
				case COLUMN_DIMS_COLS:
					ret = new MatrixCharacteristics(1, mc.getCols(), -1, -1);
					break;
				case RANK_DIMS_COLS: {
					DataCharacteristics dc2 = memo.getAllInputStats(getInput().get(1));
					if( dc2.dimsKnown() )
						ret = new MatrixCharacteristics(1, dc2.getCols(), -1, -1);
					break;
				}
				case INPUT_DIMS:
					ret = new MatrixCharacteristics(mc.getRows(), mc.getCols(), -1, -1);
					break;
				case INPUT_DIMS_CONST2:
					ret = new MatrixCharacteristics(mc.getRows(), _constDim2, -1, -1);
					break;
				case VECT_CONST2:
					ret = new MatrixCharacteristics(1, _constDim2, -1, -1);
					break;	
				case SCALAR:
					ret = new MatrixCharacteristics(0, 0, -1, -1);
					break;
				case MULTI_SCALAR:
					//dim2 statically set from outside
					ret = new MatrixCharacteristics(1, _dc.getCols(), -1, -1);
					break;
				case ROW_RANK_DIMS: {
					DataCharacteristics dc2 = memo.getAllInputStats(getInput().get(1));
					if( dc2.dimsKnown() )
						ret = new MatrixCharacteristics(mc.getRows(), dc2.getCols(), -1, -1);
					break;
				}
				case COLUMN_RANK_DIMS: {
					DataCharacteristics dc2 = memo.getAllInputStats(getInput().get(1));
					if( dc2.dimsKnown() )
						ret = new MatrixCharacteristics(mc.getCols(), dc2.getCols(), -1, -1);
					break;
				}
				case COLUMN_RANK_DIMS_T: {
					DataCharacteristics dc2 = memo.getAllInputStats(getInput().get(1));
					if( dc2.dimsKnown() )
						ret = new MatrixCharacteristics(dc2.getCols(), mc.getCols(), -1, -1);
					break;
				}
				default:
					throw new RuntimeException("Failed to infer worst-case size information "
							+ "for type: "+_dimsType.toString());
			}
		}
		return ret;
	}
	
	@Override
	public void refreshSizeInformation() {
		switch(_dimsType)
		{
			case ROW_DIMS:
				setDim1(getInput().get(0).getDim1());
				setDim2(1);
				break;
			case COLUMN_DIMS_ROWS:
				setDim1(getInput().get(0).getDim2());
				setDim2(1);
				break;
			case COLUMN_DIMS_COLS:
				setDim1(1);
				setDim2(getInput().get(0).getDim2());
				break;
			case RANK_DIMS_COLS:
				setDim1(1);
				setDim2(getInput().get(1).getDim2());
				break;
			case INPUT_DIMS:
				setDim1(getInput().get(0).getDim1());
				setDim2(getInput().get(0).getDim2());
				break;
			case INPUT_DIMS_CONST2:
				setDim1(getInput().get(0).getDim1());
				setDim2(_constDim2);
				break;
			case VECT_CONST2:
				setDim1(1);
				setDim2(_constDim2);
				break;
			case SCALAR:
				setDim1(0);
				setDim2(0);
				break;
			case MULTI_SCALAR:
				setDim1(1); //row vector
				//dim2 statically set from outside
				break;
			case ROW_RANK_DIMS:
				setDim1(getInput().get(0).getDim1());
				setDim2(getInput().get(1).getDim2());
				break;
			case COLUMN_RANK_DIMS:
				setDim1(getInput().get(0).getDim2());
				setDim2(getInput().get(1).getDim2());
				break;
			case COLUMN_RANK_DIMS_T:
				setDim1(getInput().get(1).getDim2());
				setDim2(getInput().get(0).getDim2());
				break;	
			default:
				throw new RuntimeException("Failed to refresh size information "
					+ "for type: "+_dimsType.toString());
		}
	}

	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		SpoofFusedOp ret = new SpoofFusedOp();
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._class = _class;
		ret._distSupported = _distSupported;
		ret._maxNumThreads = _maxNumThreads;
		ret._constDim2 = _constDim2;
		ret._dimsType = _dimsType;
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof SpoofFusedOp) )
			return false;
		
		SpoofFusedOp that2 = (SpoofFusedOp)that;
		//note: class implies dims type as well
		boolean ret = (Objects.equals(_class, that2._class)
				&& _distSupported == that2._distSupported
				&& _maxNumThreads == that2._maxNumThreads
				&& _constDim2 == that2._constDim2
				&& getInput().size() == that2.getInput().size()
				&& _api == that2._api);
		
		if( ret ) {
			for( int i=0; i<getInput().size(); i++ )
				ret &= (getInput().get(i) == that2.getInput().get(i));
		}
		
		return ret;
	}
}
