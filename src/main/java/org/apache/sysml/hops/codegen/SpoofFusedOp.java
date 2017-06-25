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

package org.apache.sysml.hops.codegen;

import java.util.ArrayList;

import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.MultiThreadedHop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.MemoTable;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.lops.SpoofFused;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;

public class SpoofFusedOp extends Hop implements MultiThreadedHop
{
	public enum SpoofOutputDimsType {
		INPUT_DIMS,
		ROW_DIMS,
		ROW_DIMS2,
		COLUMN_DIMS_ROWS,
		COLUMN_DIMS_COLS,
		SCALAR,
		MULTI_SCALAR,
		ROW_RANK_DIMS, // right wdivmm 
		COLUMN_RANK_DIMS  // left wdivmm
	}
	
	private Class<?> _class = null;
	private boolean _distSupported = false;
	private int _numThreads = -1;
	private SpoofOutputDimsType _dimsType;
	
	public SpoofFusedOp ( ) {
	
	}
	
	public SpoofFusedOp( String name, DataType dt, ValueType vt, Class<?> cla, boolean dist, SpoofOutputDimsType type ) {
		super(name, dt, vt);
		_class = cla;
		_distSupported = dist;
		_dimsType = type;
	}

	@Override
	public void checkArity() throws HopsException {}

	@Override
	public void setMaxNumThreads(int k) {
		_numThreads = k;
	}

	@Override
	public int getMaxNumThreads() {
		return _numThreads;
	}

	@Override
	public boolean allowsAllExecTypes() {
		return _distSupported;
	}

	@Override
	protected double computeOutputMemEstimate(long dim1, long dim2, long nnz) {
		return OptimizerUtils.estimateSize(dim1, dim2);
	}

	@Override
	protected double computeIntermediateMemEstimate(long dim1, long dim2, long nnz) {
		return 0;
	}
	
	@Override
	public Lop constructLops() throws HopsException, LopsException {
		if( getLops() != null )
			return getLops();
		
		ExecType et = optFindExecType();
		
		ArrayList<Lop> inputs = new ArrayList<Lop>();
		for( Hop c : getInput() )
			inputs.add(c.constructLops());
		
		int k = OptimizerUtils.getConstrainedNumThreads(_numThreads);
		SpoofFused lop = new SpoofFused(inputs, getDataType(), getValueType(), _class, k, et);
		setOutputDimensions(lop);
		setLineNumbers(lop);
		setLops(lop);
	
		return lop;
	}
	
	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();
		
		if( _etypeForced != null ) {		
			_etype = _etypeForced;
		}
		else {
			_etype = findExecTypeByMemEstimate();
			checkAndSetInvalidCPDimsAndSize();
		}
		
		//ensure valid execution plans
		if( _etype == ExecType.MR )
			_etype = ExecType.CP;
		
		return _etype;
	}

	@Override
	public String getOpString() {
		return "spoof("+_class.getSimpleName()+")";
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		long[] ret = null;
	
		//get statistics of main input
		MatrixCharacteristics mc = memo.getAllInputStats(getInput().get(0));
		
		if( mc.dimsKnown() ) {
			switch(_dimsType)
			{
				case ROW_DIMS:
					ret = new long[]{mc.getRows(), 1, -1};
					break;
				case ROW_DIMS2:
					ret = new long[]{mc.getRows(), 2, -1};
					break;
				case COLUMN_DIMS_ROWS:
					ret = new long[]{mc.getCols(), 1, -1};
					break;
				case COLUMN_DIMS_COLS:
					ret = new long[]{1, mc.getCols(), -1};
					break;
				case INPUT_DIMS:
					ret = new long[]{mc.getRows(), mc.getCols(), -1};
					break;
				case SCALAR:
					ret = new long[]{0, 0, -1};
					break;
				case MULTI_SCALAR:
					//dim2 statically set from outside
					ret = new long[]{1, _dim2, -1};
					break;
				case ROW_RANK_DIMS: {
					MatrixCharacteristics mc2 = memo.getAllInputStats(getInput().get(1));
					if( mc2.dimsKnown() )
						ret = new long[]{mc.getRows(), mc2.getCols(), -1};
					break;
				}
				case COLUMN_RANK_DIMS: {
					MatrixCharacteristics mc2 = memo.getAllInputStats(getInput().get(1));
					if( mc2.dimsKnown() )
						ret = new long[]{mc.getCols(), mc2.getCols(), -1};
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
			case ROW_DIMS2:
				setDim1(getInput().get(0).getDim1());
				setDim2(2);
				break;
			case COLUMN_DIMS_ROWS:
				setDim1(getInput().get(0).getDim2());
				setDim2(1);
				break;
			case COLUMN_DIMS_COLS:
				setDim1(1);
				setDim2(getInput().get(0).getDim2());
				break;
			case INPUT_DIMS:
				setDim1(getInput().get(0).getDim1());
				setDim2(getInput().get(0).getDim2());
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
		ret._numThreads = _numThreads;
		ret._dimsType = _dimsType;
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof SpoofFusedOp) )
			return false;
		
		SpoofFusedOp that2 = (SpoofFusedOp)that;		
		boolean ret = ( _class.equals(that2._class)
				&& _distSupported == that2._distSupported
				&& _numThreads == that2._numThreads
				&& getInput().size() == that2.getInput().size());
		
		if( ret ) {
			for( int i=0; i<getInput().size(); i++ )
				ret &= (getInput().get(i) == that2.getInput().get(i));
		}
		
		return ret;
	}
}
