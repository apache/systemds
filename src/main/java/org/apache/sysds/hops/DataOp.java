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

package org.apache.sysds.hops;

import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.Data;
import org.apache.sysds.lops.Federated;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.lops.LopsException;
import org.apache.sysds.lops.Sql;
import org.apache.sysds.lops.Tee;
import org.apache.sysds.parser.DataExpression;
import static org.apache.sysds.parser.DataExpression.FED_RANGES;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.LocalFileUtils;

/**
 *  A DataOp can be either a persistent read/write or transient read/write - writes will always have at least one input,
 *  but all types can have parameters (e.g., for csv literals of delimiter, header, etc).
 */
public class DataOp extends Hop {
	private static final Log LOG =  LogFactory.getLog(DataOp.class.getName());
	private OpOpData _op;
	private String _fileName = null;
	
	//read dataop properties
	private FileFormat _inFormat = FileFormat.TEXT;
	private long _inBlocksize = -1;
	private boolean _hasOnlyRDD = false;
	
	private boolean _recompileRead = true;

	private boolean _isTeeOp = false;

	/**
	 * List of "named" input parameters. They are maintained as a hashmap:
	 * parameter names (String) are mapped as indices (Integer) into getInput()
	 * arraylist.
	 * 
	 * i.e., getInput().get(_paramIndexMap.get(parameterName)) refers to the Hop
	 * that is associated with parameterName.
	 */
	private HashMap<String, Integer> _paramIndexMap = new HashMap<>();

	private DataOp() {
		//default constructor for clone
	}

	public void setIsTeeOp(boolean isTeeOp) {
		this._isTeeOp = isTeeOp;
	}
	
	/**
	 * READ operation for Matrix w/ dim1, dim2. 
	 * This constructor does not support any expression in parameters
	 * 
	 * @param l ?
	 * @param dt data type
	 * @param vt value type
	 * @param dop data operator type
	 * @param fname file name
	 * @param dim1 dimension 1
	 * @param dim2 dimension 2
	 * @param nnz number of non-zeros
	 * @param blen rows/cols per block
	 */
	public DataOp(String l, DataType dt, ValueType vt, OpOpData dop,
			String fname, long dim1, long dim2, long nnz, int blen) {
		super(l, dt, vt);
		_op = dop;
		
		_fileName = fname;
		setDim1(dim1);
		setDim2(dim2);
		setBlocksize(blen);
		setNnz(nnz);
		
		if( dop == OpOpData.TRANSIENTREAD )
			setFileFormat(FileFormat.BINARY);
	}

	public DataOp(String l, DataType dt, ValueType vt, OpOpData dop,
			String fname, long dim1, long dim2, long nnz, UpdateType update, int blen) {
		this(l, dt, vt, dop, fname, dim1, dim2, nnz, blen);
		setUpdateType(update);
	}

	/**
	 * READ operation for Matrix / SQL operation for tensor
	 * This constructor supports expressions in parameters
	 * 
	 * @param l ?
	 * @param dt data type
	 * @param vt value type
	 * @param dop data operator type
	 * @param params input parameters
	 */
	public DataOp(String l, DataType dt, ValueType vt,
			OpOpData dop, HashMap<String, Hop> params) {
		super(l, dt, vt);
		_op = dop;

		int index = 0;
		for( Entry<String, Hop> e : params.entrySet() ) {
			String s = e.getKey();
			Hop input = e.getValue();
			getInput().add(input);
			if (LOG.isDebugEnabled()){
				LOG.debug(String.format("%15s - %s",s,input));
			}
			input.getParent().add(this);

			_paramIndexMap.put(s, index);
			index++;
		}
		if (dop == OpOpData.TRANSIENTREAD ){
			setFileFormat(FileFormat.BINARY);
		}
		
		if( params.containsKey(DataExpression.READROWPARAM) )
			setDim1(((LiteralOp)params.get(DataExpression.READROWPARAM)).getLongValue());
		if( params.containsKey(DataExpression.READCOLPARAM) )
			setDim2(((LiteralOp)params.get(DataExpression.READCOLPARAM)).getLongValue());
		if( params.containsKey(DataExpression.READNNZPARAM) )
			setNnz(((LiteralOp)params.get(DataExpression.READNNZPARAM)).getLongValue());
	}
	
	// WRITE operation
	// This constructor does not support any expression in parameters
	public DataOp(String l, DataType dt, ValueType vt, Hop in,
			OpOpData dop, String fname) {
		super(l, dt, vt);
		_op = dop;
		getInput().add(0, in);
		in.getParent().add(this);
		_fileName = fname;

		if (dop == OpOpData.TRANSIENTWRITE || dop == OpOpData.FUNCTIONOUTPUT )
			setFileFormat(FileFormat.BINARY);
	}
	
	/**
	 *  WRITE operation for Matrix
	 *  This constructor supports expression in parameters
	 * 
	 * @param l ?
	 * @param dt data type
	 * @param vt value type
	 * @param dop data operator type
	 * @param in high-level operator
	 * @param inputParameters input parameters
	 */
	public DataOp(String l, DataType dt, ValueType vt, 
		OpOpData dop, Hop in, HashMap<String, Hop> inputParameters) {
		super(l, dt, vt);
		_op = dop;
		
		getInput().add(0, in);
		in.getParent().add(this);
		
		if (inputParameters != null){
			int index = 1;
			for( Entry<String, Hop> e : inputParameters.entrySet() ) 
			{
				String s = e.getKey();
				Hop input = e.getValue();
				getInput().add(input);
				input.getParent().add(this);

				_paramIndexMap.put(s, index);
				index++;
			}
		
		}

		if (dop == OpOpData.TRANSIENTWRITE)
			setFileFormat(FileFormat.BINARY);
	}

	public OpOpData getOp() {
		return _op;
	}
	
	public void setDataOpType(OpOpData type) {
		_op = type;
	}
	
	public void setOutputParams(long dim1, long dim2, long nnz, UpdateType update, int blen) {
		setDim1(dim1);
		setDim2(dim2);
		setNnz(nnz);
		setUpdateType(update);
		setBlocksize(blen);
	}

	public void setFileName(String fn) {
		_fileName = fn;
	}

	public String getFileName() {
		return _fileName;
	}

	public int getParameterIndex(String name) {
		return _paramIndexMap.get(name);
	}
	
	public void setOnlyRDD(boolean flag) {
		_hasOnlyRDD = flag;
	}
	
	public boolean hasOnlyRDD() {
		return _hasOnlyRDD;
	}
	
	@Override
	public boolean isGPUEnabled() {
		return false;
	}
	
	@Override
	public Lop constructLops()
	{
		//return already created lops
		if( getLops() != null )
			return getLops();

		ExecType et = optFindExecType();
		Lop l = null;

		if (_isTeeOp) {
			Tee teeLop = new Tee(getInput().get(0).constructLops(),
							getDataType(), getValueType());
			setLineNumbers(teeLop);
			setLops(teeLop);
			setOutputDimensions(teeLop);
		}
		else {

			// construct lops for all input parameters
			HashMap<String, Lop> inputLops = new HashMap<>();
			for (Entry<String, Integer> cur : _paramIndexMap.entrySet()) {
				inputLops.put(cur.getKey(), getInput().get(cur.getValue()).constructLops());
			}

			// Create the lop
			switch (_op) {
				case TRANSIENTREAD:
					l = new Data(_op, null, inputLops, getName(), null,
									getDataType(), getValueType(), getFileFormat());
					setOutputDimensions(l);
					break;

				case PERSISTENTREAD:
					l = new Data(_op, null, inputLops, getName(), null,
									getDataType(), getValueType(), getFileFormat());
					l.getOutputParameters().setDimensions(getDim1(), getDim2(), _inBlocksize, getNnz(), getUpdateType());
					break;

				case PERSISTENTWRITE:
				case FUNCTIONOUTPUT:
					l = new Data(_op, getInput().get(0).constructLops(), inputLops, getName(), null,
									getDataType(), getValueType(), getFileFormat());
					((Data) l).setExecType(et);
					setOutputDimensions(l);
					break;

				case TRANSIENTWRITE:
					l = new Data(_op, getInput().get(0).constructLops(), inputLops, getName(), null,
									getDataType(), getValueType(), getFileFormat());
					setOutputDimensions(l);
					break;

				case SQLREAD:
					l = new Sql(inputLops, getDataType(), getValueType());
					break;

				case FEDERATED:
					l = new Federated(inputLops, getDataType(), getValueType());
					break;

				default:
					throw new LopsException("Invalid operation type for Data LOP: " + _op);
			}
			setLineNumbers(l);
			setLops(l);
		}
		
		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();

		return getLops();

	}

	public void setFileFormat(FileFormat ft) {
		_inFormat = ft;
	}

	public FileFormat getFileFormat() {
		return _inFormat;
	}
	
	public void setInputBlocksize(long blen){
		_inBlocksize = blen;
	}
	
	public long getInputBlocksize(){
		return _inBlocksize;
	}
	
	public boolean isRead() {
		return( _op == OpOpData.PERSISTENTREAD || _op == OpOpData.TRANSIENTREAD );
	}
	
	public boolean isWrite() {
		return( _op == OpOpData.PERSISTENTWRITE || _op == OpOpData.TRANSIENTWRITE );
	}
	
	public boolean isPersistentReadWrite() {
		return( _op == OpOpData.PERSISTENTREAD || _op == OpOpData.PERSISTENTWRITE );
	}

	@Override
	public boolean isFederatedDataOp(){
		return _op == OpOpData.FEDERATED;
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += _op.toString();
		if (_isTeeOp) {
			s += " tee";
		}
		s += " "+getName();
		return s;
	}

	@Override
	public boolean allowsAllExecTypes()
	{
		return false;
	}	
	
	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		double ret = 0;
		final DataType dt = getDataType();
		if ( dt == DataType.SCALAR ) 
		{
			switch( getValueType() ) 
			{
				case INT64:
					ret = OptimizerUtils.INT_SIZE; break;
				case FP64:
					ret = OptimizerUtils.DOUBLE_SIZE; break;
				case BOOLEAN:
					ret = OptimizerUtils.BOOLEAN_SIZE; break;
				case STRING: 
					// by default, it estimates the size of string[100]
					ret = 100 * OptimizerUtils.CHAR_SIZE; break;
				case UNKNOWN:
					ret = OptimizerUtils.DEFAULT_SIZE; break;
				default:
					ret = 0;
			}
		}
		else if(dt == DataType.FRAME) {
			if(_op == OpOpData.PERSISTENTREAD || _op == OpOpData.TRANSIENTREAD) {
				ret = OptimizerUtils.estimateSizeExactFrame(dim1, dim2);
			}
		}
		else //MATRIX / FRAME
		{
			if(   _op == OpOpData.PERSISTENTREAD 
			   || _op == OpOpData.TRANSIENTREAD ) 
			{
				double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
				ret = OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity, getDataType());
			}
			// output memory estimate is not required for "write" nodes (just input)
		}
		return ret;
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz ) {
		return LocalFileUtils.BUFFER_SIZE;
	}
	
	@Override
	protected DataCharacteristics inferOutputCharacteristics( MemoTable memo ) {
		DataCharacteristics ret = null;
		if( _op == OpOpData.PERSISTENTWRITE || _op == OpOpData.TRANSIENTWRITE )  {
			DataCharacteristics tmp = memo.getAllInputStats(getInput().get(0));
			if( tmp.dimsKnown() )
				ret = tmp;
		}
		else if( _op == OpOpData.TRANSIENTREAD ) {
			//prepare statistics, passed from cross-dag transient writes
			DataCharacteristics tmp = memo.getAllInputStats(this);
			if( tmp.dimsKnown() )
				ret = tmp;
		}
		return ret;
	}
	
	@Override
	protected ExecType optFindExecType(boolean transitive) 
	{
		//MB: find exec type has two meanings here: (1) for write it means the actual
		//exec type, while (2) for read it affects the recompilation decision as needed
		//for example for sum(X) where the memory consumption is solely determined by the DataOp
		
		ExecType letype = (OptimizerUtils.isMemoryBasedOptLevel()) ? findExecTypeByMemEstimate() : null;
		
		//NOTE: independent of etype executed in MR (piggybacked) if input to persistent write is MR
		if( _op == OpOpData.PERSISTENTWRITE || _op == OpOpData.TRANSIENTWRITE )
		{
			checkAndSetForcedPlatform();

			//additional check for write only
			if( getDataType()==DataType.SCALAR )
				_etypeForced = ExecType.CP;
			
			if( _etypeForced != null )
			{
				_etype = _etypeForced;
			}
			else 
			{
				if ( OptimizerUtils.isMemoryBasedOptLevel() ) 
				{
					_etype = letype;
				}
				else if ( getInput().get(0).areDimsBelowThreshold() )
				{
					_etype = ExecType.CP;
				}
				else
				{
					_etype = ExecType.SPARK;
				}
			
				//check for valid CP dimensions and matrix size
				checkAndSetInvalidCPDimsAndSize();
			}
			
			//mark for recompile (forever)
			setRequiresRecompileIfNecessary();
		}
		else //READ
		{
			if( DMLScript.USE_OOC )
				checkAndSetForcedPlatform();
			
			//mark for recompile (forever)
			if( ConfigurationManager.isDynamicRecompilation() && !dimsKnown(true) && letype==ExecType.SPARK 
				&& (_recompileRead || _requiresCheckpoint) ) 
			{
				setRequiresRecompile();
			}
			
			_etype = letype;
			if ( _etypeForced == ExecType.FED || _etypeForced == ExecType.OOC )
				_etype = _etypeForced;
		}

		return _etype;
	}

	@Override
	public void refreshSizeInformation() {
		if( _op == OpOpData.PERSISTENTWRITE || _op == OpOpData.TRANSIENTWRITE ) {
			Hop input1 = getInput().get(0);
			setDim1(input1.getDim1());
			setDim2(input1.getDim2());
			setNnz(input1.getNnz());
		}
		else if( _op == OpOpData.FEDERATED ) {
			Hop ranges = getInput().get(getParameterIndex(FED_RANGES));
			long nrow = -1, ncol = -1;
			for( Hop c : ranges.getInput() ) {
				if( !(c.getInput(0) instanceof LiteralOp && c.getInput(1) instanceof LiteralOp))
					return; // invalid size inference if not all know.
				nrow = Math.max(nrow, HopRewriteUtils.getIntValueSafe(c.getInput(0)));
				ncol = Math.max(ncol, HopRewriteUtils.getIntValueSafe(c.getInput(1)));
			}
			setDim1(nrow);
			setDim2(ncol);
		}
		else { //READ
			//do nothing; dimensions updated via set output params
		}
	}

	
	/**
	 * Explicitly disables recompilation of transient reads, this additional information 
	 * is required because requiresRecompile is set in a top-down manner, hence any value
	 * set from a consuming operating would be overwritten by opFindExecType.
	 */
	public void disableRecompileRead()
	{
		_recompileRead = false;
	}
	
	
	@Override
	@SuppressWarnings("unchecked")
	public Object clone() throws CloneNotSupportedException 
	{
		DataOp ret = new DataOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._op = _op;
		ret._fileName = _fileName;
		ret._inFormat = _inFormat;
		ret._inBlocksize = _inBlocksize;
		ret._recompileRead = _recompileRead;
		ret._isTeeOp = _isTeeOp; // copy the Tee flag
		ret._paramIndexMap = (HashMap<String, Integer>) _paramIndexMap.clone();
		//note: no deep cp of params since read-only 
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof DataOp) )
			return false;
		
		//common subexpression elimination for redundant persistent reads, in order
		//to avoid unnecessary read and reblocks as well as to prevent specific anomalies, e.g., 
		//with multiple piggybacked csvreblock of the same input w/ unknown input sizes
		
		DataOp that2 = (DataOp)that;
		boolean ret = ( OptimizerUtils.ALLOW_COMMON_SUBEXPRESSION_ELIMINATION 
			&& ConfigurationManager.getCompilerConfigFlag(ConfigType.ALLOW_CSE_PERSISTENT_READS) 
			&& _op == that2._op
			&& _op == OpOpData.PERSISTENTREAD
			&& _fileName.equals(that2._fileName)
			&& _inFormat == that2._inFormat
			&& _inBlocksize == that2._inBlocksize
			&& _paramIndexMap!=null && that2._paramIndexMap!=null );
		
		//above conditions also ensure consistency with regard to 
		//(1) checkpointing, (2) reblock and (3) recompile.
		
		if( ret ) {
			for( Entry<String,Integer> e : _paramIndexMap.entrySet() ) {
				String key1 = e.getKey();
				int pos1 = e.getValue();
				int pos2 = that2._paramIndexMap.get(key1);
				ret &= (   that2.getInput().get(pos2)!=null
					    && getInput().get(pos1) == that2.getInput().get(pos2) );
			}
		}
		
		return ret;
	}

	/**
	 * Remove an input from the list of inputs and from the parameter index map.
	 * Parameter index map values higher than the index of the removed input
	 * will be decremented by one.
	 * 
	 * @param inputName The name of the input to remove
	 */
	public void removeInput(String inputName) {
		int inputIndex = getParameterIndex(inputName);
		Hop tmp = _input.remove(inputIndex);
		tmp._parent.remove(this);
		_paramIndexMap.remove(inputName);
		for (Entry<String, Integer> entry : _paramIndexMap.entrySet()) {
			if (entry.getValue() > inputIndex) {
				_paramIndexMap.put(entry.getKey(), (entry.getValue() - 1));
			}
		}
	}
}
