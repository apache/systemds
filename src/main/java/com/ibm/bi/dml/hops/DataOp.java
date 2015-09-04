/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;

import java.util.HashMap;
import java.util.Map.Entry;


public class DataOp extends Hop 
{

	private DataOpTypes _dataop;
	private String _fileName = null;
	
	//read dataop properties
	private FileFormatTypes _inFormat = FileFormatTypes.TEXT;
	private long _inRowsInBlock = -1;
	private long _inColsInBlock = -1;
	
	private boolean _recompileRead = true;
	
	/**
	 * List of "named" input parameters. They are maintained as a hashmap:
	 * parameter names (String) are mapped as indices (Integer) into getInput()
	 * arraylist.
	 * 
	 * i.e., getInput().get(_paramIndexMap.get(parameterName)) refers to the Hop
	 * that is associated with parameterName.
	 */
	private HashMap<String, Integer> _paramIndexMap = new HashMap<String, Integer>();

	private DataOp() {
		//default constructor for clone
	}
	
	/**
	 *  READ operation for Matrix w/ dim1, dim2. 
	 * This constructor does not support any expression in parameters
	 */
	public DataOp(String l, DataType dt, ValueType vt, DataOpTypes dop,
			String fname, long dim1, long dim2, long nnz, long rowsPerBlock, long colsPerBlock) {
		super(l, dt, vt);
		_dataop = dop;
		
		_fileName = fname;
		setDim1(dim1);
		setDim2(dim2);
		setRowsInBlock(rowsPerBlock);
		setColsInBlock(colsPerBlock);
		setNnz(nnz);
		
		if( dop == DataOpTypes.TRANSIENTREAD )
			setInputFormatType(FileFormatTypes.BINARY);
	}

	/**
	 * READ operation for Matrix
	 * This constructor supports expressions in parameters
	 */
	public DataOp(String l, DataType dt, ValueType vt, 
			DataOpTypes dop, HashMap<String, Hop> inputParameters) {
		super(l, dt, vt);

		_dataop = dop;

		int index = 0;
		for( Entry<String, Hop> e : inputParameters.entrySet() ) 
		{
			String s = e.getKey();
			Hop input = e.getValue();
			getInput().add(input);
			input.getParent().add(this);

			_paramIndexMap.put(s, index);
			index++;
		}
		if (dop == DataOpTypes.TRANSIENTREAD ){
			setInputFormatType(FileFormatTypes.BINARY);
		}
	}
	
	// WRITE operation
	// This constructor does not support any expression in parameters
	public DataOp(String l, DataType dt, ValueType vt, Hop in,
			DataOpTypes dop, String fname) {
		super(l, dt, vt);
		_dataop = dop;
		getInput().add(0, in);
		in.getParent().add(this);
		_fileName = fname;

		if (dop == DataOpTypes.TRANSIENTWRITE || dop == DataOpTypes.FUNCTIONOUTPUT )
			setInputFormatType(FileFormatTypes.BINARY);
	}
	
	// CHECKPOINT operation
	// This constructor does not support any expression in parameters
	public DataOp(String l, DataType dt, ValueType vt, Hop in,
			LiteralOp level, DataOpTypes dop, String fname) {
		super(l, dt, vt);
		_dataop = dop;
		getInput().add(0, in);
		getInput().add(1, level);
		in.getParent().add(this);
		level.getParent().add(this);
		_fileName = fname;

		if (dop == DataOpTypes.TRANSIENTWRITE || dop == DataOpTypes.FUNCTIONOUTPUT )
			setInputFormatType(FileFormatTypes.BINARY);
	}
	
	
	/**
	 *  WRITE operation for Matrix
	 *  This constructor supports expression in parameters
	 */
	public DataOp(String l, DataType dt, ValueType vt, 
		DataOpTypes dop, Hop in, HashMap<String, Hop> inputParameters) {
		super(l, dt, vt);

		_dataop = dop;
		
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

		if (dop == DataOpTypes.TRANSIENTWRITE)
			setInputFormatType(FileFormatTypes.BINARY);
	}
	
	public DataOpTypes getDataOpType()
	{
		return _dataop;
	}
	
	public void setDataOpType( DataOpTypes type )
	{
		_dataop = type;
	}
	
	public void setOutputParams(long dim1, long dim2, long nnz, long rowsPerBlock, long colsPerBlock) {
		setDim1(dim1);
		setDim2(dim2);
		setNnz(nnz);
		setRowsInBlock(rowsPerBlock);
		setColsInBlock(colsPerBlock);
	}

	public void setFileName(String fn) {
		_fileName = fn;
	}

	public String getFileName() {
		return _fileName;
	}

	public int getParameterIndex(String name)
	{
		return _paramIndexMap.get(name);
	}
	
	@Override
	public Lop constructLops()
			throws HopsException, LopsException 
	{	
		//return already created lops
		if( getLops() != null )
			return getLops();

		ExecType et = optFindExecType();
		Lop l = null;
		
		// construct lops for all input parameters
		HashMap<String, Lop> inputLops = new HashMap<String, Lop>();
		for (Entry<String, Integer> cur : _paramIndexMap.entrySet()) {
			inputLops.put(cur.getKey(), getInput().get(cur.getValue())
					.constructLops());
		}

		// Create the lop
		switch(_dataop) 
		{
			case TRANSIENTREAD:
				l = new Data(HopsData2Lops.get(_dataop), null, inputLops, getName(), null, 
						getDataType(), getValueType(), true, getInputFormatType());
				setOutputDimensions(l);
				break;
				
			case PERSISTENTREAD:
				l = new Data(HopsData2Lops.get(_dataop), null, inputLops, getName(), null, 
						getDataType(), getValueType(), false, getInputFormatType());
				l.getOutputParameters().setDimensions(getDim1(), getDim2(), _inRowsInBlock, _inColsInBlock, getNnz());
				break;
				
			case PERSISTENTWRITE:
				l = new Data(HopsData2Lops.get(_dataop), getInput().get(0).constructLops(), inputLops, getName(), null, 
						getDataType(), getValueType(), false, getInputFormatType());
				((Data)l).setExecType(et);
				setOutputDimensions(l);
				break;
				
			case TRANSIENTWRITE:
				l = new Data(HopsData2Lops.get(_dataop), getInput().get(0).constructLops(), inputLops, getName(), null,
						getDataType(), getValueType(), true, getInputFormatType());
				setOutputDimensions(l);
				break;
				
			case FUNCTIONOUTPUT:
				l = new Data(HopsData2Lops.get(_dataop), getInput().get(0).constructLops(), inputLops, getName(), null, 
						getDataType(), getValueType(), true, getInputFormatType());
				((Data)l).setExecType(et);
				setOutputDimensions(l);
				break;
			
			default:
				throw new LopsException("Invalid operation type for Data LOP: " + _dataop);	
		}
		
		setLineNumbers(l);
		setLops(l);
		
		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();
	
		return getLops();

	}

	public void setInputFormatType(FileFormatTypes ft) {
		_inFormat = ft;
	}

	public FileFormatTypes getInputFormatType() {
		return _inFormat;
	}
	
	public void setInputBlockSizes( long brlen, long bclen ){
		setInputRowsInBlock(brlen);
		setInputColsInBlock(bclen);
	}
	
	public void setInputRowsInBlock( long brlen ){
		_inRowsInBlock = brlen;
	}
	
	public long getInputRowsInBlock(){
		return _inRowsInBlock;
	}
	
	public void setInputColsInBlock( long bclen ){
		_inColsInBlock = bclen;
	}
	
	public long getInputColsInBlock(){
		return _inColsInBlock;
	}
	
	public boolean isRead()
	{
		return( _dataop == DataOpTypes.PERSISTENTREAD || _dataop == DataOpTypes.TRANSIENTREAD );
	}
	
	public boolean isWrite()
	{
		return( _dataop == DataOpTypes.PERSISTENTWRITE || _dataop == DataOpTypes.TRANSIENTWRITE );
	}
	
	public boolean isPersistentReadWrite()
	{
		return( _dataop == DataOpTypes.PERSISTENTREAD || _dataop == DataOpTypes.PERSISTENTWRITE );
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += HopsData2String.get(_dataop);
		s += " "+getName();
		return s;
	}

	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (getVisited() != VisitStatus.DONE) {
				super.printMe();
				LOG.debug("  DataOp: " + _dataop);
				if (_fileName != null) {
					LOG.debug(" file: " + _fileName);
				}
				LOG.debug(" format: " + getInputFormatType());
				for (Hop h : getInput()) {
					h.printMe();
				}
			}
			setVisited(VisitStatus.DONE);
		}
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
		
		if ( getDataType() == DataType.SCALAR ) 
		{
			switch( getValueType() ) 
			{
				case INT:
					ret = OptimizerUtils.INT_SIZE; break;
				case DOUBLE:
					ret = OptimizerUtils.DOUBLE_SIZE; break;
				case BOOLEAN:
					ret = OptimizerUtils.BOOLEAN_SIZE; break;
				case STRING: 
					// by default, it estimates the size of string[100]
					ret = 100 * OptimizerUtils.CHAR_SIZE; break;
				case OBJECT:
					ret = OptimizerUtils.DEFAULT_SIZE; break;
				default:
					ret = 0;
			}
		}
		else //MATRIX 
		{
			if(   _dataop == DataOpTypes.PERSISTENTREAD 
			   || _dataop == DataOpTypes.TRANSIENTREAD ) 
			{
				double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
				ret = OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);	
			}
			// output memory estimate is not required for "write" nodes (just input)
		}
		
		return ret;
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		return LocalFileUtils.BUFFER_SIZE;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		long[] ret = null;
		
		if(   _dataop == DataOpTypes.PERSISTENTWRITE
			|| _dataop == DataOpTypes.TRANSIENTWRITE ) 
		{
			MatrixCharacteristics mc = memo.getAllInputStats(getInput().get(0));
			if( mc.dimsKnown() )
				ret = new long[]{ mc.getRows(), mc.getCols(), mc.getNonZeros() };
		}
		else if( _dataop == DataOpTypes.TRANSIENTREAD )
		{
			//prepare statistics, passed from cross-dag transient writes
			MatrixCharacteristics mc = memo.getAllInputStats(this);
			if( mc.dimsKnown() )
				ret = new long[]{ mc.getRows(), mc.getCols(), mc.getNonZeros() };
		}
		
		return ret;
	}
	
	
	
	@Override
	protected ExecType optFindExecType() 
		throws HopsException 
	{
		//MB: find exec type has two meanings here: (1) for write it means the actual
		//exec type, while (2) for read it affects the recompilation decision as needed
		//for example for sum(X) where the memory consumption is solely determined by the DataOp
		
		ExecType letype = (OptimizerUtils.isMemoryBasedOptLevel()) ? findExecTypeByMemEstimate() : null;
		ExecType REMOTE = OptimizerUtils.isSparkExecutionMode() ? ExecType.SPARK : ExecType.MR;
		
		//NOTE: independent of etype executed in MR (piggybacked) if input to persistent write is MR
		if( _dataop == DataOpTypes.PERSISTENTWRITE || _dataop == DataOpTypes.TRANSIENTWRITE )
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
					_etype = REMOTE;
				}
			
				//check for valid CP dimensions and matrix size
				checkAndSetInvalidCPDimsAndSize();
			}
			
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==REMOTE ) {
				setRequiresRecompile();
			}
		}
	    else //READ
		{
	    	//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && letype==REMOTE 
				&& (_recompileRead || _requiresCheckpoint) ) 
			{
				setRequiresRecompile();
			}
			
			_etype = letype;
		}
		
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		if( _dataop == DataOpTypes.PERSISTENTWRITE || _dataop == DataOpTypes.TRANSIENTWRITE )
		{
			Hop input1 = getInput().get(0);
			setDim1(input1.getDim1());
			setDim2(input1.getDim2());
			setNnz(input1.getNnz());
		}
		else //READ
		{
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
		ret._dataop = _dataop;
		ret._fileName = _fileName;
		ret._inFormat = _inFormat;
		ret._inRowsInBlock = _inRowsInBlock;
		ret._inColsInBlock = _inColsInBlock;
		ret._recompileRead = _recompileRead;
		ret._paramIndexMap = (HashMap<String, Integer>) _paramIndexMap.clone();
		//note: no deep cp of params since read-only 
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		return false;
	}
}
