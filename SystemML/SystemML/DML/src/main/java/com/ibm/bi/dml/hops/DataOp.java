/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
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
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLSelectStatement;
import com.ibm.bi.dml.sql.sqllops.SQLTableReference;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;

import java.util.HashMap;
import java.util.Map.Entry;


public class DataOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private DataOpTypes _dataop;
	private String _fileName = null;
	private FileFormatTypes _formatType = FileFormatTypes.TEXT;
	
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
	
	// READ operation for Matrix w/ dim1, dim2. 
	// This constructor does not support any expression in parameters
	public DataOp(String l, DataType dt, ValueType vt, DataOpTypes dop,
			String fname, long dim1, long dim2, long nnz, long rowsPerBlock, long colsPerBlock) {
		super(Kind.DataOp, l, dt, vt);
		_dataop = dop;
		
		_fileName = fname;
		setDim1(dim1);
		setDim2(dim2);
		setNnz(nnz);
		setRowsInBlock(rowsPerBlock);
		setColsInBlock(colsPerBlock);
		
		if (dop == DataOpTypes.TRANSIENTREAD)
			setFormatType(FileFormatTypes.BINARY);
	}

	// WRITE operation
	// This constructor does not support any expression in parameters
	public DataOp(String l, DataType dt, ValueType vt, Hop in,
			DataOpTypes dop, String fname) {
		super(Kind.DataOp, l, dt, vt);
		_dataop = dop;
		getInput().add(0, in);
		in.getParent().add(this);
		_fileName = fname;

		if (dop == DataOpTypes.TRANSIENTWRITE || dop == DataOpTypes.FUNCTIONOUTPUT )
			setFormatType(FileFormatTypes.BINARY);
	}
	
	/**
	 * READ operation for Matrix
	 * This constructor supports expressions in parameters
	 */
	public DataOp(String l, DataType dt, ValueType vt, 
			DataOpTypes dop, HashMap<String, Hop> inputParameters) {
		super(Kind.DataOp, l, dt, vt);

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
			setFormatType(FileFormatTypes.BINARY);
		}
	}
	
	/**
	 *  WRITE operation for Matrix
	 *  This constructor supports expression in parameters
	 */
	public DataOp(String l, DataType dt, ValueType vt, 
		DataOpTypes dop, Hop in, HashMap<String, Hop> inputParameters) {
		super(Kind.DataOp, l, dt, vt);

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
			setFormatType(FileFormatTypes.BINARY);
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
		if (getLops() == null) {
			Data l = null;

			ExecType et = optFindExecType();
			
			// construct lops for all input parameters
			HashMap<String, Lop> inputLops = new HashMap<String, Lop>();
			for (Entry<String, Integer> cur : _paramIndexMap.entrySet()) {
				inputLops.put(cur.getKey(), getInput().get(cur.getValue())
						.constructLops());
			}

			// Set the transient flag
			boolean isTransient = false;
			switch(_dataop) 
			{
				case PERSISTENTREAD:
				case PERSISTENTWRITE:
					isTransient = false;
					break;
					
				case TRANSIENTREAD:
				case TRANSIENTWRITE:
					isTransient = true;
					break;
					
				case FUNCTIONOUTPUT:
					/* TODO: currently, function outputs are treated as transient.
					 * This needs to be revisited whenever function calls are fully integrated into Hop DAGs.
					 */
					isTransient = true;
					break;
					
				default:
					throw new LopsException("Invalid operation type for Data LOP: " + _dataop);	
			}
			
			// Cretae the lop
			switch(_dataop) 
			{
				case TRANSIENTREAD:
				case PERSISTENTREAD:
					l = new Data(HopsData2Lops.get(_dataop), null, inputLops, getName(), null, getDataType(), getValueType(), isTransient, getFormatType());
					break;
					
				case PERSISTENTWRITE:
				case TRANSIENTWRITE:
				case FUNCTIONOUTPUT:
					l = new Data(HopsData2Lops.get(_dataop), this.getInput().get(0).constructLops(), inputLops, getName(), null, getDataType(), getValueType(), isTransient, getFormatType());
					
					// TODO: should we set the exec type for transient write ?
					if (_dataop == DataOpTypes.PERSISTENTWRITE || _dataop == DataOpTypes.FUNCTIONOUTPUT)
						l.setExecType(et);
					break;
			
				default:
					throw new LopsException("Invalid operation type for Data LOP: " + _dataop);	
			}
			
			//set remaining meta data
			l.setFileFormatType(this.getFormatType());
			l.getOutputParameters().setDimensions(getDim1(), getDim2(),getRowsInBlock(), getColsInBlock(), getNnz());
			l.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			
			setLops(l);
		}
	
		return getLops();

	}

	public void setFormatType(FileFormatTypes ft) {
		_formatType = ft;
	}

	public FileFormatTypes getFormatType() {
		return _formatType;
	}
	
	public boolean isRead()
	{
		return( _dataop == DataOpTypes.PERSISTENTREAD || _dataop == DataOpTypes.TRANSIENTREAD );
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
				LOG.debug(" format: " + getFormatType());
				for (Hop h : getInput()) {
					h.printMe();
				}
			}
			setVisited(VisitStatus.DONE);
		}
	}

	public DataOpTypes get_dataop() {
		return _dataop;
	}

	private SQLLopProperties getProperties(String input)
	{
		SQLLopProperties prop = new SQLLopProperties();
		prop.setJoinType(JOINTYPE.NONE);
		prop.setAggType(AGGREGATIONTYPE.NONE);
		
		String op = "PRead ";
		if(this._dataop == DataOpTypes.PERSISTENTWRITE)
			op = "PWrite ";
		else if(this._dataop == DataOpTypes.TRANSIENTREAD)
			op = "TRead ";
		else if(this._dataop == DataOpTypes.TRANSIENTWRITE)
			op = "TWrite ";

		prop.setOpString(op + input);

		return prop;
	}
	
	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		if(this.getSqlLops() == null)
		{
			SQLLops sqllop;
			
			Hop input = null;
			if( !getInput().isEmpty() )
				input = this.getInput().get(0);
			
			//Should not have any inputs
			if(this._dataop == DataOpTypes.PERSISTENTREAD)
			{
				sqllop = new SQLLops(this.getName(),
									GENERATES.SQL,
									this.getValueType(),
									this.getDataType());
				
				if(this.getDataType() == DataType.SCALAR)
					sqllop.set_sql(String.format(SQLLops.SELECTSCALAR, SQLLops.addQuotes(this.getFileName())));
				else
				{
					//sqllop.set_flag(GENERATES.DML);
					sqllop.set_tableName(this.getFileName());
					sqllop.set_flag(GENERATES.NONE);
					sqllop.set_sql(String.format(SQLLops.SELECTSTAR, this.getFileName()));
				}
			}
			else if(this._dataop == DataOpTypes.TRANSIENTREAD)
			{
				//Here we do not have a file name, so the name is taken
				sqllop = new SQLLops(this.getName(),
									GENERATES.NONE,
									this.getValueType(),
									this.getDataType());
				
				String name = this.getName();
				
				if(this.getDataType() == DataType.MATRIX)
					name += "_transmatrix";
				else 
					name = "##" + name + "_transscalar##";//TODO: put ## here for placeholders
				
				sqllop.set_tableName(name);
				sqllop.set_sql(name);
			}
			else if(this._dataop == DataOpTypes.PERSISTENTWRITE && this.getDataType() == DataType.SCALAR)
			{
				sqllop = new SQLLops(this.getFileName(),
						GENERATES.DML_PERSISTENT,
						input.constructSQLLOPs(),
						this.getValueType(),
						this.getDataType());
				sqllop.set_tableName(this.getFileName());
				//sqllop.setDataType(DataType.MATRIX);
				
				if(input.getSqlLops().get_flag() == GENERATES.NONE)
					sqllop.set_sql("SELECT " + input.getSqlLops().get_tableName());
				else
					sqllop.set_sql(input.getSqlLops().get_sql());
			}
			else
			{
				if(this.getInput().size() < 1)
					throw new HopsException(this.printErrorLocation() + "In DataOp Hop, A write needs at least one input \n");

				String name = this.getFileName();
				
				//With scalars or transient writes there is no filename
				if(this.getDataType() == DataType.SCALAR)
					name = "##" + this.getName() + "_transscalar##";
				else if(this._dataop == DataOpTypes.TRANSIENTWRITE)
					name = this.getName() + "_transmatrix";
				
				sqllop = new SQLLops(name,
				(this._dataop == DataOpTypes.TRANSIENTWRITE) ? GENERATES.DML_TRANSIENT : GENERATES.DML_PERSISTENT,
									input.constructSQLLOPs(),
									this.getValueType(),
									this.getDataType());
				sqllop.set_tableName(name);
				
				if(input.getSqlLops().getDataType() == DataType.MATRIX)
					sqllop.set_sql(String.format(SQLLops.SELECTSTAR, input.getSqlLops().get_tableName()));
				//Scalar with SQL
				else if(input.getSqlLops().get_flag() == GENERATES.SQL)
					sqllop.set_sql(String.format(SQLLops.SIMPLESCALARSELECTFROM, "sval", SQLLops.addQuotes(input.getSqlLops().get_tableName() )));
				//Other scalars such as variable names and literals
				else
					sqllop.set_sql(input.getSqlLops().get_tableName());
			}
			String i = _fileName;
			if(input != null && _fileName != null)
				i = (input.getSqlLops() != null) ? input.getSqlLops().get_tableName() + "->" + i : "->" + i;
				//i = input.getSqlLops().get_tableName() + "->" + i;
			else
				i = "";
			sqllop.set_properties(getProperties(i));
			this.setSqlLops(sqllop);
		}
		return this.getSqlLops();
	}
	
	@SuppressWarnings("unused")
	private SQLSelectStatement getSQLSelect(Hop input)
	{
		SQLSelectStatement stmt = new SQLSelectStatement();
		
		if(this.getDataType() == DataType.MATRIX)
		{
			if(this._dataop == DataOpTypes.PERSISTENTREAD)
			{
				stmt.getColumns().add("*");
				stmt.setTable(new SQLTableReference(this.getFileName()));
			}
			else if(this._dataop == DataOpTypes.TRANSIENTREAD)
			{
				return null;
			}
			else if(this._dataop == DataOpTypes.PERSISTENTWRITE || this._dataop == DataOpTypes.TRANSIENTWRITE)
			{
				stmt.getColumns().add("*");
				stmt.setTable(new SQLTableReference(input.getSqlLops().get_tableName()));
			}
		}
		else if(this.getDataType() == DataType.SCALAR)
		{
			if(this._dataop == DataOpTypes.TRANSIENTREAD)
			{
				return null;
			}
			else if(this._dataop == DataOpTypes.PERSISTENTREAD || this._dataop == DataOpTypes.PERSISTENTWRITE || this._dataop == DataOpTypes.TRANSIENTWRITE)
			{
				stmt.getColumns().add("*");
				stmt.setTable(new SQLTableReference(input.getSqlLops().get_tableName()));
			}
		}
		return stmt;
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
			switch(this.getValueType()) 
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
				ret = new long[]{ mc.get_rows(), mc.get_cols(), mc.getNonZeros() };
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
					_etype = ExecType.MR;
				}
				
				//mark for recompile (forever)
				if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==ExecType.MR )
					setRequiresRecompile();
			}
		}
	    else //READ
		{
	    	//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && letype==ExecType.MR && _recompileRead )
				setRequiresRecompile();
	    	
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
		ret._formatType = _formatType;
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
