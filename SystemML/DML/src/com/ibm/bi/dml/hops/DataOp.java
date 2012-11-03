package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLSelectStatement;
import com.ibm.bi.dml.sql.sqllops.SQLTableReference;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LopsException;

import java.util.HashMap;
import java.util.Map.Entry;


public class DataOp extends Hops {

	DataOpTypes _dataop;
	//TODO: remove fileName
	String _fileName;
	private FileFormatTypes _formatType = FileFormatTypes.TEXT;
	
	
	/**
	 * List of "named" input parameters. They are maintained as a hashmap:
	 * parameter names (String) are mapped as indices (Integer) into getInput()
	 * arraylist.
	 * 
	 * i.e., getInput().get(_paramIndexMap.get(parameterName)) refers to the Hop
	 * that is associated with parameterName.
	 */
	private HashMap<String, Integer> _paramIndexMap = new HashMap<String, Integer>();

	// READ operation for Matrix w/ dim1, dim2. 
	// This constructor does not support any expression in parameters
	public DataOp(String l, DataType dt, ValueType vt, DataOpTypes dop,
			String fname, long dim1, long dim2, long nnz, long rowsPerBlock, long colsPerBlock) {
		super(Kind.DataOp, l, dt, vt);
		_dataop = dop;
		
		_fileName = fname;
		set_dim1(dim1);
		set_dim2(dim2);
		setNnz(nnz);
		set_rows_in_block(rowsPerBlock);
		set_cols_in_block(colsPerBlock);
				
		if (dop == DataOpTypes.TRANSIENTREAD)
			setFormatType(FileFormatTypes.BINARY);
	}

	// WRITE operation
	// This constructor does not support any expression in parameters
	public DataOp(String l, DataType dt, ValueType vt, Hops in,
			DataOpTypes dop, String fname) {
		super(Kind.DataOp, l, dt, vt);
		_dataop = dop;
		getInput().add(0, in);
		in.getParent().add(this);
		_fileName = fname;

		if (dop == DataOpTypes.TRANSIENTWRITE)
			setFormatType(FileFormatTypes.BINARY);
	}
	
	/**
	 * READ operation for Matrix
	 * This constructor supports expressions in parameters
	 */
	public DataOp(String l, DataType dt, ValueType vt, 
			DataOpTypes dop, HashMap<String, Hops> inputParameters) {
		super(Kind.DataOp, l, dt, vt);

		_dataop = dop;

		int index = 0;
		for (String s : inputParameters.keySet()) {
			Hops input = inputParameters.get(s);
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
		DataOpTypes dop, Hops in, HashMap<String, Hops> inputParameters) {
		super(Kind.DataOp, l, dt, vt);

		_dataop = dop;
		
		getInput().add(0, in);
		in.getParent().add(this);
		
		if (inputParameters != null){
			int index = 1;
			for (String s : inputParameters.keySet()) {
				Hops input = inputParameters.get(s);
				getInput().add(input);
				input.getParent().add(this);

				_paramIndexMap.put(s, index);
				index++;
			}
		
		}

		if (dop == DataOpTypes.TRANSIENTWRITE)
			setFormatType(FileFormatTypes.BINARY);
	}
	
	public void setOutputParams(long dim1, long dim2, long nnz, long rowsPerBlock, long colsPerBlock) {
		set_dim1(dim1);
		set_dim2(dim2);
		setNnz(nnz);
		set_rows_in_block(rowsPerBlock);
		set_cols_in_block(colsPerBlock);
	}

	public void setFileName(String fn) {
		_fileName = fn;
	}

	public String getFileName() {
		return _fileName;
	}

	@Override
	public Lops constructLops()
			throws HopsException, LopsException {
				
		if (get_lops() == null) {
			Lops l = null;

			//TODO: need to remove this if statement
			/*if (!(_fileName==null)){
				
				if (_dataop == DataOpTypes.PERSISTENTREAD) {
					l = new Data(
							getFileName(), HopsData2Lops.get(_dataop), get_name(), null,
							get_dataType(), get_valueType(), false);
				} else if (_dataop == DataOpTypes.TRANSIENTREAD) {
					l = new Data(
							getFileName(), HopsData2Lops.get(_dataop), get_name(),
							null, get_dataType(), get_valueType(), true);
				} else if (_dataop == DataOpTypes.PERSISTENTWRITE) {
					l = new Data(
							getFileName(), HopsData2Lops.get(_dataop), this
									.getInput().get(0).constructLops(), get_name(), null,
							get_dataType(), get_valueType(), false);
				} else if (_dataop == DataOpTypes.TRANSIENTWRITE) {
					l = new Data(
							getFileName(), HopsData2Lops.get(_dataop), this
									.getInput().get(0).constructLops(), get_name(),
							null, get_dataType(), get_valueType(), true);
				}
			}
			
			else {*/

				// construct lops for all input parameters
				HashMap<String, Lops> inputLops = new HashMap<String, Lops>();
				for (Entry<String, Integer> cur : _paramIndexMap.entrySet()) {
					inputLops.put(cur.getKey(), getInput().get(cur.getValue())
							.constructLops());
				}

				if (_dataop == DataOpTypes.PERSISTENTREAD) {
					l = new Data(
							HopsData2Lops.get(_dataop), null,
							inputLops, get_name(), null,
							get_dataType(), get_valueType(), false);
				} else if (_dataop == DataOpTypes.TRANSIENTREAD) {
					l = new Data(
							HopsData2Lops.get(_dataop), null,
							inputLops, get_name(), null, 
							get_dataType(), get_valueType(), true);
				} else if (_dataop == DataOpTypes.PERSISTENTWRITE) {
					l = new Data(
							HopsData2Lops.get(_dataop), this
							.getInput().get(0).constructLops(),inputLops, 
							get_name(), null, get_dataType(), get_valueType(), false);
				} else if (_dataop == DataOpTypes.TRANSIENTWRITE) {
					l = new Data(
							HopsData2Lops.get(_dataop), this
							.getInput().get(0).constructLops(),inputLops, 
							get_name(), null, get_dataType(), get_valueType(), true);
				}
			//}
			
			((Data) l).setFileFormatType(this.getFormatType());

			l.getOutputParameters().setDimensions(get_dim1(), get_dim2(),
					get_rows_in_block(), get_cols_in_block(), getNnz());
			
			l.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			
			set_lops(l);
		}

		return get_lops();

	}

	public void setFormatType(FileFormatTypes ft) {
		_formatType = ft;
	}

	public FileFormatTypes getFormatType() {
		return _formatType;
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += HopsData2String.get(_dataop);
		return s;
	}

	public void printMe() throws HopsException {
		if (get_visited() != VISIT_STATUS.DONE) {
			super.printMe();
			System.out.print("  DataOp: " + _dataop);
			if (_fileName != null) {
				System.out.print(" file: " + _fileName);
			}
			System.out.println(" format: " + getFormatType());
			System.out.print("\n");
			for (Hops h : getInput()) {
				h.printMe();
			}
			;
		}
		set_visited(VISIT_STATUS.DONE);
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
		if(this.get_sqllops() == null)
		{
			SQLLops sqllop;
			
			Hops input = null;
			if(this.getInput().size() > 0)
				input = this.getInput().get(0);
			
			//Should not have any inputs
			if(this._dataop == DataOpTypes.PERSISTENTREAD)
			{
				sqllop = new SQLLops(this.get_name(),
									GENERATES.SQL,
									this.get_valueType(),
									this.get_dataType());
				
				if(this.get_dataType() == DataType.SCALAR)
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
				sqllop = new SQLLops(this.get_name(),
									GENERATES.NONE,
									this.get_valueType(),
									this.get_dataType());
				
				String name = this.get_name();
				
				if(this.get_dataType() == DataType.MATRIX)
					name += "_transmatrix";
				else 
					name = "##" + name + "_transscalar##";//TODO: put ## here for placeholders
				
				sqllop.set_tableName(name);
				sqllop.set_sql(name);
			}
			else if(this._dataop == DataOpTypes.PERSISTENTWRITE && this.get_dataType() == DataType.SCALAR)
			{
				sqllop = new SQLLops(this.getFileName(),
						GENERATES.DML_PERSISTENT,
						input.constructSQLLOPs(),
						this.get_valueType(),
						this.get_dataType());
				sqllop.set_tableName(this.getFileName());
				//sqllop.set_dataType(DataType.MATRIX);
				
				if(input.get_sqllops().get_flag() == GENERATES.NONE)
					sqllop.set_sql("SELECT " + input.get_sqllops().get_tableName());
				else
					sqllop.set_sql(input.get_sqllops().get_sql());
			}
			else
			{
				if(this.getInput().size() < 1)
					throw new HopsException(this.printErrorLocation() + "In DataOp Hop, A write needs at least one input \n");

				String name = this.getFileName();
				
				//With scalars or transient writes there is no filename
				if(this.get_dataType() == DataType.SCALAR)
					name = "##" + this.get_name() + "_transscalar##";
				else if(this._dataop == DataOpTypes.TRANSIENTWRITE)
					name = this.get_name() + "_transmatrix";
				
				sqllop = new SQLLops(name,
				(this._dataop == DataOpTypes.TRANSIENTWRITE) ? GENERATES.DML_TRANSIENT : GENERATES.DML_PERSISTENT,
									input.constructSQLLOPs(),
									this.get_valueType(),
									this.get_dataType());
				sqllop.set_tableName(name);
				
				if(input.get_sqllops().get_dataType() == DataType.MATRIX)
					sqllop.set_sql(String.format(SQLLops.SELECTSTAR, input.get_sqllops().get_tableName()));
				//Scalar with SQL
				else if(input.get_sqllops().get_flag() == GENERATES.SQL)
					sqllop.set_sql(String.format(SQLLops.SIMPLESCALARSELECTFROM, "sval", SQLLops.addQuotes(input.get_sqllops().get_tableName() )));
				//Other scalars such as variable names and literals
				else
					sqllop.set_sql(input.get_sqllops().get_tableName());
			}
			String i = _fileName;
			if(input != null && _fileName != null)
				i = input.get_sqllops().get_tableName() + "->" + i;
			else
				i = "";
			sqllop.set_properties(getProperties(i));
			this.set_sqllops(sqllop);
		}
		return this.get_sqllops();
	}
	
	private SQLSelectStatement getSQLSelect(Hops input)
	{
		SQLSelectStatement stmt = new SQLSelectStatement();
		
		if(this.get_dataType() == DataType.MATRIX)
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
				stmt.setTable(new SQLTableReference(input.get_sqllops().get_tableName()));
			}
		}
		else if(this.get_dataType() == DataType.SCALAR)
		{
			if(this._dataop == DataOpTypes.TRANSIENTREAD)
			{
				return null;
			}
			else if(this._dataop == DataOpTypes.PERSISTENTREAD || this._dataop == DataOpTypes.PERSISTENTWRITE || this._dataop == DataOpTypes.TRANSIENTWRITE)
			{
				stmt.getColumns().add("*");
				stmt.setTable(new SQLTableReference(input.get_sqllops().get_tableName()));
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
	public double computeMemEstimate() {
		
		if ( get_dataType() == DataType.SCALAR ) {
			switch(this.get_valueType()) {
			case INT:
				_outputMemEstimate = OptimizerUtils.INT_SIZE; break;
			case DOUBLE:
				_outputMemEstimate = OptimizerUtils.DOUBLE_SIZE; break;
			case BOOLEAN:
				_outputMemEstimate = OptimizerUtils.BOOLEAN_SIZE; break;
			case STRING: 
				// by default, it estimates the size of string[100]
				_outputMemEstimate = 100 * OptimizerUtils.CHAR_SIZE; break;
			case OBJECT:
				_outputMemEstimate = OptimizerUtils.DEFAULT_SIZE; break;
			}
		}
		else {
			if (_dataop == DataOpTypes.PERSISTENTREAD || _dataop == DataOpTypes.TRANSIENTREAD ) {
				if ( getNnz() > 0 ) {
					_outputMemEstimate = OptimizerUtils.estimate(_dim1, _dim2, (double)_nnz/(_dim1*_dim2));
				}
				else {
					_outputMemEstimate = OptimizerUtils.estimateSize(_dim1, _dim2, OptimizerUtils.DEF_SPARSITY);
				}
			}
			else {
				// memory estimate is not required for "write" nodes
				// as a placeholder, we simply use input's estimate
				_outputMemEstimate = 0;
			}
		}
		_memEstimate = getInputOutputSize();
		
		return _memEstimate;
	}
	
	@Override
	protected ExecType optFindExecType() throws HopsException {
		// Since a DATA hop does not represent any computation, 
		// this function is not applicable. 
		return null;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		//do nothing; dimensions updated via set output params
	}
}
