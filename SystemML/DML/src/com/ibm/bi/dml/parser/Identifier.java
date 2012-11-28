package com.ibm.bi.dml.parser;

import java.util.HashMap;
import com.ibm.bi.dml.utils.LanguageException;


public abstract class Identifier extends Expression{

	protected DataType _dataType;
	protected ValueType _valueType;
	protected long _dim1;
	protected long _dim2;
	protected long _rows_in_block;
	protected long _columns_in_block;
	protected long _nnz;
	protected FormatType _formatType;
		
	public Identifier(Identifier i){
		_dataType = i.getDataType();
		_valueType = i.getValueType();
		_dim1 = i.getDim1();
		_dim2 = i.getDim2();
		_rows_in_block = i.getRowsInBlock();
		_columns_in_block = i.getColumnsInBlock();
		_nnz = i.getNnz();
		_formatType = i.getFormatType();
		
		// copy position information
		_beginLine 	 = i.getBeginLine();
		_beginColumn = i.getBeginColumn();
		_endLine 	 = i.getEndLine();
		_endColumn 	 = i.getEndColumn();
	}
	
	public Identifier(){
		_dim1 = -1;
		_dim2 = -1;
		_dataType = DataType.UNKNOWN;
		_valueType = ValueType.UNKNOWN;
		_rows_in_block = -1;
		_columns_in_block = -1;
		_nnz = -1;
		_output = this;
		_formatType = null;
	}
	
	public void setProperties(Identifier i){
		_dataType = i.getDataType();
		_valueType = i.getValueType();
		if (i instanceof IndexedIdentifier){
			_dim1 = ((IndexedIdentifier)i).getOrigDim1();
			_dim2 = ((IndexedIdentifier)i).getOrigDim2();
		}
		else {
			_dim1 = i.getDim1();
			_dim2 = i.getDim2();
		}
		_rows_in_block = i.getRowsInBlock();
		_columns_in_block = i.getColumnsInBlock();
		_nnz = i.getNnz();
		_formatType = i.getFormatType();
				
	}
	
	public void setDimensionValueProperties(Identifier i){
			_dim1 = i.getDim1();
			_dim2 = i.getDim2();
			_nnz = i.getNnz();
			_dataType = i.getDataType();
			_valueType = i.getValueType();
	}
	
	public void setDataType(DataType dt){
		_dataType = dt;
	}
	
	public void setValueType(ValueType vt){
		_valueType = vt;
	}
	
	public void setFormatType(FormatType ft){
		_formatType = ft;
	}
	
	public void setDimensions(long dim1, long dim2){
		_dim1 = dim1;
		_dim2 = dim2;
	}
	
	public void setDimensions(InputStatement is){
		
		// check if 
		if (is.getExprParam("rows") != null && is.getExprParam("rows") instanceof IntIdentifier) 
			_dim1 = new Long(((IntIdentifier)is.getExprParam("rows")).getValue());
		if (is.getExprParam("cols") != null && is.getExprParam("cols") instanceof IntIdentifier) 
			_dim2 = new Long(((IntIdentifier)is.getExprParam("cols")).getValue());
	}
	
	public void setBlockDimensions(long dim1, long dim2){
		 _rows_in_block = dim1;
		 _columns_in_block = dim2;
	}
	
	public void setNnz(long nnzs){
		_nnz = nnzs;
	}
	
	public long getDim1(){
		return _dim1;
	}
	
	public long getDim2(){
		return _dim2;
	}
	
	public DataType getDataType(){
		return _dataType;
	}
	
	public ValueType getValueType(){
		return _valueType;
	}
	
	public FormatType getFormatType(){
		return _formatType;
	}
	
	public long getRowsInBlock(){
		return _rows_in_block;
	}
	
	public long getColumnsInBlock(){
		return _columns_in_block;
	}
	
	public long getNnz(){
		return _nnz;
	}
	
	public void validateExpression(HashMap<String,DataIdentifier> ids, HashMap<String,ConstIdentifier> constVars) throws LanguageException {
		//Identifier out = this.getOutput();
		
		if (this.getOutput() instanceof DataIdentifier){
			
			// set properties for Data identifer
			String name = ((DataIdentifier)this.getOutput()).getName();
			Identifier id = ids.get(name);
			if ( id == null ){
				//LiveVariableAnalysis.throwUndefinedVar(name, null);
				
				LOG.error(this.printErrorLocation() + "Undefined Variable (" + name + ") used in statement");
				
				throw new LanguageException(this.printErrorLocation() + "Undefined Variable (" + name + ") used in statement",
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			this.getOutput().setProperties(id);
			
			// validate IndexedIdentifier -- which is substype of DataIdentifer with index
			if (this.getOutput() instanceof IndexedIdentifier){
				
				// validate the row / col index bounds (if defined)
				IndexedIdentifier indexedIdentiferOut = (IndexedIdentifier)this.getOutput();
				if (indexedIdentiferOut.getRowLowerBound() != null) 
					indexedIdentiferOut.getRowLowerBound().validateExpression(ids, constVars);
				if (indexedIdentiferOut.getRowUpperBound() != null) 
					indexedIdentiferOut.getRowUpperBound().validateExpression(ids, constVars);
				if (indexedIdentiferOut.getColLowerBound() != null) 
					indexedIdentiferOut.getColLowerBound().validateExpression(ids,constVars);	
				if (indexedIdentiferOut.getColUpperBound() != null) 
					indexedIdentiferOut.getColUpperBound().validateExpression(ids, constVars);
				
				IndexPair updatedIndices = ((IndexedIdentifier)this.getOutput()).calculateIndexedDimensions(constVars);
				((IndexedIdentifier)this.getOutput()).setDimensions(updatedIndices._row, updatedIndices._col);
				
				
				
			}
							
		} else {
			this.getOutput().setProperties(this.getOutput());
		}
	}
	
	public void computeDataType() {
				
		if ((_dim1 == 0) && (_dim2 == 0)) {
			_dataType = DataType.SCALAR;
		} else if ((_dim1 >= 1) || (_dim2 >= 1)){
			// Vector also set as matrix
			// Data type is set as matrix, if either of dimensions is -1
			_dataType = DataType.MATRIX;
		} else _dataType = DataType.UNKNOWN;	 
		
	}
	
	public void setBooleanProperties(){
		_dataType = DataType.SCALAR;
		_valueType = ValueType.BOOLEAN;
		_dim1 = 0;
		_dim2 = 0;
		_rows_in_block = 0;
		_columns_in_block = 0;
		_nnz = -1;
		_formatType = null;
	}
	
	public void setIntProperties(){
		_dataType = DataType.SCALAR;
		_valueType = ValueType.INT;
		_dim1 = 0;
		_dim2 = 0;
		_rows_in_block = 0;
		_columns_in_block = 0;
		_nnz = -1;
		_formatType = null;
	}
	
	
	public boolean isScalarBoolean(){
		return (_valueType == ValueType.BOOLEAN) && (_dataType == DataType.SCALAR);
	}
}
