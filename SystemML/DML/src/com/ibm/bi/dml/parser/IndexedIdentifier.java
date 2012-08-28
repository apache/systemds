package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.utils.LanguageException;


public class IndexedIdentifier extends DataIdentifier {

	// stores the expressions containing the ranges for the 
	private Expression 	_rowLowerBound = null, _rowUpperBound = null, _colLowerBound = null, _colUpperBound = null;
	
	// stores whether row / col indices have same value (thus selecting either (1 X n) row-vector OR (n X 1) col-vector)
	private boolean _rowLowerEqualsUpper = false, _colLowerEqualsUpper = false;
	
	public IndexedIdentifier(String name, boolean passedRows, boolean passedCols){
		super(name);
		_rowLowerBound = null; 
   		_rowUpperBound = null; 
   		_colLowerBound = null; 
   		_colUpperBound = null;
   		_rowLowerEqualsUpper = passedRows;
   		_colLowerEqualsUpper = passedCols;
	}
		
	
	public IndexPair calculateIndexedDimensions(HashMap<String, ConstIdentifier> currConstVars) throws LanguageException {
		
		// stores the updated row / col dimension info
		long updatedRowDim = 1, updatedColDim = 1;
		
		boolean isConst_rowLowerBound = false;
		boolean isConst_rowUpperBound = false;
		boolean isConst_colLowerBound = false;
		boolean isConst_colUpperBound = false;
		
		///////////////////////////////////////////////////////////////////////
		// perform constant propagation for index boundaries
		//////////////////////////////////////////////////////////////////////
		if (_rowLowerBound != null && _rowLowerBound instanceof DataIdentifier && !(_rowLowerBound instanceof IndexedIdentifier)) {
			String identifierName = ((DataIdentifier)_rowLowerBound).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof IntIdentifier && ((IntIdentifier)constValue).getValue() >= 1))
					throw new LanguageException("ERROR:  IndexedIdentifier statement, can only assign indices a long value (>= 1) attempted to assign value: " + constValue.toString());
				else{
					_rowLowerBound = new IntIdentifier((IntIdentifier)constValue);
					isConst_rowLowerBound = true;
				}
			}	
		}
		
		if (_rowUpperBound != null && _rowUpperBound instanceof DataIdentifier && !(_rowUpperBound instanceof IndexedIdentifier)) {
			String identifierName = ((DataIdentifier)_rowUpperBound).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof IntIdentifier && ((IntIdentifier)constValue).getValue() >= 1))
					throw new LanguageException("ERROR:  IndexedIdentifier statement, can only assign indices a long value (>= 1) attempted to assign value: " + constValue.toString());
				else{
					_rowUpperBound = new IntIdentifier((IntIdentifier)constValue);
					isConst_rowUpperBound = true;
				}
			}	
		}
		
		if (_colLowerBound != null && _colLowerBound instanceof DataIdentifier && !(_colLowerBound instanceof IndexedIdentifier)) {
			String identifierName = ((DataIdentifier)_colLowerBound).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof IntIdentifier && ((IntIdentifier)constValue).getValue() >= 1))
					throw new LanguageException("ERROR:  IndexedIdentifier statement, can only assign indices a long value (>= 1) attempted to assign value: " + constValue.toString());
				else{
					_colLowerBound = new IntIdentifier((IntIdentifier)constValue);
					isConst_colLowerBound = true;
				}
			}	
		}
		
		if (_colUpperBound != null && _colUpperBound instanceof DataIdentifier && !(_colUpperBound instanceof IndexedIdentifier)) {
			String identifierName = ((DataIdentifier)_colUpperBound).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof IntIdentifier && ((IntIdentifier)constValue).getValue() >= 1))
					throw new LanguageException("ERROR:  IndexedIdentifier statement, can only assign indices a long value (>= 1) attempted to assign value: " + constValue.toString());
				else{
					_colUpperBound = new IntIdentifier((IntIdentifier)constValue);
					isConst_colUpperBound = true;
				}
			}	
		}
		
		///////////////////////////////////////////////////////////////////////
		// update row dimensions
		///////////////////////////////////////////////////////////////////////
			
		// CASE:  lower == upper --> updated row dim = 1
		if (_rowLowerEqualsUpper) 
			updatedRowDim = 1;
		
		// CASE: (lower == null && upper == null) --> updated row dim = (rows of input)
		else if (_rowLowerBound == null && _rowUpperBound == null){
			updatedRowDim = this.getDim1(); 
		}
		// CASE: (lower == null) && (upper == constant) --> updated row dim = (constant)
		else if (_rowLowerBound == null && isConst_rowUpperBound) {
			updatedRowDim = ((IntIdentifier)_rowUpperBound).getValue();
		}
			
		// CASE: (lower == constant) && (upper == null) && (dimIndex > 0) --> rowCount - lower bound + 1
		else if (isConst_rowLowerBound && _rowUpperBound == null && this.getDim1() > 0) {
			long rowCount = this.getDim1();
			updatedRowDim = rowCount - ((IntIdentifier)_rowLowerBound).getValue() + 1;
		}
		
		// CASE: row dimension is unknown --> assign -1
		else{ 
			updatedRowDim = -1;
		}
		
		
		//////////////////////////////////////////////////////////////////////
		// update column dimensions
		///////////////////////////////////////////////////////////////////////
			
		// CASE:  lower == upper --> updated col dim = 1
		if (_colLowerEqualsUpper) 
			updatedColDim = 1;
		
		// CASE: (lower == null && upper == null) --> updated col dim = (cols of input)
		else if (_colLowerBound == null && _colUpperBound == null){
			updatedColDim = this.getDim2(); 
		}
		// CASE: (lower == null) && (upper == constant) --> updated col dim = (constant)
		else if (_colLowerBound == null && isConst_colUpperBound) {
			updatedColDim = ((IntIdentifier)_colUpperBound).getValue();
		}
			
		// CASE: (lower == constant) && (upper == null) && (dimIndex > 0) --> colCount - lower bound + 1
		else if (isConst_colLowerBound && _colUpperBound == null && this.getDim2() > 0) {
			long colCount = this.getDim2();
			updatedColDim = colCount - ((IntIdentifier)_colLowerBound).getValue() + 1;
		}
		
		// CASE: column dimension is unknown --> assign -1
		else{ 
			updatedColDim = -1;
		}
		
		return new IndexPair(updatedRowDim, updatedColDim);
		
	}
	
	
	public void updateIndexedDimensions(HashMap<String, ConstIdentifier> currConstVars) throws LanguageException{
	
		IndexPair updatedIndices = calculateIndexedDimensions(currConstVars);
		long updatedRowDim = updatedIndices._row;
		long updatedColDim = updatedIndices._col;
		this.setDimensions(updatedRowDim, updatedColDim);
	}
	
	
	public Expression rewriteExpression(String prefix) throws LanguageException {
		
		IndexedIdentifier newIndexedIdentifier = new IndexedIdentifier(this.getName(), this._rowLowerEqualsUpper, this._colLowerEqualsUpper);
		
		// set dimensionality information and other Identifier specific properties for new IndexedIdentifier
		newIndexedIdentifier.setProperties(this);
		
		// set remaining properties (specific to DataIdentifier)
		newIndexedIdentifier._kind = Kind.Data;
		newIndexedIdentifier._name = prefix + this._name;
		newIndexedIdentifier._valueTypeString = this.getValueType().toString();	
		newIndexedIdentifier._defaultValue = this._defaultValue;
	
		// creates rewritten expression (deep copy)
		newIndexedIdentifier._rowLowerBound = (_rowLowerBound == null) ? null : _rowLowerBound.rewriteExpression(prefix);
		newIndexedIdentifier._rowUpperBound = (_rowUpperBound == null) ? null : _rowUpperBound.rewriteExpression(prefix);
		newIndexedIdentifier._colLowerBound = (_colLowerBound == null) ? null : _colLowerBound.rewriteExpression(prefix);
		newIndexedIdentifier._colUpperBound = (_colUpperBound == null) ? null : _colUpperBound.rewriteExpression(prefix);
		
		return newIndexedIdentifier;
	}
		
	public void setIndices(ArrayList<ArrayList<Expression>> passed) throws ParseException {
		if (passed.size() != 2)
			throw new ParseException("[E] matrix indices must be specified for 2 dimensions -- currently specified indices for " + passed.size() + " dimensions ");
		
		ArrayList<Expression> rowIndices = passed.get(0);
		ArrayList<Expression> colIndices = passed.get(1);
	
		// case: both upper and lower are defined
		if (rowIndices.size() == 2){			
			_rowLowerBound = rowIndices.get(0);
			_rowUpperBound = rowIndices.get(1);
		}
		// case: only one index is defined --> thus lower = upper
		else if (rowIndices.size() == 1){
			_rowLowerBound = rowIndices.get(0);
			_rowUpperBound = rowIndices.get(0);
			_rowLowerEqualsUpper = true;
		}
		else {
			throw new ParseException("[E]  row indices are length " + rowIndices.size() + " -- should be either 1 or 2");
		}
		
		// case: both upper and lower are defined
		if (colIndices.size() == 2){			
			_colLowerBound = colIndices.get(0);
			_colUpperBound = colIndices.get(1);
		}
		// case: only one index is defined --> thus lower = upper
		else if (colIndices.size() == 1){
			_colLowerBound = colIndices.get(0);
			_colUpperBound = colIndices.get(0);
			_colLowerEqualsUpper = true;
		}
		else {
			throw new ParseException("[E] col indices are length " + + colIndices.size() + " -- should be either 1 or 2");
		}
		//System.out.println(this);
	}
	
	public Expression getRowLowerBound(){ return this._rowLowerBound; }
	public Expression getRowUpperBound(){ return this._rowUpperBound; }
	public Expression getColLowerBound(){ return this._colLowerBound; }
	public Expression getColUpperBound(){ return this._colUpperBound; }
	
	public void setRowLowerBound(Expression passed){ this._rowLowerBound = passed; }
	public void setRowUpperBound(Expression passed){ this._rowUpperBound = passed; }
	public void setColLowerBound(Expression passed){ this._colLowerBound = passed; }
	public void setColUpperBound(Expression passed){ this._colUpperBound = passed; }
	
		
	public String toString() {
		String retVal = new String();
		retVal += this.getName();
		if (_rowLowerBound != null || _rowUpperBound != null || _colLowerBound != null || _colUpperBound != null){
				retVal += "[";
				
				if (_rowLowerBound != null && _rowUpperBound != null){
					if (_rowLowerBound.toString().equals(_rowUpperBound.toString()))
						retVal += _rowLowerBound.toString();
					else 
						retVal += _rowLowerBound.toString() + ":" + _rowUpperBound.toString();
				}
				else {
					if (_rowLowerBound != null || _rowUpperBound != null){
						if (_rowLowerBound != null)
							retVal += _rowLowerBound.toString();
						
						retVal += ":";
						
						if (_rowUpperBound != null)
							retVal += _rowUpperBound.toString();
					}
				}
					
				retVal += ",";
				
				if (_colLowerBound != null && _colUpperBound != null){
					if (_colLowerBound.toString().equals(_colUpperBound.toString()))
						retVal += _colLowerBound.toString();
					else
						retVal += _colLowerBound.toString() + ":" + _colUpperBound.toString();
				}
				else {
					if (_colLowerBound != null || _colUpperBound != null) {
						
						if (_colLowerBound != null)
							retVal += _colLowerBound.toString();
						
						retVal += ":";
						
						if (_colUpperBound != null)
							retVal += _colUpperBound.toString();
					}
				}
				
				
				retVal += "]";
		}
		return retVal;
	}

	@Override
	// handles case when IndexedIdentifier is on RHS for assignment 
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		
		// add variable being indexed to read set
		result.addVariable(this.getName(), this);
		
		// add variables for indexing expressions
		if (_rowLowerBound != null)
			result.addVariables(_rowLowerBound.variablesRead());
		if (_rowUpperBound != null)
			result.addVariables(_rowUpperBound.variablesRead());
		if (_colLowerBound != null)
			result.addVariables(_colLowerBound.variablesRead());
		if (_colUpperBound != null)
			result.addVariables(_colUpperBound.variablesRead());
		
		return result;
	}
	
} // end class
	
class IndexPair {
	
	public long _row, _col;
	
	public IndexPair (long row, long col){
		_row = row;
		_col = col;
	}
} // end class
