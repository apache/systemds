package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.utils.LanguageException;


public class IndexedIdentifier extends DataIdentifier {

	// stores the expressions containing the ranges for the 
	private Expression 	_rowLowerBound = null, _rowUpperBound = null, _colLowerBound = null, _colUpperBound = null;
	
	// stores whether row / col indices have same value (thus selecting either (1 X n) row-vector OR (n X 1) col-vector)
	private boolean _rowLowerEqualsUpper = false, _colLowerEqualsUpper = false;
	
	// for IndexedIdentifier, dim1 and dim2 will ultimately be the dimensions of the indexed region and NOT the dims of what is being indexed
	// E.g., for A[1:10,1:10], where A = Rand (rows = 20, cols = 20), dim1 = 10, dim2 = 10, origDim1 = 20, origDim2 = 20
	
	// stores the dimensions of Identifier prior to indexing 
	private long _origDim1, _origDim2;
	
	public IndexedIdentifier(String name, boolean passedRows, boolean passedCols){
		super(name);
		_rowLowerBound = null; 
   		_rowUpperBound = null; 
   		_colLowerBound = null; 
   		_colUpperBound = null;
   		
   		_rowLowerEqualsUpper = passedRows;
   		_colLowerEqualsUpper = passedCols;
   		
   		_origDim1 = -1L;
   		_origDim2 = -1L;
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
		///////////////////////////////////////////////////////////////////////
		
		// process row lower bound
		if (_rowLowerBound instanceof ConstIdentifier && ( _rowLowerBound instanceof IntIdentifier || _rowLowerBound instanceof DoubleIdentifier )){
			isConst_rowLowerBound = true;
		}
		else if (_rowLowerBound instanceof ConstIdentifier) {
			throw new LanguageException(this.printErrorLocation() + "attempted to assign lower-bound row index for Indexed Identifier " + this.toString() + " the non-numeric value " + _rowLowerBound.toString());
		}
		
		// perform constant propogation
		else if (_rowLowerBound != null && _rowLowerBound instanceof DataIdentifier && !(_rowLowerBound instanceof IndexedIdentifier)) {
			String identifierName = ((DataIdentifier)_rowLowerBound).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				
				if (!(constValue instanceof IntIdentifier || constValue instanceof DoubleIdentifier ))
					throw new LanguageException(this.printErrorLocation() + "attempted to assign indices for Indexed Identifier " + this.toString() + "the non-numeric value " + constValue.getOutput().toString());
	
				else{
					if (constValue instanceof IntIdentifier){
						_rowLowerBound = new IntIdentifier((IntIdentifier)constValue);
						_rowLowerBound.setAllPositions(constValue.getBeginLine(), constValue.getBeginColumn(), constValue.getEndLine(), constValue.getEndColumn());
					}
					else{
						_rowLowerBound = new DoubleIdentifier((DoubleIdentifier)constValue);
						_rowLowerBound.setAllPositions(constValue.getBeginLine(), constValue.getBeginColumn(), constValue.getEndLine(), constValue.getEndColumn());
					}
					isConst_rowLowerBound = true;
				}
			}	
		}
		
		// check 1 < indexed row lower-bound < rows in IndexedIdentifier 
		// (assuming row dims available for upper bound)
		Long rowLB_1 = -1L;
		if (isConst_rowLowerBound) {
				
			if (_rowLowerBound instanceof IntIdentifier) 
				rowLB_1 = ((IntIdentifier)_rowLowerBound).getValue();
			else
				rowLB_1 = Math.round(((DoubleIdentifier)_rowLowerBound).getValue());
			
			if (rowLB_1 < 1)
				throw new LanguageException(this.printErrorLocation() + "lower-bound row index " + rowLB_1 + " is out of bounds. Must be >= 1");
			
			if ((this.getDim1() > 0)  && (rowLB_1 > this.getDim1())) 
				throw new LanguageException(this.printErrorLocation() + "lower-bound row index " + rowLB_1 + " is out of bounds.  Rows in " + this.getName() + ": " + this.getDim1());
		}
		
		
		if (_rowUpperBound instanceof ConstIdentifier && ( _rowUpperBound instanceof IntIdentifier || _rowUpperBound instanceof DoubleIdentifier )){
			isConst_rowUpperBound = true;
		}	
		else if (_rowUpperBound instanceof ConstIdentifier){
			throw new LanguageException(this.printErrorLocation() + "attempted to assign upper-bound row index for Indexed Identifier " + this.toString() + "the non-numeric value " + _rowUpperBound.toString());
		}
		
		else if (_rowUpperBound != null && _rowUpperBound instanceof DataIdentifier && !(_rowUpperBound instanceof IndexedIdentifier)) {
			String identifierName = ((DataIdentifier)_rowUpperBound).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				
				if (!(constValue instanceof IntIdentifier || constValue instanceof DoubleIdentifier ))
					throw new LanguageException(this.printErrorLocation() + "attempted to assign indices for Indexed Identifier " + this.toString() + "the non-numeric value " + constValue.getOutput().toString());
	
				else{
					if (constValue instanceof IntIdentifier){
						_rowUpperBound = new IntIdentifier((IntIdentifier)constValue);
						_rowUpperBound.setAllPositions(constValue.getBeginLine(), constValue.getBeginColumn(), constValue.getEndLine(), constValue.getEndColumn());
					}
					else {
						_rowUpperBound = new DoubleIdentifier((DoubleIdentifier)constValue);	
						_rowUpperBound.setAllPositions(constValue.getBeginLine(), constValue.getBeginColumn(), constValue.getEndLine(), constValue.getEndColumn());
					}
					isConst_rowUpperBound = true;
					
				}
			}	
		}
		
		// check 1 < indexed row upper-bound < rows in IndexedIdentifier 
		// (assuming row dims available for upper bound)
		Long rowUB_2 = -1L;
		if (isConst_rowUpperBound) {
				
			if (_rowUpperBound instanceof IntIdentifier) 
				rowUB_2 = ((IntIdentifier)_rowUpperBound).getValue();
			else
				rowUB_2 = Math.round(((DoubleIdentifier)_rowUpperBound).getValue());
			
			if (rowUB_2 < 1)
				throw new LanguageException(this.printErrorLocation() + "upper-bound row index " + rowUB_2 + " is out of bounds. Must be >= 1");
			
			if ((this.getDim1() > 0)  && (rowUB_2 > this.getDim1())) 
				throw new LanguageException(this.printErrorLocation() + "upper-bound row index " + rowUB_2 + " is out of bounds.  Rows in " + this.getName() + ": " + this.getDim1());
		
			if (isConst_rowLowerBound && rowUB_2 < rowLB_1)
				throw new LanguageException(this.printErrorLocation() + "upper-bound row index " + rowUB_2 + " is greater than lower-bound row index " + rowLB_1);
		
		}
	
		
		
		
		if (_colLowerBound instanceof ConstIdentifier && ( _colLowerBound instanceof IntIdentifier || _colLowerBound instanceof DoubleIdentifier )){
			isConst_colLowerBound = true;
		}	
		else if (_colLowerBound instanceof ConstIdentifier){
			throw new LanguageException(this.printErrorLocation() + "attempted to assign lower-bound column index for " + this.toString() + "the non-numeric value " + _colLowerBound.toString());
		}
		
		else if (_colLowerBound != null && _colLowerBound instanceof DataIdentifier && !(_colLowerBound instanceof IndexedIdentifier)) {
			String identifierName = ((DataIdentifier)_colLowerBound).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				
				if (!(constValue instanceof IntIdentifier || constValue instanceof DoubleIdentifier ))
					throw new LanguageException(this.printErrorLocation() + "attempted to assign lower-bound column index for " + this.toString() + " the non-numeric value " + constValue.getOutput().toString());
	
				else{
					if (constValue instanceof IntIdentifier){
						_colLowerBound = new IntIdentifier((IntIdentifier)constValue);
						_colLowerBound.setAllPositions(constValue.getBeginLine(), constValue.getBeginColumn(), constValue.getEndLine(), constValue.getEndColumn());
					}
					else {
						_colLowerBound = new DoubleIdentifier((DoubleIdentifier)constValue);
						_colLowerBound.setAllPositions(constValue.getBeginLine(), constValue.getBeginColumn(), constValue.getEndLine(), constValue.getEndColumn());
					}
					isConst_colLowerBound = true;
				}
			}	
		}
		
		// check 1 < indexed col lower-bound < rows in IndexedIdentifier 
		// (assuming row dims available for upper bound)
		Long colLB_3 = -1L;
		if (isConst_colLowerBound) {
				
			if (_colLowerBound instanceof IntIdentifier) 
				colLB_3 = ((IntIdentifier)_colLowerBound).getValue();
			else
				colLB_3 = Math.round(((DoubleIdentifier)_colLowerBound).getValue());
			
			if (colLB_3 < 1)
				throw new LanguageException(this.printErrorLocation() + "lower-bound column index " + colLB_3 + " is out of bounds. Must be >= 1");
			
			if ((this.getDim1() > 0)  && (colLB_3 > this.getDim1())) 
				throw new LanguageException(this.printErrorLocation() + "lower-bound column index " + colLB_3 + " is out of bounds.  Columns in " + this.getName() + ": " + this.getDim2());
		}
		
		
		if (_colUpperBound instanceof ConstIdentifier && ( _colUpperBound instanceof IntIdentifier || _colUpperBound instanceof DoubleIdentifier )){
			isConst_colUpperBound = true;
		}	
		else if (_colUpperBound instanceof ConstIdentifier){
			throw new LanguageException(this.printErrorLocation() + "attempted to assign upper-bound column index for Indexed Identifier " + this.toString() + "the non-numeric value " + _colUpperBound.toString());
		}
		
		else if (_colUpperBound != null && _colUpperBound instanceof DataIdentifier && !(_colUpperBound instanceof IndexedIdentifier)) {
			String identifierName = ((DataIdentifier)_colUpperBound).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				
				if (!(constValue instanceof IntIdentifier || constValue instanceof DoubleIdentifier ))
					throw new LanguageException(this.printErrorLocation() + "attempted to assign indices for Indexed Identifier " + this.toString() + "the non-numeric value " + constValue.getOutput().toString());
	
				else{
					if (constValue instanceof IntIdentifier){
						_colUpperBound = new IntIdentifier((IntIdentifier)constValue);
						_colUpperBound.setAllPositions(constValue.getBeginLine(), constValue.getBeginColumn(), constValue.getEndLine(), constValue.getEndColumn());
					}
					else {
						_colUpperBound = new DoubleIdentifier((DoubleIdentifier)constValue);
						_colUpperBound.setAllPositions(constValue.getBeginLine(), constValue.getBeginColumn(), constValue.getEndLine(), constValue.getEndColumn());
					}
					isConst_colUpperBound = true;
				}
			}	
		}
		
		// check 1 < indexed col lower-bound < rows in IndexedIdentifier 
		// (assuming row dims available for upper bound)
		Long colUB_4 = -1L;
		if (isConst_colUpperBound) {
				
			if (_colUpperBound instanceof IntIdentifier) 
				colUB_4 = ((IntIdentifier)_colUpperBound).getValue();
			else
				colUB_4 = Math.round(((DoubleIdentifier)_colUpperBound).getValue());
			
			if (colUB_4 < 1)
				throw new LanguageException(this.printErrorLocation() + "upper-bound column index " + colUB_4 + " is out of bounds. Must be >= 1");
			
			if ((this.getDim1() > 0)  && (colUB_4 > this.getDim1())) 
				throw new LanguageException(this.printErrorLocation() + "upper-bound column index " + colUB_4 + " is out of bounds.  Columns in " + this.getName() + ": " + this.getDim2());
		
			if (isConst_colLowerBound && colUB_4 < colLB_3)
				throw new LanguageException(this.printErrorLocation() + "upper-bound column index " + colUB_4 + " is greater than lower-bound column index " + colLB_3);
			
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
			if (_rowUpperBound instanceof IntIdentifier)
				updatedRowDim = ((IntIdentifier)_rowUpperBound).getValue();
			else if (_rowUpperBound instanceof DoubleIdentifier)
				updatedRowDim = Math.round(((DoubleIdentifier)_rowUpperBound).getValue());
		}
			
		// CASE: (lower == constant) && (upper == null) && (dimIndex > 0) --> rowCount - lower bound + 1
		else if (isConst_rowLowerBound && _rowUpperBound == null && this.getDim1() > 0) {
			long rowCount = this.getDim1();
			if (_rowLowerBound instanceof IntIdentifier)
				updatedRowDim = rowCount - ((IntIdentifier)_rowLowerBound).getValue() + 1;
			else if (_rowLowerBound instanceof DoubleIdentifier)
				updatedRowDim = Math.round(rowCount - ((DoubleIdentifier)_rowLowerBound).getValue() + 1);
		}
		// CASE: (lower == constant) && (upper == constant) --> upper bound - lower bound + 1
		else if (isConst_rowLowerBound && isConst_rowUpperBound) {
			if (_rowLowerBound instanceof IntIdentifier && _rowUpperBound instanceof IntIdentifier)
				updatedRowDim = ((IntIdentifier)_rowUpperBound).getValue() - ((IntIdentifier)_rowLowerBound).getValue() + 1;
			
			else if (_rowLowerBound instanceof DoubleIdentifier && _rowUpperBound instanceof DoubleIdentifier)
				updatedRowDim = Math.round( ((DoubleIdentifier)_rowUpperBound).getValue() - ((DoubleIdentifier)_rowLowerBound).getValue() + 1);
			
			else if (_rowLowerBound instanceof IntIdentifier && _rowUpperBound instanceof DoubleIdentifier)
				updatedRowDim = Math.round( ((DoubleIdentifier)_rowUpperBound).getValue() - ((IntIdentifier)_rowLowerBound).getValue() + 1);
			
			else if (_rowLowerBound instanceof DoubleIdentifier && _rowUpperBound instanceof IntIdentifier)
				updatedRowDim = Math.round( ((IntIdentifier)_rowUpperBound).getValue() - ((DoubleIdentifier)_rowLowerBound).getValue() + 1);
			
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
			if (_colUpperBound instanceof IntIdentifier)
				updatedColDim = ((IntIdentifier)_colUpperBound).getValue();
			else if (_colUpperBound instanceof DoubleIdentifier)
				updatedColDim = Math.round(((DoubleIdentifier)_colUpperBound).getValue());
		}
			
		// CASE: (lower == constant) && (upper == null) && (dimIndex > 0) --> colCount - lower bound + 1
		else if (isConst_colLowerBound && _colUpperBound == null && this.getDim2() > 0) {
			long colCount = this.getDim2();
			if (_colLowerBound instanceof IntIdentifier)
				updatedColDim = colCount - ((IntIdentifier)_colLowerBound).getValue() + 1;
			else if (_colLowerBound instanceof DoubleIdentifier)
				updatedColDim = Math.round(colCount - ((IntIdentifier)_colLowerBound).getValue() + 1);
		}
		
		// CASE: (lower == constant) && (upper == constant) --> upper bound - lower bound + 1
		else if (isConst_colLowerBound && isConst_colUpperBound) {
			if (_colLowerBound instanceof IntIdentifier && _colUpperBound instanceof IntIdentifier)
				updatedColDim = ((IntIdentifier)_colUpperBound).getValue() - ((IntIdentifier)_colLowerBound).getValue() + 1;
			
			else if (_colLowerBound instanceof DoubleIdentifier && _colUpperBound instanceof DoubleIdentifier)
				updatedColDim = Math.round( ((DoubleIdentifier)_colUpperBound).getValue() - ((DoubleIdentifier)_colLowerBound).getValue() + 1);
			
			else if (_colLowerBound instanceof IntIdentifier && _colUpperBound instanceof DoubleIdentifier)
				updatedColDim = Math.round( ((DoubleIdentifier)_colUpperBound).getValue() - ((IntIdentifier)_colLowerBound).getValue() + 1);
			
			else if (_colLowerBound instanceof DoubleIdentifier && _colUpperBound instanceof IntIdentifier)
				updatedColDim = Math.round( ((IntIdentifier)_colUpperBound).getValue() - ((DoubleIdentifier)_colLowerBound).getValue() + 1);
			
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
		
		// set the "original" dimensions (dimension values of object being indexed
		this.setOriginalDimensions(this.getDim1(), this.getDim2());
		
		// set the updated dimensions
		this.setDimensions(updatedRowDim, updatedColDim);
	}
	
	public void setOriginalDimensions(long passedDim1, long passedDim2){
		this._origDim1 = passedDim1;
		this._origDim2 = passedDim2;
	}
	
	public long getOrigDim1() { return this._origDim1; }
	public long getOrigDim2() { return this._origDim2; }
	
	
	public Expression rewriteExpression(String prefix) throws LanguageException {
		
		IndexedIdentifier newIndexedIdentifier = new IndexedIdentifier(this.getName(), this._rowLowerEqualsUpper, this._colLowerEqualsUpper);
		
		newIndexedIdentifier._beginLine 	= this._beginLine;
		newIndexedIdentifier._beginColumn 	= this._beginColumn;
		newIndexedIdentifier._endLine 		= this._endLine;
		newIndexedIdentifier._endColumn 	= this._endColumn;
			
		// set dimensionality information and other Identifier specific properties for new IndexedIdentifier
		newIndexedIdentifier.setProperties(this);
		newIndexedIdentifier.setOriginalDimensions(this._origDim1, this._origDim2);
		
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
			throw new ParseException(this.printErrorLocation() + "matrix indeices must be specified for 2 dimensions") ; //"[E] matrix indices must be specified for 2 dimensions -- currently specified indices for " + passed.size() + " dimensions ");
		
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
			throw new ParseException(this.printErrorLocation() + "row indices are length " + rowIndices.size() + " -- should be either 1 or 2");
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
			throw new ParseException(this.printErrorLocation() + "column indices are length " + + colIndices.size() + " -- should be either 1 or 2");
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
	
	public void setAllPositions (int blp, int bcp, ArrayList<ArrayList<Expression>> exprListList){
		this._beginLine = blp;
		this._beginColumn = bcp;
		
		Expression rightMostIndexExpr = null;
		if (exprListList != null){
			for(ArrayList<Expression> exprList : exprListList){
				if (exprList != null){
					for (Expression expr : exprList){
						if (expr != null) {
							rightMostIndexExpr = expr;
						}
					}
				}
			}
		}
		
		if (rightMostIndexExpr != null){
			this._endLine 	= rightMostIndexExpr.getEndLine();
			this._endColumn = rightMostIndexExpr.getEndColumn();
		}
		else {
			this._endLine = blp;
			this._endColumn = bcp;
		}
	}
	
	
} // end class
	
class IndexPair {
	
	public long _row, _col;
	
	public IndexPair (long row, long col){
		_row = row;
		_col = col;
	}
} // end class
