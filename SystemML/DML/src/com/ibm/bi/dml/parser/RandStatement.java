package com.ibm.bi.dml.parser;

import java.util.HashMap;

import com.ibm.bi.dml.parser.Expression.FormatType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LanguageException;


public class RandStatement extends Statement
{
		
	// target identifier which will hold the random object
	private DataIdentifier _id = null;
	
	// parameter values for rand statement (with default values assigned)
	private Expression _rowsExpr = new IntIdentifier(1L);
	private Expression _colsExpr = new IntIdentifier(1L);
	private Expression _minValueExpr = new DoubleIdentifier(0.0);
	private Expression _maxValueExpr = new DoubleIdentifier(1.0);
	private Expression _sparsityExpr = new DoubleIdentifier(1.0);
	private Expression _seedExpr = new IntIdentifier(-1L);
	private Expression _pdfExpr = new StringIdentifier("uniform");
	
	
	// rewrite the RandStatement to support function inlining 
	// creates a deep-copy of RandStatement
	public Statement rewriteStatement(String prefix) throws LanguageException{
		
		RandStatement newStatement = new RandStatement();
	
		// rewrite data identifier for target
		newStatement._id = (DataIdentifier)this._id.rewriteExpression(prefix);

		// rewrite the indexed expressions
		newStatement._rowsExpr = this._rowsExpr.rewriteExpression(prefix);
		newStatement._colsExpr = this._colsExpr.rewriteExpression(prefix);
		newStatement._minValueExpr = this._minValueExpr.rewriteExpression(prefix);
		newStatement._maxValueExpr = this._maxValueExpr.rewriteExpression(prefix);
		newStatement._sparsityExpr = this._sparsityExpr.rewriteExpression(prefix);
		newStatement._seedExpr = this._seedExpr.rewriteExpression(prefix);
		newStatement._pdfExpr = this._pdfExpr.rewriteExpression(prefix);
		
		return newStatement;
	}
	
	public RandStatement(){}
	
	public RandStatement(DataIdentifier id){
		_id = id;
		
		_rowsExpr = new IntIdentifier(1L);
		_colsExpr = new IntIdentifier(1L);
		_minValueExpr = new DoubleIdentifier(0.0);
		_maxValueExpr = new DoubleIdentifier(1.0);
		_sparsityExpr = new DoubleIdentifier(1.0);
		_seedExpr = new IntIdentifier(-1L);
		_pdfExpr = new StringIdentifier("uniform");
		
	}
	
	// class getter methods
	public DataIdentifier getIdentifier(){ return _id; }
	public Expression getRowsExpr() { return _rowsExpr; } 
	public Expression getColsExpr() { return _colsExpr; }
	public Expression getMinValueExpr() { return _minValueExpr; }
	public Expression getMaxValueExpr() { return _maxValueExpr; }
	public Expression getSparsityExpr() { return _sparsityExpr; }
	public Expression getSeedExpr() { return _seedExpr; }
	public Expression getPdfExpr()  { return _pdfExpr; }
	
	
	public void addExprParam(String paramName, Expression paramValue) throws ParseException
	{
		if(paramName.equals("rows")) 
			_rowsExpr = paramValue;
		else if(paramName.equals("cols"))
			_colsExpr = paramValue;
		else if(paramName.equals("min"))
			_minValueExpr = paramValue;
		else if(paramName.equals("max"))
			_maxValueExpr = paramValue;
		else if (paramName.equals("sparsity"))
			_sparsityExpr = paramValue;
		else if (paramName.equals("seed"))
			_seedExpr = paramValue; 
		else if (paramName.equals("pdf"))
			_pdfExpr = paramValue;
		
		else
			throw new ParseException("unexpected parameter \"" + paramName +
					"\". Legal parameters for Rand statement are " +
					"(capitalization-sensitive): rows, cols, min, max, sparsity, seed, pdf ");
	}
	
	
	// performs basic constant propagation by replacing DataIdentifier with ConstIdentifier 
	// perform "best-effort" validation of exprParams.  If exprParam is a ConstIdentifier expression
	//	(has constant value), then perform static validation.
	public void performConstantPropagation(HashMap<String, ConstIdentifier> currConstVars) throws LanguageException{
		
		// handle exprParam for rows
		if (_rowsExpr instanceof DataIdentifier && !(_rowsExpr instanceof IndexedIdentifier)) {
			
			// check if the DataIdentifier variable is a ConstIdentifier
			String identifierName = ((DataIdentifier)_rowsExpr).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof IntIdentifier && ((IntIdentifier)constValue).getValue() >= 1))
					throw new LanguageException("ERROR:  In rand statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
				else
					_rowsExpr = new IntIdentifier((IntIdentifier)constValue);
			}
		}	
		
		// handle exprParam for cols
		if (_colsExpr instanceof DataIdentifier && !(_colsExpr instanceof IndexedIdentifier)) {
			
			// check if the DataIdentifier variable is a ConstIdentifier
			String identifierName = ((DataIdentifier)_colsExpr).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof IntIdentifier && ((IntIdentifier)constValue).getValue() >= 1))
					throw new LanguageException("ERROR:  In rand statement, can only assign cols a long " +
							"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
				else
					_colsExpr = new IntIdentifier((IntIdentifier)constValue);
			}
		}
		
		// handle exprParam for min value
		if (_minValueExpr instanceof DataIdentifier && !(_minValueExpr instanceof IndexedIdentifier)) {
			
			// check if the DataIdentifier variable is a ConstIdentifier
			String identifierName = ((DataIdentifier)_minValueExpr).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof IntIdentifier || constValue instanceof DoubleIdentifier))
					throw new LanguageException("ERROR:  In rand statement, can only assign min a double " +
							"value -- attempted to assign value: " + constValue.toString());
				else
					_minValueExpr = new DoubleIdentifier(new Double(constValue.toString()));
			}
		}
		
		// handle exprParam for max value
		if (_maxValueExpr instanceof DataIdentifier && !(_maxValueExpr instanceof IndexedIdentifier)) {
			
			// check if the DataIdentifier variable is a ConstIdentifier
			String identifierName = ((DataIdentifier)_maxValueExpr).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof IntIdentifier || constValue instanceof DoubleIdentifier))
					throw new LanguageException("ERROR:  In rand statement, can only assign max a double " +
							"value -- attempted to assign value: " + constValue.toString());
				else
					_maxValueExpr = new DoubleIdentifier(new Double(constValue.toString()));
			}
		}
		
		// handle exprParam for seed
		if (_seedExpr instanceof DataIdentifier && !(_seedExpr instanceof IndexedIdentifier)) {
			
			// check if the DataIdentifier variable is a ConstIdentifier
			String identifierName = ((DataIdentifier)_seedExpr).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof IntIdentifier))
					throw new LanguageException("ERROR:  In rand statement, can only assign seed a long " +
							"value -- attempted to assign value: " + constValue.toString());
				else
					_seedExpr = new IntIdentifier((IntIdentifier)constValue);
			}
		}
		
		// handle exprParam for pdf (probability density function)
		if (_pdfExpr instanceof DataIdentifier && !(_pdfExpr instanceof IndexedIdentifier)) {
			
			// check if the DataIdentifier variable is a ConstIdentifier
			String identifierName = ((DataIdentifier)_pdfExpr).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof StringIdentifier && (constValue.toString().equals(""))))
					throw new LanguageException("ERROR:  In rand statement, can only assign pdf " +
							"following one of following string values (capitalization-sensitive): uniform. " +
							"Attempted to assign value: " + constValue.toString());
				else
					_pdfExpr = new IntIdentifier((IntIdentifier)constValue);	}			
		}
		
	} // end method performConstantPropagation
	
		
	public void setIdentifierProperties() throws LanguageException
	{
		long rowsLong = -1, colsLong = -1;
		
		if (_rowsExpr instanceof IntIdentifier)
			rowsLong = ((IntIdentifier)_rowsExpr).getValue();
		
		if (_colsExpr instanceof IntIdentifier)
			colsLong = ((IntIdentifier)_colsExpr).getValue();
		
		_id.setFormatType(FormatType.BINARY);
		_id.setValueType(ValueType.DOUBLE);
		_id.setDimensions(rowsLong, colsLong);
		_id.computeDataType();
		
		if (_id instanceof IndexedIdentifier){
			System.out.println("WARNING: Output for RandStatement may have incorrect size information");
		}
		
	}
	
	
	@Override
	public boolean controlStatement() { return false; }

	@Override
	public VariableSet initializebackwardLV(VariableSet lo){ return lo; }

	@Override
	public void initializeforwardLV(VariableSet activeIn){}

	@Override

	public VariableSet variablesRead(){
		VariableSet result = new VariableSet();
				
		// add variables read by parameter expressions
		result.addVariables(_rowsExpr.variablesRead());
		result.addVariables(_colsExpr.variablesRead());
		result.addVariables(_minValueExpr.variablesRead());
		result.addVariables(_maxValueExpr.variablesRead());
		result.addVariables(_sparsityExpr.variablesRead());
		result.addVariables(_seedExpr.variablesRead());
		result.addVariables(_pdfExpr.variablesRead());
		
		// for LHS IndexedIdentifier, add variables for indexing expressions in target
		if (_id instanceof IndexedIdentifier)
			result.addVariables(((IndexedIdentifier)_id).variablesRead());
			
		return result;
	}

	@Override 
	public VariableSet variablesUpdated()
	{
		// add target variable
		VariableSet result = new VariableSet();
		result.addVariable(_id.getName(), _id);
		return result;
	}
    
    /**
     * <p>Returns a string representation of the rand function call.</p>
     */
    public String toString()
    {
        StringBuffer sb = new StringBuffer();
        sb.append(_id.getName() + " = Rand( ");
        sb.append(  "rows=" + _rowsExpr.toString());
        sb.append(", cols=" + _colsExpr.toString());
        sb.append(", min="  + _minValueExpr.toString());
        sb.append(", max="  + _maxValueExpr.toString());
        sb.append(", sparsity=" + _sparsityExpr.toString());
        sb.append(", pdf=" +      _pdfExpr.toString());
        if (_seedExpr instanceof IntIdentifier && ((IntIdentifier)_seedExpr).getValue() == -1L)
        	sb.append(", seed=RANDOM");
        else
        	sb.append(", seed=" + _seedExpr.toString());
        sb.append(" );");
        return sb.toString();
    }
}
