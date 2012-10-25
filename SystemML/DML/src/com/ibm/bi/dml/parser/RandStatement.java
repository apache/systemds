package com.ibm.bi.dml.parser;

import java.util.HashMap;

import com.ibm.bi.dml.hops.RandOp;
import com.ibm.bi.dml.parser.Expression.FormatType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LanguageException;


public class RandStatement extends Statement
{
	
	public static final String[] RAND_VALID_PARAM_NAMES = 
	{ RAND_ROWS, RAND_COLS, RAND_MIN, RAND_MAX, RAND_SPARSITY, RAND_SEED, RAND_PDF}; 

	
	// target identifier which will hold the random object
	private DataIdentifier _id = null;
	private HashMap<String,Expression> _exprParams = null;
	

	// rewrite the RandStatement to support function inlining 
	// creates a deep-copy of RandStatement
	public Statement rewriteStatement(String prefix) throws LanguageException{
		
		RandStatement newStatement = new RandStatement();
		newStatement._beginLine		= this.getBeginLine();
		newStatement._beginColumn	= this.getBeginColumn();
		newStatement._endLine		= this.getEndLine();
		newStatement._endColumn 	= this.getEndColumn();
	
		// rewrite data identifier for target (creates deep copy)
		newStatement._id = (DataIdentifier)this._id.rewriteExpression(prefix);

		// rewrite the parameters (creates deep copy)
		HashMap<String,Expression> newExprParams = new HashMap<String,Expression>();
		for (String key : _exprParams.keySet()){
			Expression newExpr = _exprParams.get(key).rewriteExpression(prefix);
			newExprParams.put(key, newExpr);
		}	
		newStatement._exprParams = newExprParams;
	
		return newStatement;
	}
	
	public RandStatement(){}
	
	public RandStatement(DataIdentifier id){
		_id = id;
		_exprParams = new HashMap<String,Expression>();
		
		// set defaults for parameter values
		_exprParams.put(RAND_ROWS, 	new IntIdentifier(-1L));
		_exprParams.put(RAND_COLS, 	new IntIdentifier(-1L));
		_exprParams.put(RAND_MIN, 	new DoubleIdentifier(0.0));
		_exprParams.put(RAND_MAX, 	new DoubleIdentifier(1.0));
		_exprParams.put(RAND_SPARSITY, 	new DoubleIdentifier(1.0)); 			
		_exprParams.put(RAND_SEED,		new IntIdentifier(RandOp.UNSPECIFIED_SEED));
		_exprParams.put(RAND_PDF,		new StringIdentifier("uniform"));		
	}
	
	// class getter methods
	public DataIdentifier getIdentifier(){ return _id; }
	public Expression getExprParam(String paramName) { return _exprParams.get(paramName); } 
	
	public void addExprParam(String paramName, Expression paramValue) throws ParseException
	{
		// check name is valid
		boolean found = false;
		for (String name : RAND_VALID_PARAM_NAMES){
			if (name.equals(paramName))
				found = true;
		}
		if (!found)
			throw new ParseException(paramValue.printErrorLocation() + "unexpected parameter \"" + paramName +
					"\". Legal parameters for Rand statement are " 
					+ "(capitalization-sensitive): " 	+ RAND_ROWS 	
					+ ", " + RAND_COLS		+ ", " + RAND_MIN + ", " + RAND_MAX  	
					+ ", " + RAND_SPARSITY + ", " + RAND_SEED     + ", " + RAND_PDF);
		
		// add the parameter to expression list
		_exprParams.put(paramName,paramValue);
		
	}
	
	// performs basic constant propagation by replacing DataIdentifier with ConstIdentifier 
	// perform "best-effort" validation of exprParams.  If exprParam is a ConstIdentifier expression
	//	(has constant value), then perform static validation.
	public void validateStatement(VariableSet ids, HashMap<String, ConstIdentifier> currConstVars) throws LanguageException{
		
		//////////////////////////////////////////////////////////////////////////
		// handle exprParam for rows
		//////////////////////////////////////////////////////////////////////////
		Expression rowsExpr = _exprParams.get(RAND_ROWS);
		if (rowsExpr instanceof DataIdentifier && !(rowsExpr instanceof IndexedIdentifier)) {
			
			// check if the DataIdentifier variable is a ConstIdentifier
			String identifierName = ((DataIdentifier)rowsExpr).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof IntIdentifier && ((IntIdentifier)constValue).getValue() >= 1))
					throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
				else{
					rowsExpr = new IntIdentifier((IntIdentifier)constValue);
					rowsExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					_exprParams.put(RAND_ROWS, rowsExpr);
				}
			}
			else {
				throw new LanguageException(this.printErrorLocation() + "In rand statement, must assign constant value to rows dimension");
			}
		}	
		else {
			// handle general expression
			rowsExpr.validateExpression(ids.getVariables(), currConstVars);
		}
		
		///////////////////////////////////////////////////////////////////////
		// handle exprParam for cols
		///////////////////////////////////////////////////////////////////////
		Expression colsExpr = _exprParams.get(RAND_COLS);
		if (colsExpr instanceof DataIdentifier && !(colsExpr instanceof IndexedIdentifier)) {
			
			// check if the DataIdentifier variable is a ConstIdentifier
			String identifierName = ((DataIdentifier)colsExpr).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof IntIdentifier && ((IntIdentifier)constValue).getValue() >= 1))
					throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign cols a long " +
							"(integer) value >= 1 -- attempted to assign value: " + constValue.toString());
				else{
					colsExpr = new IntIdentifier((IntIdentifier)constValue);
					colsExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					_exprParams.put(RAND_COLS, colsExpr);
				}
			}
			else {
				throw new LanguageException(this.printErrorLocation() + "In rand statement, must assign constant value to cols dimension");
			}
		}
		else {
			// handle general expression
			colsExpr.validateExpression(ids.getVariables(), currConstVars);
		}
		
		///////////////////////////////////////////////////////////////////////
		// handle exprParam for min value
		///////////////////////////////////////////////////////////////////////
		Expression minValueExpr = _exprParams.get(RAND_MIN);
		if (minValueExpr instanceof DataIdentifier && !(minValueExpr instanceof IndexedIdentifier)) {
			
			// check if the DataIdentifier variable is a ConstIdentifier
			String identifierName = ((DataIdentifier)minValueExpr).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof IntIdentifier || constValue instanceof DoubleIdentifier))
					throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign min a double " +
							"value -- attempted to assign value: " + constValue.toString());
				else {
					minValueExpr = new DoubleIdentifier(new Double(constValue.toString()));
					minValueExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					_exprParams.put(RAND_MIN, minValueExpr);
				}
			}
			else {
				throw new LanguageException(this.printErrorLocation() + "In rand statement, must assign constant value to min parameter");
			}
		}
		else {
			// handle general expression
			minValueExpr.validateExpression(ids.getVariables(), currConstVars);
		}
		
		///////////////////////////////////////////////////////////////////////
		// handle exprParam for max value
		///////////////////////////////////////////////////////////////////////
		Expression maxValueExpr = _exprParams.get(RAND_MAX);
		if (maxValueExpr instanceof DataIdentifier && !(maxValueExpr instanceof IndexedIdentifier)) {
			
			// check if the DataIdentifier variable is a ConstIdentifier
			String identifierName = ((DataIdentifier)maxValueExpr).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof IntIdentifier || constValue instanceof DoubleIdentifier))
					throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign max a double " +
							"value -- attempted to assign value: " + constValue.toString());
				else {
					maxValueExpr = new DoubleIdentifier(new Double(constValue.toString()));
					maxValueExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					_exprParams.put(RAND_MAX, maxValueExpr);
				}
			}
			else {
				throw new LanguageException(this.printErrorLocation() + "In rand statement, must assign constant value to max parameter");
			}
		}
		else {
			// handle general expression
			maxValueExpr.validateExpression(ids.getVariables(), currConstVars);
		}
		
		///////////////////////////////////////////////////////////////////////
		// handle exprParam for seed
		///////////////////////////////////////////////////////////////////////
		Expression seedExpr = _exprParams.get(RAND_SEED);
		if (seedExpr instanceof DataIdentifier && !(seedExpr instanceof IndexedIdentifier)) {
			
			// check if the DataIdentifier variable is a ConstIdentifier
			String identifierName = ((DataIdentifier)seedExpr).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof IntIdentifier))
					throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign seed a long " +
							"value -- attempted to assign value: " + constValue.toString());
				else {
					seedExpr = new IntIdentifier((IntIdentifier)constValue);
					seedExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					_exprParams.put(RAND_SEED, seedExpr);
				}
			}
		}
		else {
			// handle general expression
			seedExpr.validateExpression(ids.getVariables(), currConstVars);
		}
		
		
		///////////////////////////////////////////////////////////////////////
		// handle exprParam for sparsity
		///////////////////////////////////////////////////////////////////////
		Expression sparsityExpr = _exprParams.get(RAND_SPARSITY);
		if (sparsityExpr instanceof DataIdentifier && !(sparsityExpr instanceof IndexedIdentifier)) {
			
			// check if the DataIdentifier variable is a ConstIdentifier
			String identifierName = ((DataIdentifier)sparsityExpr).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof DoubleIdentifier))
					throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign sparsity a double " +
							"value -- attempted to assign value: " + constValue.toString());
				else {
					sparsityExpr = new IntIdentifier((IntIdentifier)constValue);
					sparsityExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					_exprParams.put(RAND_SPARSITY, sparsityExpr);
				}
			}
		}
		else {
			// handle general expression
			sparsityExpr.validateExpression(ids.getVariables(), currConstVars);
		}
		
		///////////////////////////////////////////////////////////////////////
		// handle exprParam for pdf (probability density function)
		///////////////////////////////////////////////////////////////////////
		Expression pdfExpr = _exprParams.get(RAND_PDF);
		if (pdfExpr instanceof DataIdentifier && !(pdfExpr instanceof IndexedIdentifier)) {
			
			// check if the DataIdentifier variable is a ConstIdentifier
			String identifierName = ((DataIdentifier)pdfExpr).getName();
			if (currConstVars.containsKey(identifierName)){
				ConstIdentifier constValue = currConstVars.get(identifierName);
				if (!(constValue instanceof StringIdentifier && (constValue.toString().equals(""))))
					throw new LanguageException(this.printErrorLocation() + "In rand statement, can only assign pdf " +
							"following one of following string values (capitalization-sensitive): uniform. " +
							"Attempted to assign value: " + constValue.toString());
				else {
					pdfExpr = new IntIdentifier((IntIdentifier)constValue);	
					pdfExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					_exprParams.put(RAND_PDF, pdfExpr);
				}
			}
		}
		else {
			pdfExpr.validateExpression(ids.getVariables(), currConstVars);
		}
		
	} // end method performConstantPropagation
	
		
	public void setIdentifierProperties() throws LanguageException
	{
		long rowsLong = -1L, colsLong = -1L;
		
		
		if (_exprParams.get(RAND_ROWS) instanceof IntIdentifier)
			rowsLong = ((IntIdentifier)_exprParams.get(RAND_ROWS)).getValue();
		
		if (_exprParams.get(RAND_COLS) instanceof IntIdentifier)
			colsLong = ((IntIdentifier)_exprParams.get(RAND_COLS)).getValue();
		
		_id.setFormatType(FormatType.BINARY);
		_id.setValueType(ValueType.DOUBLE);
		_id.setDimensions(rowsLong, colsLong);
		if (_id instanceof IndexedIdentifier){
			((IndexedIdentifier) _id).setOriginalDimensions(_id.getDim1(), _id.getDim2());
		}
		_id.computeDataType();
		
		if (_id instanceof IndexedIdentifier){
			System.out.println(this.printWarningLocation() + "Output for RandStatement may have incorrect size information");
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
		for (String key : _exprParams.keySet()){
			result.addVariables(_exprParams.get(key).variablesRead());
		}
			
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
        sb.append(  "rows=" + _exprParams.get(RAND_ROWS).toString());
        sb.append(", cols=" + _exprParams.get(RAND_COLS).toString());
        sb.append(", min="  + _exprParams.get(RAND_MIN).toString());
        sb.append(", max="  + _exprParams.get(RAND_MAX).toString());
        sb.append(", sparsity=" + _exprParams.get(RAND_SPARSITY).toString());
        sb.append(", pdf=" +      _exprParams.get(RAND_PDF).toString());
        if (_exprParams.get(RAND_SEED) instanceof IntIdentifier && ((IntIdentifier)_exprParams.get(RAND_SEED)).getValue() == -1L)
        	sb.append(", seed=RANDOM");
        else
        	sb.append(", seed=" + _exprParams.get(RAND_SEED).toString());
        sb.append(" );");
        return sb.toString();
    }

    @Override
    public void setAllPositions(int blp, int bcp, int elp, int ecp){
		_beginLine	 = blp; 
		_beginColumn = bcp; 
		_endLine 	 = elp;
		_endColumn 	 = ecp;
		
		for (String key : _exprParams.keySet()){
			Expression expr = _exprParams.get(key);
			if (expr.getBeginLine() == 0)
				expr._beginLine = _beginLine;
			if (expr.getBeginColumn() == 0)
				expr._beginColumn = _beginColumn;
			if (expr.getEndLine() == 0)
				expr._endLine = _endLine;
			if (expr.getEndColumn() == 0)
				expr._endColumn = _endColumn;
			_exprParams.put(key, expr);
		}
	}

}
