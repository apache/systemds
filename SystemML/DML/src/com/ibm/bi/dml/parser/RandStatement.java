package com.ibm.bi.dml.parser;

import java.util.HashMap;

import com.ibm.bi.dml.hops.RandOp;
import com.ibm.bi.dml.parser.Expression.DataOp;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LanguageException;


public class RandStatement extends Statement
{
	
	public static final String[] RAND_VALID_PARAM_NAMES = 
	{ RAND_ROWS, RAND_COLS, RAND_MIN, RAND_MAX, RAND_SPARSITY, RAND_SEED, RAND_PDF}; 

	public static final String RAND_PDF_UNIFORM = "uniform";
	
	// target identifier which will hold the random object
	private DataIdentifier _id = null;
	private DataExpression _paramsExpr;

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
		DataOp op = _paramsExpr.getOpCode();
		HashMap<String,Expression> newExprParams = new HashMap<String,Expression>();
		for (String key : _paramsExpr.getVarParams().keySet()){
			Expression newExpr = _paramsExpr.getVarParam(key).rewriteExpression(prefix);
			newExprParams.put(key, newExpr);
		}	

		DataExpression newParamerizedExpr = new DataExpression(op, newExprParams);
		newParamerizedExpr.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		newStatement.setExprParams(newParamerizedExpr);

		
		return newStatement;
	}
	
	public RandStatement(){}
	
	public RandStatement(DataIdentifier id){
		_id = id;
		_paramsExpr = new DataExpression(DataOp.RAND);
	}
	

	public void setRandDefault(){
		if (_paramsExpr.getVarParam(RAND_ROWS)== null){
			IntIdentifier id = new IntIdentifier(1L);
		    id.setDimensions(0,0);
            id.computeDataType();
            id.setValueType(ValueType.INT);
			_paramsExpr.addVarParam(RAND_ROWS, 	id);
		}
		if (_paramsExpr.getVarParam(RAND_COLS)== null){
			IntIdentifier id = new IntIdentifier(1L);
		    id.setDimensions(0,0);
            id.computeDataType();
            id.setValueType(ValueType.INT);
            _paramsExpr.addVarParam(RAND_COLS, 	id);
		}
		if (_paramsExpr.getVarParam(RAND_MIN)== null){
			DoubleIdentifier id = new DoubleIdentifier(0.0);
		    id.setDimensions(0,0);
            id.computeDataType();
            id.setValueType(ValueType.DOUBLE);
			_paramsExpr.addVarParam(RAND_MIN, id);
		}
		if (_paramsExpr.getVarParam(RAND_MAX)== null){
			DoubleIdentifier id = new DoubleIdentifier(1.0);
		    id.setDimensions(0,0);
            id.computeDataType();
            id.setValueType(ValueType.DOUBLE);
			_paramsExpr.addVarParam(RAND_MAX, id);
		}
		if (_paramsExpr.getVarParam(RAND_SPARSITY)== null){
			DoubleIdentifier id = new DoubleIdentifier(1.0);
		    id.setDimensions(0,0);
            id.computeDataType();
            id.setValueType(ValueType.DOUBLE);
			_paramsExpr.addVarParam(RAND_SPARSITY,	id);
		}
		if (_paramsExpr.getVarParam(RAND_SEED)== null){
			IntIdentifier id = new IntIdentifier(RandOp.UNSPECIFIED_SEED);
		    id.setDimensions(0,0);
            id.computeDataType();
            id.setValueType(ValueType.INT);
			_paramsExpr.addVarParam(RAND_SEED, id);
		}
		if (_paramsExpr.getVarParam(RAND_PDF)== null){
			StringIdentifier id = new StringIdentifier(RAND_PDF_UNIFORM);
		    id.setDimensions(0,0);
            id.computeDataType();
            id.setValueType(ValueType.STRING);
			_paramsExpr.addVarParam(RAND_PDF, id);
		}
		//setIdentifierProperties();
	}
	
	// class getter methods
	public DataIdentifier getIdentifier(){ return _id; }

	public Expression getExprParam(String name){
		return _paramsExpr.getVarParam(name);
	}
	
	public void setExprParams(DataExpression paramsExpr) {
		_paramsExpr = paramsExpr;
	}
	
	public DataExpression getSource(){
		return _paramsExpr;
	}
	
	public void addExprParam(String paramName, Expression paramValue) throws ParseException
	{
		// check name is valid
		boolean found = false;
		for (String name : RAND_VALID_PARAM_NAMES){
			if (name.equals(paramName)) {
				found = true;
				break;
			}			
		}
		if (!found){
			
			throw new ParseException(paramValue.printErrorLocation() + "unexpected parameter \"" + paramName +
					"\". Legal parameters for Rand statement are " 
					+ "(capitalization-sensitive): " 	+ RAND_ROWS 	
					+ ", " + RAND_COLS		+ ", " + RAND_MIN + ", " + RAND_MAX  	
					+ ", " + RAND_SPARSITY + ", " + RAND_SEED     + ", " + RAND_PDF);
		}
		if (_paramsExpr.getVarParam(paramName) != null){
			throw new ParseException(paramValue.printErrorLocation() + "attempted to add Rand statement parameter " + paramValue + " more than once");
		}
		// Process the case where user provides double values to rows or cols
		if (paramName.equals(RAND_ROWS) && paramValue instanceof DoubleIdentifier){
			paramValue = new IntIdentifier((int)((DoubleIdentifier)paramValue).getValue());
		    ((IntIdentifier)paramValue).setDimensions(0,0);
            ((IntIdentifier)paramValue).computeDataType();
            ((IntIdentifier)paramValue).setValueType(ValueType.INT);
		}
		else if (paramName.equals(RAND_COLS) && paramValue instanceof DoubleIdentifier){
			paramValue = new IntIdentifier((int)((DoubleIdentifier)paramValue).getValue());
		    ((IntIdentifier)paramValue).setDimensions(0,0);
            ((IntIdentifier)paramValue).computeDataType();
            ((IntIdentifier)paramValue).setValueType(ValueType.INT);
		}
			
		// add the parameter to expression list
		paramValue.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		_paramsExpr.addVarParam(paramName,paramValue);
		
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
		
		HashMap<String,Expression> paramsExpr = _paramsExpr.getVarParams();
		// add variables read by parameter expressions
		for (String key : paramsExpr.keySet()){
			result.addVariables(paramsExpr.get(key).variablesRead());
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
        sb.append(  "rows=" + _paramsExpr.getVarParam(RAND_ROWS).toString());
        sb.append(", cols=" + _paramsExpr.getVarParam(RAND_COLS).toString());
        sb.append(", min="  + _paramsExpr.getVarParam(RAND_MIN).toString());
        sb.append(", max="  + _paramsExpr.getVarParam(RAND_MAX).toString());
        sb.append(", sparsity=" + _paramsExpr.getVarParam(RAND_SPARSITY).toString());
        sb.append(", pdf=" +      _paramsExpr.getVarParam(RAND_PDF).toString());
        if (_paramsExpr.getVarParam(RAND_SEED) instanceof IntIdentifier && ((IntIdentifier)_paramsExpr.getVarParam(RAND_SEED)).getValue() == -1L)
        	sb.append(", seed=RANDOM");
        else
        	sb.append(", seed=" + _paramsExpr.getVarParam(RAND_SEED).toString());
        sb.append(" );");
        return sb.toString();
    }

    @Override
    public void setAllPositions(int blp, int bcp, int elp, int ecp){
		_beginLine	 = blp; 
		_beginColumn = bcp; 
		_endLine 	 = elp;
		_endColumn 	 = ecp;
		
		HashMap<String,Expression> paramsExpr = _paramsExpr.getVarParams();
		
		for (String key : paramsExpr.keySet()){
			Expression expr = paramsExpr.get(key);
			if (expr.getBeginLine() == 0)
				expr._beginLine = _beginLine;
			if (expr.getBeginColumn() == 0)
				expr._beginColumn = _beginColumn;
			if (expr.getEndLine() == 0)
				expr._endLine = _endLine;
			if (expr.getEndColumn() == 0)
				expr._endColumn = _endColumn;
			paramsExpr.put(key, expr);
		};
		_paramsExpr.setVarParams(paramsExpr);
	}

}
