/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.Hop.ParamBuiltinOp;
import com.ibm.bi.dml.parser.LanguageException.LanguageErrorCodes;


public class ParameterizedBuiltinFunctionExpression extends DataIdentifier 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private ParameterizedBuiltinFunctionOp _opcode;
	private HashMap<String,Expression> _varParams;
	
	public static final String TF_FN_PARAM_DATA = "target";
	public static final String TF_FN_PARAM_TXMTD = "transformPath";
	public static final String TF_FN_PARAM_TXSPEC = "transformSpec";
	public static final String TF_FN_PARAM_APPLYMTD = "applyTransformPath";
	public static final String TF_FN_PARAM_OUTNAMES = "outputNames";
	
	private static HashMap<String, Expression.ParameterizedBuiltinFunctionOp> opcodeMap;
	static {
		opcodeMap = new HashMap<String, Expression.ParameterizedBuiltinFunctionOp>();
		opcodeMap.put("aggregate", Expression.ParameterizedBuiltinFunctionOp.GROUPEDAGG);
		opcodeMap.put("groupedAggregate", Expression.ParameterizedBuiltinFunctionOp.GROUPEDAGG);
		opcodeMap.put("removeEmpty",Expression.ParameterizedBuiltinFunctionOp.RMEMPTY);
		opcodeMap.put("replace", 	Expression.ParameterizedBuiltinFunctionOp.REPLACE);
		opcodeMap.put("order", 		Expression.ParameterizedBuiltinFunctionOp.ORDER);
		
		// Distribution Functions
		opcodeMap.put("cdf",	Expression.ParameterizedBuiltinFunctionOp.CDF);
		opcodeMap.put("pnorm",	Expression.ParameterizedBuiltinFunctionOp.PNORM);
		opcodeMap.put("pt",		Expression.ParameterizedBuiltinFunctionOp.PT);
		opcodeMap.put("pf",		Expression.ParameterizedBuiltinFunctionOp.PF);
		opcodeMap.put("pchisq",	Expression.ParameterizedBuiltinFunctionOp.PCHISQ);
		opcodeMap.put("pexp",	Expression.ParameterizedBuiltinFunctionOp.PEXP);
		
		opcodeMap.put("icdf",	Expression.ParameterizedBuiltinFunctionOp.INVCDF);
		opcodeMap.put("qnorm",	Expression.ParameterizedBuiltinFunctionOp.QNORM);
		opcodeMap.put("qt",		Expression.ParameterizedBuiltinFunctionOp.QT);
		opcodeMap.put("qf",		Expression.ParameterizedBuiltinFunctionOp.QF);
		opcodeMap.put("qchisq",	Expression.ParameterizedBuiltinFunctionOp.QCHISQ);
		opcodeMap.put("qexp",	Expression.ParameterizedBuiltinFunctionOp.QEXP);

		// data transformation functions
		opcodeMap.put("transform",	Expression.ParameterizedBuiltinFunctionOp.TRANSFORM);
	}
	
	public static HashMap<Expression.ParameterizedBuiltinFunctionOp, ParamBuiltinOp> pbHopMap;
	static {
		pbHopMap = new HashMap<Expression.ParameterizedBuiltinFunctionOp, ParamBuiltinOp>();
		
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.GROUPEDAGG, ParamBuiltinOp.GROUPEDAGG);
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.RMEMPTY, ParamBuiltinOp.RMEMPTY);
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.REPLACE, ParamBuiltinOp.REPLACE);
		
		// For order, a ReorgOp is constructed with ReorgOp.SORT type
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.ORDER, ParamBuiltinOp.INVALID);
		
		// Distribution Functions
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.CDF, ParamBuiltinOp.CDF);
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.PNORM, ParamBuiltinOp.CDF);
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.PT, ParamBuiltinOp.CDF);
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.PF, ParamBuiltinOp.CDF);
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.PCHISQ, ParamBuiltinOp.CDF);
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.PEXP, ParamBuiltinOp.CDF);
		
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.INVCDF, ParamBuiltinOp.INVCDF);
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.QNORM, ParamBuiltinOp.INVCDF);
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.QT, ParamBuiltinOp.INVCDF);
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.QF, ParamBuiltinOp.INVCDF);
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.QCHISQ, ParamBuiltinOp.INVCDF);
		pbHopMap.put(Expression.ParameterizedBuiltinFunctionOp.QEXP, ParamBuiltinOp.INVCDF);
	}
	
	public static ParameterizedBuiltinFunctionExpression getParamBuiltinFunctionExpression(String functionName, ArrayList<ParameterExpression> paramExprsPassed,
			String fileName, int blp, int bcp, int elp, int ecp){
	
		if (functionName == null || paramExprsPassed == null)
			return null;
		
		Expression.ParameterizedBuiltinFunctionOp pbifop = opcodeMap.get(functionName);
		
		if ( pbifop == null ) 
			return null;
		
		HashMap<String,Expression> varParams = new HashMap<String,Expression>();
		for (ParameterExpression pexpr : paramExprsPassed)
			varParams.put(pexpr.getName(), pexpr.getExpr());
		
		ParameterizedBuiltinFunctionExpression retVal = new ParameterizedBuiltinFunctionExpression(pbifop,varParams,
				fileName, blp, bcp, elp, ecp);
		return retVal;
	} // end method getBuiltinFunctionExpression
	
			
	public ParameterizedBuiltinFunctionExpression(ParameterizedBuiltinFunctionOp op, HashMap<String,Expression> varParams,
			String filename, int blp, int bcp, int elp, int ecp) {
		_kind = Kind.ParameterizedBuiltinFunctionOp;
		_opcode = op;
		_varParams = varParams;
		this.setAllPositions(filename, blp, bcp, elp, ecp);
	}
   
	public ParameterizedBuiltinFunctionExpression(String filename, int blp, int bcp, int elp, int ecp) {
		_kind = Kind.ParameterizedBuiltinFunctionOp;
		_opcode = ParameterizedBuiltinFunctionOp.INVALID;
		_varParams = new HashMap<String,Expression>();
		this.setAllPositions(filename, blp, bcp, elp, ecp);
	}
    
	public Expression rewriteExpression(String prefix) throws LanguageException {
		
		HashMap<String,Expression> newVarParams = new HashMap<String,Expression>();
		for (String key : _varParams.keySet()){
			Expression newExpr = _varParams.get(key).rewriteExpression(prefix);
			newVarParams.put(key, newExpr);
		}	
		ParameterizedBuiltinFunctionExpression retVal = new ParameterizedBuiltinFunctionExpression(_opcode, newVarParams,
				this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
		return retVal;
	}

	public void setOpcode(ParameterizedBuiltinFunctionOp op) {
		_opcode = op;
	}
	
	public ParameterizedBuiltinFunctionOp getOpCode() {
		return _opcode;
	}
	
	public HashMap<String,Expression> getVarParams() {
		return _varParams;
	}
	
	public Expression getVarParam(String name) {
		return _varParams.get(name);
	}

	public void addVarParam(String name, Expression value){
		_varParams.put(name, value);
	}
	
	public void removeVarParam(String name) {
		_varParams.remove(name);
	}
	
	/**
	 * Validate parse tree : Process BuiltinFunction Expression in an assignment
	 * statement
	 * 
	 * @throws LanguageException
	 */
	@Override
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars, boolean conditional)
			throws LanguageException 
	{		
		// validate all input parameters
		for ( String s : getVarParams().keySet() ) {
			Expression paramExpr = getVarParam(s);
			
			if (paramExpr instanceof FunctionCallIdentifier){
				raiseValidateError("UDF function call not supported as parameter to built-in function call", false);
			}
			
			paramExpr.validateExpression(ids, constVars, conditional);
		}
		
		String outputName = getTempName();
		DataIdentifier output = new DataIdentifier(outputName);
		//output.setProperties(this.getFirstExpr().getOutput());
		this.setOutput(output);

		// IMPORTANT: for each operation, one must handle unnamed parameters
		
		switch (this.getOpCode()) {
		
		case GROUPEDAGG:
			validateGroupedAgg(output, conditional);
			break; 
			
		case CDF:
		case INVCDF:
		case PNORM:
		case QNORM:
		case PT:
		case QT:
		case PF:
		case QF:
		case PCHISQ:
		case QCHISQ:
		case PEXP:
		case QEXP:
			validateDistributionFunctions(output, conditional);
			break;
			
		case RMEMPTY:
			validateRemoveEmpty(output, conditional);
			break;
		
		case REPLACE:
			validateReplace(output, conditional);
			break;
		
		case ORDER:
			validateOrder(output, conditional);
			break;

		case TRANSFORM:
			validateTransform(output, conditional);
			break;
			
		default: //always unconditional (because unsupported operation)
			raiseValidateError("Unsupported parameterized function "+ this.getOpCode(), false, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		return;
	}
	
	// example: A = transform(data=D, txmtd="", txspec="")
	private void validateTransform(DataIdentifier output, boolean conditional) throws LanguageException {
		Expression data = getVarParam(TF_FN_PARAM_DATA);
		if( data==null ) {				
			raiseValidateError("Named parameter '" + TF_FN_PARAM_DATA + "' missing. Please specify the input data set.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		else if( data.getOutput().getDataType() != DataType.FRAME ){
			raiseValidateError("Input to tansform() must be of type 'table'. It is of type '"+data.getOutput().getDataType()+"'.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}	
		
		Expression txmtd = getVarParam(TF_FN_PARAM_TXMTD);
		if( txmtd==null ) {
			raiseValidateError("Named parameter '" + TF_FN_PARAM_TXMTD + "' missing. Please specify the transformation metadata file path.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		else if( txmtd.getOutput().getDataType() != DataType.SCALAR || txmtd.getOutput().getValueType() != ValueType.STRING ){				
			raiseValidateError("Transformation metadata file '" + TF_FN_PARAM_TXMTD + "' must be a string value (a scalar).", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		
		Expression txspec = getVarParam(TF_FN_PARAM_TXSPEC);
		Expression applyMTD = getVarParam(TF_FN_PARAM_APPLYMTD);
		if( txspec==null ) {
			if ( applyMTD == null )
				raiseValidateError("Named parameter '" + TF_FN_PARAM_TXSPEC + "' missing. Please specify the transformation specification file (in JSON format).", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		else if( txspec.getOutput().getDataType() != DataType.SCALAR  || txspec.getOutput().getValueType() != ValueType.STRING ){	
			raiseValidateError("Transformation specification file '" + TF_FN_PARAM_TXSPEC + "' must be a string value (a scalar).", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}	
		
		if ( applyMTD != null ) {
			if( applyMTD.getOutput().getDataType() != DataType.SCALAR  || applyMTD.getOutput().getValueType() != ValueType.STRING ){	
				raiseValidateError("Apply transformation metadata file'" + TF_FN_PARAM_APPLYMTD + "' must be a string value (a scalar).", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			
			if(txspec != null ) {
				raiseValidateError("Only one of '" + TF_FN_PARAM_APPLYMTD + "' or '" + TF_FN_PARAM_TXSPEC + "' can be specified in transform().", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
		}
		
		Expression outNames = getVarParam(TF_FN_PARAM_OUTNAMES);
		if ( outNames != null ) {
			if( outNames.getOutput().getDataType() != DataType.SCALAR || outNames.getOutput().getValueType() != ValueType.STRING )				
				raiseValidateError("The parameter specifying column names in the output file '" + TF_FN_PARAM_TXMTD + "' must be a string value (a scalar).", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			if ( applyMTD != null)
				raiseValidateError("Only one of '" + TF_FN_PARAM_APPLYMTD + "' or '" + TF_FN_PARAM_OUTNAMES + "' can be specified in transform().", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		
		// Output is a matrix with same dims as input
		output.setDataType(DataType.MATRIX);
		output.setFormatType(FormatType.CSV);
		output.setValueType(ValueType.DOUBLE);
		// Output dimensions may not be known at compile time, for example when dummycoding.
		output.setDimensions(-1, -1);
	}
	
	private void validateReplace(DataIdentifier output, boolean conditional) throws LanguageException {
		//check existence and correctness of arguments
		Expression target = getVarParam("target");
		if( target==null ) {				
			raiseValidateError("Named parameter 'target' missing. Please specify the input matrix.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		else if( target.getOutput().getDataType() != DataType.MATRIX ){
			raiseValidateError("Input matrix 'target' is of type '"+target.getOutput().getDataType()+"'. Please specify the input matrix.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}	
		
		Expression pattern = getVarParam("pattern");
		if( pattern==null ) {
			raiseValidateError("Named parameter 'pattern' missing. Please specify the replacement pattern.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		else if( pattern.getOutput().getDataType() != DataType.SCALAR ){				
			raiseValidateError("Replacement pattern 'pattern' is of type '"+pattern.getOutput().getDataType()+"'. Please, specify a scalar replacement pattern.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}	
		
		Expression replacement = getVarParam("replacement");
		if( replacement==null ) {
			raiseValidateError("Named parameter 'replacement' missing. Please specify the replacement value.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		else if( replacement.getOutput().getDataType() != DataType.SCALAR ){	
			raiseValidateError("Replacement value 'replacement' is of type '"+replacement.getOutput().getDataType()+"'. Please, specify a scalar replacement value.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}	
		
		// Output is a matrix with same dims as input
		output.setDataType(DataType.MATRIX);
		output.setValueType(ValueType.DOUBLE);
		output.setDimensions(target.getOutput().getDim1(), target.getOutput().getDim2());
	}

	private void validateOrder(DataIdentifier output, boolean conditional) throws LanguageException {
		//check existence and correctness of arguments
		Expression target = getVarParam("target"); //[MANDATORY] TARGET
		if( target==null ) {				
			raiseValidateError("Named parameter 'target' missing. Please specify the input matrix.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		else if( target.getOutput().getDataType() != DataType.MATRIX ){
			raiseValidateError("Input matrix 'target' is of type '"+target.getOutput().getDataType()+"'. Please specify the input matrix.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}	
		
		//check for unsupported parameters
		for(String param : getVarParams().keySet())
			if( !(param.equals("target") || param.equals("by") || param.equals("decreasing") || param.equals("index.return")) )
				raiseValidateError("Unsupported order parameter: '"+param+"'", false);
		
		Expression orderby = getVarParam("by"); //[OPTIONAL] BY
		if( orderby == null ) { //default first column, good fit for vectors
			orderby = new IntIdentifier(1, "1", -1, -1, -1, -1);
			addVarParam("by", orderby);
		}
		else if( orderby !=null && orderby.getOutput().getDataType() != DataType.SCALAR ){				
			raiseValidateError("Orderby column 'by' is of type '"+orderby.getOutput().getDataType()+"'. Please, specify a scalar order by column index.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}	
		
		Expression decreasing = getVarParam("decreasing"); //[OPTIONAL] DECREASING
		if( decreasing == null ) { //default: ascending
			addVarParam("decreasing", new BooleanIdentifier(false, "false", -1, -1, -1, -1));
		}
		else if( decreasing!=null && decreasing.getOutput().getDataType() != DataType.SCALAR ){				
			raiseValidateError("Ordering 'decreasing' is of type '"+decreasing.getOutput().getDataType()+"', '"+decreasing.getOutput().getValueType()+"'. Please, specify 'decreasing' as a scalar boolean.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		
		Expression indexreturn = getVarParam("index.return"); //[OPTIONAL] DECREASING
		if( indexreturn == null ) { //default: sorted data
			indexreturn = new BooleanIdentifier(false, "false", -1, -1, -1, -1);
			addVarParam("index.return", indexreturn);
		}
		else if( indexreturn!=null && indexreturn.getOutput().getDataType() != DataType.SCALAR ){				
			raiseValidateError("Return type 'index.return' is of type '"+indexreturn.getOutput().getDataType()+"', '"+indexreturn.getOutput().getValueType()+"'. Please, specify 'indexreturn' as a scalar boolean.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		long dim2 = ( indexreturn instanceof BooleanIdentifier ) ? 
				((BooleanIdentifier)indexreturn).getValue() ? 1: target.getOutput().getDim2() : -1; 
		
		// Output is a matrix with same dims as input
		output.setDataType(DataType.MATRIX);
		output.setValueType(ValueType.DOUBLE);
		output.setDimensions(target.getOutput().getDim1(), dim2 );
		
	}

	private void validateRemoveEmpty(DataIdentifier output, boolean conditional) throws LanguageException {
		//check existence and correctness of arguments
		Expression target = getVarParam("target");
		if( target==null ) {
			raiseValidateError("Named parameter 'target' missing. Please specify the input matrix.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		else if( target.getOutput().getDataType() != DataType.MATRIX ){
			raiseValidateError("Input matrix 'target' is of type '"+target.getOutput().getDataType()+"'. Please specify the input matrix.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		
		Expression margin = getVarParam("margin");
		if( margin==null ){
			raiseValidateError("Named parameter 'margin' missing. Please specify 'rows' or 'cols'.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		else if( !(margin instanceof DataIdentifier) && !margin.toString().equals("rows") && !margin.toString().equals("cols") ){
			raiseValidateError("Named parameter 'margin' has an invalid value '"+margin.toString()+"'. Please specify 'rows' or 'cols'.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		
		// Output is a matrix with unknown dims
		output.setDataType(DataType.MATRIX);
		output.setValueType(ValueType.DOUBLE);
		output.setDimensions(-1, -1);
		
	}
	
	private void validateGroupedAgg(DataIdentifier output, boolean conditional) throws LanguageException {
		int colwise = -1;
		if (getVarParam(Statement.GAGG_TARGET)  == null || getVarParam(Statement.GAGG_GROUPS) == null){
			raiseValidateError("Must define both target and groups and both must have same dimensions", conditional);
		}
		if (getVarParam(Statement.GAGG_TARGET) instanceof DataIdentifier && getVarParam(Statement.GAGG_GROUPS) instanceof DataIdentifier && (getVarParam(Statement.GAGG_WEIGHTS) == null || getVarParam(Statement.GAGG_WEIGHTS) instanceof DataIdentifier))
		{
			
			DataIdentifier targetid = (DataIdentifier)getVarParam(Statement.GAGG_TARGET);
			DataIdentifier groupsid = (DataIdentifier)getVarParam(Statement.GAGG_GROUPS);
			DataIdentifier weightsid = (DataIdentifier)getVarParam(Statement.GAGG_WEIGHTS);
		
			if ( targetid.dimsKnown() ) {
				colwise = targetid.getDim1() > targetid.getDim2() ? 1 : 0;
			}
			else if ( groupsid.dimsKnown() ) {
				colwise = groupsid.getDim1() > groupsid.getDim2() ? 1 : 0;
			}
			else if ( weightsid != null && weightsid.dimsKnown() ) {
				colwise = weightsid.getDim1() > weightsid.getDim2() ? 1 : 0;
			}
			
			//precompute number of rows and columns because target can be row or column vector
			long rowsTarget = targetid.getDim1(); // Math.max(targetid.getDim1(),targetid.getDim2());
			long colsTarget = targetid.getDim2(); // Math.min(targetid.getDim1(),targetid.getDim2());
			
			if( targetid.dimsKnown() && groupsid.dimsKnown() &&
				(rowsTarget != groupsid.getDim1() || colsTarget != groupsid.getDim2()) )
			{					
				raiseValidateError("target and groups must have same dimensions -- " 
						+ " targetid dims: " + targetid.getDim1() +" rows, " + targetid.getDim2() + " cols -- groupsid dims: " + groupsid.getDim1() + " rows, " + groupsid.getDim2() + " cols ", conditional);
			}
			
			if( weightsid != null && (targetid.dimsKnown() && weightsid.dimsKnown()) &&
				(rowsTarget != weightsid.getDim1() || colsTarget != weightsid.getDim2() ))
			{		
				raiseValidateError("target and weights must have same dimensions -- "
						+ " targetid dims: " + targetid.getDim1() +" rows, " + targetid.getDim2() + " cols -- weightsid dims: " + weightsid.getDim1() + " rows, " + weightsid.getDim2() + " cols ", conditional);
			}
		}
		
		
		if (getVarParam(Statement.GAGG_FN) == null){
			raiseValidateError("must define function name (fn=<function name>) for aggregate()", conditional);
		}
		
		Expression functParam = getVarParam(Statement.GAGG_FN);
		
		if (functParam instanceof Identifier)
		{
			// standardize to lowercase and dequote fname
			String fnameStr = getVarParam(Statement.GAGG_FN).toString();
			
			
			// check that IF fname="centralmoment" THEN order=m is defined, where m=2,3,4 
			// check ELSE IF fname is allowed
			if(fnameStr.equals(Statement.GAGG_FN_CM)){
				String orderStr = getVarParam(Statement.GAGG_FN_CM_ORDER) == null ? null : getVarParam(Statement.GAGG_FN_CM_ORDER).toString();
				if (orderStr == null || !(orderStr.equals("2") || orderStr.equals("3") || orderStr.equals("4"))){
					raiseValidateError("for centralmoment, must define order.  Order must be equal to 2,3, or 4", conditional);
				}
			}
			else if (fnameStr.equals(Statement.GAGG_FN_COUNT) 
					|| fnameStr.equals(Statement.GAGG_FN_SUM) 
					|| fnameStr.equals(Statement.GAGG_FN_MEAN)
					|| fnameStr.equals(Statement.GAGG_FN_VARIANCE)){}
			else { 
				raiseValidateError("fname is " + fnameStr + " but must be either centeralmoment, count, sum, mean, variance", conditional);
			}
		}
		
		Expression ngroupsParam = getVarParam(Statement.GAGG_NUM_GROUPS);
		long outputDim1 = -1, outputDim2 = -1;
		if( ngroupsParam != null && ngroupsParam instanceof Identifier ) 
		{
			Identifier numGroups = (Identifier) ngroupsParam;
			if ( numGroups != null && numGroups instanceof ConstIdentifier) {
				long ngroups = ((ConstIdentifier)numGroups).getLongValue();
				if ( colwise == 1 ) {
					outputDim1 = ngroups;
					outputDim2 = 1;
				}
				else if ( colwise == 0 ) {
					outputDim1 = 1;
					outputDim2 = ngroups;
				}
			}
		}
		
		// Output is a matrix with unknown dims
		output.setDataType(DataType.MATRIX);
		output.setValueType(ValueType.DOUBLE);
		output.setDimensions(outputDim1, outputDim2);
	}
	
	private void validateDistributionFunctions(DataIdentifier output, boolean conditional) throws LanguageException {
		// CDF and INVCDF expects one unnamed parameter, it must be renamed as "quantile" 
		// (i.e., we must compute P(X <= x) where x is called as "quantile" )
		
		ParameterizedBuiltinFunctionOp op = this.getOpCode();
		
		// check if quantile is of type SCALAR
		if ( getVarParam("target") == null || getVarParam("target").getOutput().getDataType() != DataType.SCALAR ) {
			raiseValidateError("target must be provided for distribution functions, and it must be a scalar value.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		
		// Distribution specific checks
		switch(op) {
		case CDF:
		case INVCDF:
			if(getVarParam("dist") == null) {
				raiseValidateError("For cdf() and icdf(), a distribution function must be specified (as a string).", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			break;
			
		case QF:
		case PF:
			if(getVarParam("df1") == null || getVarParam("df2") == null ) {
				raiseValidateError("Two degrees of freedom df1 and df2 must be provided for F-distribution.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			break;
			
		case QT:
		case PT:
			if(getVarParam("df") == null ) {
				raiseValidateError("Degrees of freedom df must be provided for t-distribution.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			break;
			
		case QCHISQ:
		case PCHISQ:
			if(getVarParam("df") == null ) {
				raiseValidateError("Degrees of freedom df must be provided for chi-squared-distribution.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			break;
			
			default:
				break;
				
			// Not checking for QNORM, PNORM: distribution parameters mean and sd are optional with default values 0.0 and 1.0, respectively
			// Not checking for QEXP, PEXP: distribution parameter rate is optional with a default values 1.0
			
			// For all cdf functions, additional parameter lower.tail is optional with a default value TRUE
		}
		
		// CDF and INVCDF specific checks:
		switch(op) {
		case INVCDF:
		case QNORM:
		case QF:
		case QT:
		case QCHISQ:
		case QEXP:
			if(getVarParam("lower.tail") != null ) {
				raiseValidateError("Lower tail argument is invalid while computing inverse cumulative probabilities.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			break;
			
		case CDF:
		case PNORM:
		case PF:
		case PT:
		case PCHISQ:
		case PEXP:
			// no checks yet
			break;
			
			default:
				break;
		}
		
		// Output is a scalar
		output.setDataType(DataType.SCALAR);
		output.setValueType(ValueType.DOUBLE);
		output.setDimensions(0, 0);
		return;
	}

	public String toString() {
		StringBuilder sb = new StringBuilder(_opcode.toString() + "(");

		 for (String key : _varParams.keySet()){
			 sb.append("," + key + "=" + _varParams.get(key));
		 }
		sb.append(" )");
		return sb.toString();
	}

	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		for (String s : _varParams.keySet()) {
			result.addVariables ( _varParams.get(s).variablesRead() );
		}
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		for (String s : _varParams.keySet()) {
			result.addVariables ( _varParams.get(s).variablesUpdated() );
		}
		result.addVariable(((DataIdentifier)this.getOutput()).getName(), (DataIdentifier)this.getOutput());
		return result;
	}

	@Override
	public boolean multipleReturns() {
		return false;
	}
}
