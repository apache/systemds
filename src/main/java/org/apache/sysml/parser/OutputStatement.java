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

package org.apache.sysml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.debug.DMLBreakpointManager;
import org.apache.sysml.parser.Expression.DataOp;

 
public class OutputStatement extends Statement
{
		
	private DataIdentifier _id;
	private DataExpression _paramsExpr;
	
	public static final String[] WRITE_VALID_PARAM_NAMES = { 	DataExpression.IO_FILENAME, 
																DataExpression.FORMAT_TYPE, 
																DataExpression.DELIM_DELIMITER, 
																DataExpression.DELIM_HAS_HEADER_ROW, 
																DataExpression.DELIM_SPARSE};

	public DataIdentifier getIdentifier(){
		return _id;
	}
	
	public DataExpression getSource(){
		return _paramsExpr;
	}
	
	public void setIdentifier(DataIdentifier t) {
		_id = t;
	}
	
	public OutputStatement(DataIdentifier t, DataOp op, 
			String filename, int blp, int bcp, int elp, int ecp){
		_id = t;
		_paramsExpr = new DataExpression(op, new HashMap<String,Expression>(),
				filename, blp, bcp, elp, ecp);
	}
	
	/**
	 * Called by the parser (both javacc and antlr).
	 * 
	 * @param fname
	 * @param fci
	 * @param filename
	 * @param blp
	 * @param bcp
	 * @param elp
	 * @param ecp
	 * @throws DMLParseException
	 */
	OutputStatement(String fname, FunctionCallIdentifier fci, 
			String filename, int blp, int bcp, int elp, int ecp) 
		throws DMLParseException 
	{
		
		this.setAllPositions(filename, blp, bcp, elp, ecp);
		DataOp op = Expression.DataOp.WRITE;
		ArrayList<ParameterExpression> passedExprs = fci.getParamExprs();
		_paramsExpr = new DataExpression(op, new HashMap<String,Expression>(),
				filename, blp, bcp, elp, ecp);
		DMLParseException runningList = new DMLParseException(fname);
		
		//check number parameters and proceed only if this will not cause errors
		if (passedExprs.size() < 2)
			runningList.add(new DMLParseException(fci.getFilename(), fci.printErrorLocation() + "write method must specify both variable to write to file, and filename to write variable to"));
		else
		{
			ParameterExpression firstParam = passedExprs.get(0);
			if (firstParam.getName() != null || (!(firstParam.getExpr() instanceof DataIdentifier)))
				runningList.add(new DMLParseException(fci.getFilename(), fci.printErrorLocation() + "first argument to write method must be name of variable to be written out"));
			else
				_id = (DataIdentifier)firstParam.getExpr();
			
			ParameterExpression secondParam = passedExprs.get(1);
			if (secondParam.getName() != null || (secondParam.getName() != null && secondParam.getName().equals(DataExpression.IO_FILENAME)))
				runningList.add(new DMLParseException(fci.getFilename(), fci.printErrorLocation() + "second argument to write method must be filename of file variable written to"));
			else
				addExprParam(DataExpression.IO_FILENAME, secondParam.getExpr(), false);
				
			for (int i = 2; i< passedExprs.size(); i++){
				ParameterExpression currParam = passedExprs.get(i);
				try {
					addExprParam(currParam.getName(), currParam.getExpr(), false);
				} catch (DMLParseException e){
					runningList.add(e);
				}
			}
			if (fname.equals("writeMM")){
				StringIdentifier writeMMExpr = new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_MATRIXMARKET,
						this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
				addExprParam(DataExpression.FORMAT_TYPE, writeMMExpr, false);
			}
			else if (fname.equals("write.csv")){
				StringIdentifier delimitedExpr = new StringIdentifier(DataExpression.FORMAT_TYPE_VALUE_CSV,
						this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
				addExprParam(DataExpression.FORMAT_TYPE, delimitedExpr, false);
			}
		}
		
		if (runningList.size() > 0)
			throw runningList;
	}
	
	public void setExprParam(String name, Expression value) {
		_paramsExpr.addVarParam(name, value);
	}
	
	public static boolean isValidParamName(String key){
		for (String paramName : WRITE_VALID_PARAM_NAMES)
			if (paramName.equals(key))
				return true;
			return false;
	}
	
	public void addExprParam(String name, Expression value, boolean fromMTDFile) throws DMLParseException
	{
		DMLParseException runningList = new DMLParseException(value.getFilename());
		
		if( _paramsExpr.getVarParam(name) != null )
			runningList.add(new DMLParseException(value.getFilename(), value.printErrorLocation() + "attempted to add IOStatement parameter " + name + " more than once"));
		
		if( !OutputStatement.isValidParamName(name) )
			runningList.add(new DMLParseException(value.getFilename(), value.printErrorLocation() + "attempted to add invalid write statement parameter: " + name));
		
		_paramsExpr.addVarParam(name, value);
		
		if (runningList.size() > 0)
			throw runningList;
	}
	
	// rewrites statement to support function inlining (create deep copy)
	public Statement rewriteStatement(String prefix) throws LanguageException{
		
		OutputStatement newStatement = new OutputStatement(null,Expression.DataOp.WRITE,
				this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		
		// rewrite outputStatement variable name (creates deep copy)
		newStatement._id = (DataIdentifier)this._id.rewriteExpression(prefix);
		
		// rewrite parameter expressions (creates deep copy)
		DataOp op = _paramsExpr.getOpCode();
		HashMap<String,Expression> newExprParams = new HashMap<String,Expression>();
		for (String key : _paramsExpr.getVarParams().keySet()){
			Expression newExpr = _paramsExpr.getVarParam(key).rewriteExpression(prefix);
			newExprParams.put(key, newExpr);
		}
		DataExpression newParamerizedExpr = new DataExpression(op, newExprParams,
				this.getFilename(), this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		newStatement.setExprParams(newParamerizedExpr);
		return newStatement;
	}
		
	public void setExprParams(DataExpression newParamerizedExpr) {
		_paramsExpr = newParamerizedExpr;
	}
	public Expression getExprParam(String key){
		return _paramsExpr.getVarParam(key);
	}
	
	public void initializeforwardLV(VariableSet activeIn){}
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	public String toString(){
		StringBuilder sb = new StringBuilder();
		 sb.append(Statement.OUTPUTSTATEMENT + " ( " );
		 sb.append( _id.toString() + ", " +  _paramsExpr.getVarParam(DataExpression.IO_FILENAME).toString());
		 for (String key : _paramsExpr.getVarParams().keySet()){
			 if (!key.equals(DataExpression.IO_FILENAME))
				 sb.append(", " + key + "=" + _paramsExpr.getVarParam(key));
		 }
		 sb.append(" );");
		 return sb.toString(); 
	}
	
	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		
		// handle variable that is being written out
		result.addVariables(_id.variablesRead());
		
		// handle variables for output filename expression
		//result.addVariables(_filenameExpr.variablesRead());
		
		// add variables for parameter expressions 
		for (String key : _paramsExpr.getVarParams().keySet())
			result.addVariables(_paramsExpr.getVarParam(key).variablesRead()) ;
		
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
	  	return null;
	}
	
	@Override
	public boolean controlStatement() {
		// ensure that breakpoints end up in own statement block 
		if (DMLScript.ENABLE_DEBUG_MODE) {
			DMLBreakpointManager.insertBreakpoint(_paramsExpr.getBeginLine());
			return true;
		}

		Expression fmt = _paramsExpr.getVarParam(DataExpression.FORMAT_TYPE);
		if ( fmt != null && fmt.toString().equalsIgnoreCase("csv")) {
			return true;
		}
		return false;
	}
}
