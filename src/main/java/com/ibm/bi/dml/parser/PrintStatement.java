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

package com.ibm.bi.dml.parser;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.debug.DMLBreakpointManager;

 
public class PrintStatement extends Statement
{
	public enum PRINTTYPE {PRINT, STOP};
	
	protected PRINTTYPE _type; // print or stop
	protected Expression _expr;

	private static PRINTTYPE getPrintType(String type) throws LanguageException {
		if(type.equalsIgnoreCase("print")) {
			return PRINTTYPE.PRINT;
		}
		else if (type.equalsIgnoreCase("stop")) {
			return PRINTTYPE.STOP;
		}
		else
			throw new LanguageException("Unknown statement type: " + type);
	}
	
	public PrintStatement(String type, Expression expr, int beginLine, int beginCol, int endLine, int endCol) 
		throws LanguageException
	{
		this(getPrintType(type), expr);
		
		setBeginLine(beginLine);
		setBeginColumn(beginCol);
		setEndLine(endLine);
		setEndColumn(endCol);
	}
	 
	public PrintStatement(PRINTTYPE type, Expression expr) throws LanguageException{
		_type = type;
		_expr = expr; 
	}
	 
	public Statement rewriteStatement(String prefix) throws LanguageException{
		Expression newExpr = _expr.rewriteExpression(prefix);
		PrintStatement retVal = new PrintStatement(_type, newExpr);
		retVal.setBeginLine(this.getBeginLine());
		retVal.setBeginColumn(this.getBeginColumn());
		retVal.setEndLine(this.getEndLine());
		retVal.setEndColumn(this.getEndColumn());
		
		return retVal;
	}
	
	public void initializeforwardLV(VariableSet activeIn){}
	
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	
	public String toString(){
		 StringBuilder sb = new StringBuilder();
		 sb.append(_type + " (" );
		 if (_expr != null){
			 sb.append(_expr.toString());
		 }
		 sb.append(");");
		 return sb.toString(); 
	}
	
	
	@Override
	public VariableSet variablesRead() {
		VariableSet result =  _expr.variablesRead();
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
	  	return new VariableSet();
	}

	@Override
	public boolean controlStatement() {	 
		// ensure that breakpoints end up in own statement block
		if (DMLScript.ENABLE_DEBUG_MODE) {
			DMLBreakpointManager.insertBreakpoint(_expr.getBeginLine());
			return true;
		}
		
		// Keep stop() statement in a separate statement block
		if(getType() == PRINTTYPE.STOP)
			return true;
		
		return false;
	}

	public Expression getExpression(){
		return _expr;
	}
	
	public PRINTTYPE getType() {
		return _type;
	}
	 
}
