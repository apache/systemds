package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.utils.LanguageException;


public class CVStatementBlock extends StatementBlock {
	
	public String toString() {
		StringBuffer sb = new StringBuffer();
		CVStatement statement = (CVStatement) getStatement(0) ;
		sb.append(statement);
		return sb.toString();
	}
	
	public boolean mergeable(){
		return false;
	}
		
	@Override
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String, ConstIdentifier> constVars) throws LanguageException {

		if (this.getNumStatements() > 1){
			LOG.error(this.printBlockErrorLocation() + "CV statement block can only have single statement");
			throw new LanguageException(this.printBlockErrorLocation() + "CV statement block can only have single statement");
		}
		CVStatement cvs = (CVStatement) this.getStatement(0);
		
		// check the input datasets are available
		for (Expression inputExpr : cvs.get_inputs().getParamExpressions()){
			inputExpr.validateExpression(ids.getVariables(),constVars);
		}	
		
		// build list of partition outputs
		// NOTE: assume each partition output is DataIdentifier
		VariableSet partitionOutputList = new VariableSet();
		for (Expression output : cvs.get_partitionOutputs().getParamExpressions()){
			partitionOutputList.addVariable( ((DataIdentifier)output).getName(),(DataIdentifier)output);
		}
		
		// check train function exists
		FunctionCallIdentifier trainFCI = cvs.get_trainFunctionCall();
		if (!dmlProg.getFunctionStatementBlocks(trainFCI.getNamespace()).containsKey(trainFCI.getName())){
			LOG.error(this.printBlockErrorLocation() + "CV training function " + trainFCI.getNamespace() + "::" + trainFCI.getName() + " is not available ");
			throw new LanguageException(this.printBlockErrorLocation() + "CV training function " + trainFCI.getNamespace() + "::" + trainFCI.getName() + " is not available ");
		}
		
		// check train function inputs are available from either:
		//	1) already defined variables
		//	2) partition function outputs
		for (Expression input : trainFCI.getParamExpressions()){
					
			if (input instanceof DataIdentifier){
				boolean found = false;
				
				if (ids.containsVariable(((DataIdentifier) input).getName()))
					found = true;
				if (partitionOutputList.containsVariable(((DataIdentifier) input).getName()))
					found = true;
			
				if (found == false){
					LOG.error(this.printBlockErrorLocation() + "In CV Statement, variable " + input.toString() + " is not available");
					throw new LanguageException(this.printBlockErrorLocation() + "In CV Statement, variable " + input.toString() + " is not available");
				}
			}
			else if (input instanceof ConstIdentifier) {
				input.validateExpression(ids.getVariables(), constVars);
			}
			else {
				LOG.error(this.printBlockErrorLocation() + "CV training function " + trainFCI.getNamespace() + "::" + trainFCI.getName() + " has unsupported formal parameter: " + input.toString());
				throw new LanguageException(this.printBlockErrorLocation() + "CV training function " + trainFCI.getNamespace() + "::" + trainFCI.getName() + " has unsupported formal parameter: " + input.toString());
			}
		}
		
		// build list of training function outputs
		// NOTE: assume each training function output is DataIdentifier
		VariableSet trainOutputList = new VariableSet();
		for (Expression output : cvs.get_trainFunctionOutputs().getParamExpressions()){
			trainOutputList.addVariable( ((DataIdentifier)output).getName(),(DataIdentifier)output);
		}
		
		// check test function exists
		FunctionCallIdentifier testFCI = cvs.get_testFunctionCall();
		if (!dmlProg.getFunctionStatementBlocks(testFCI.getNamespace()).containsKey(testFCI.getName())){
			LOG.error(this.printBlockErrorLocation() + "CV training function " + testFCI.getNamespace() + "::" + testFCI.getName() + " is not available ");
			throw new LanguageException(this.printBlockErrorLocation() + "CV training function " + testFCI.getNamespace() + "::" + testFCI.getName() + " is not available ");
		}
		
		// check test function inputs are available from either:
		//	1) already defined variables
		//	2) partition function outputs
		//	3) training function outputs
		for (Expression input : testFCI.getParamExpressions()){
					
			if (input instanceof DataIdentifier){
				boolean found = false;
				if (ids.containsVariable(((DataIdentifier) input).getName()))
					found = true;
				if (partitionOutputList.containsVariable(((DataIdentifier) input).getName()))
					found = true;
				if (trainOutputList.containsVariable(((DataIdentifier) input).getName()))
					found = true;
				
				if (found == false){
					LOG.error(this.printBlockErrorLocation() + "In CV Statement, variable " + input.toString() + " is not available");
					throw new LanguageException(this.printBlockErrorLocation() + "In CV Statement, variable " + input.toString() + " is not available");
				}
			}
			else if (input instanceof ConstIdentifier){
				// handles constants
				input.validateExpression(ids.getVariables(), constVars);
			}
			else {
				LOG.error(this.printBlockErrorLocation() + "CV training function " + testFCI.getNamespace() + "::" + testFCI.getName() + " has unsupported formal parameter: " + input.toString());
				throw new LanguageException(this.printBlockErrorLocation() + "CV training function " + testFCI.getNamespace() + "::" + testFCI.getName() + " has unsupported formal parameter: " + input.toString());
			}
		}
		
		// check test function exists
		Identifier aggFCI = cvs.get_aggFunctionCall();
		if (aggFCI instanceof FunctionCallIdentifier){
			if (!dmlProg.getFunctionStatementBlocks(testFCI.getNamespace()).containsKey(testFCI.getName())){
				LOG.error(this.printBlockErrorLocation() + "CV training function " + testFCI.getNamespace() + "::" + testFCI.getName() + " is not available ");
				throw new LanguageException(this.printBlockErrorLocation() + "CV training function " + testFCI.getNamespace() + "::" + testFCI.getName() + " is not available ");
			}
		}
		
		// check test function inputs are available from either:
		//	1) already defined variables
		//	2) partition function outputs
		//	3) training function outputs
		
		ArrayList<Expression> aggFuncExprs = new ArrayList<Expression>();
		if (aggFCI instanceof FunctionCallIdentifier){
			aggFuncExprs = ((FunctionCallIdentifier)aggFCI).getParamExpressions();
		}
		else if (aggFCI instanceof BuiltinFunctionExpression){
			if ( ((BuiltinFunctionExpression)aggFCI).getFirstExpr() != null )
				aggFuncExprs.add(((BuiltinFunctionExpression)aggFCI).getFirstExpr());
			if ( ((BuiltinFunctionExpression)aggFCI).getSecondExpr() != null )
				aggFuncExprs.add(((BuiltinFunctionExpression)aggFCI).getSecondExpr());
			if ( ((BuiltinFunctionExpression)aggFCI).getThirdExpr() != null )
				aggFuncExprs.add(((BuiltinFunctionExpression)aggFCI).getThirdExpr());
		}
		else {
			LOG.error(this.printBlockErrorLocation() + "CV aggregation function " + aggFCI.toString() + " is unsupported method type");
			throw new LanguageException(this.printBlockErrorLocation() + "CV aggregation function " + aggFCI.toString() + " is unsupported method type");
		}
		
		// build list of test function outputs
		// NOTE: assume each test function output is DataIdentifier
		VariableSet testOutputList = new VariableSet();
		for (Expression output : cvs.get_testFunctionOutputs().getParamExpressions()){
			trainOutputList.addVariable( ((DataIdentifier)output).getName(),(DataIdentifier)output);
		}
		
		
		for (Expression input : aggFuncExprs) {
					
			if (input instanceof DataIdentifier){
				boolean found = false;
				if (ids.containsVariable(((DataIdentifier) input).getName()))
					found = true;
				if (partitionOutputList.containsVariable(((DataIdentifier) input).getName()))
					found = true;
				if (trainOutputList.containsVariable(((DataIdentifier) input).getName()))
					found = true;
				if (testOutputList.containsVariable(((DataIdentifier) input).getName()))
					found = true;
				
				if (found == false){
					LOG.error(this.printBlockErrorLocation() + "In CV Statement test function, variable " + input.toString() + " is not available");
					throw new LanguageException(this.printBlockErrorLocation() + "In CV Statement test function, variable " + input.toString() + " is not available");
				}
			}
			else if (input instanceof ConstIdentifier){
				// handles constants
				input.validateExpression(ids.getVariables(), constVars);
			}
			else {
				LOG.error(this.printBlockErrorLocation() + "CV training function " + testFCI.getNamespace() + "::" + testFCI.getName() + " has unsupported formal parameter: " + input.toString());
				throw new LanguageException(this.printBlockErrorLocation() + "CV training function " + testFCI.getNamespace() + "::" + testFCI.getName() + " has unsupported formal parameter: " + input.toString());
			}
		}
		
		// add error aggregate output to returned variables 
		for (Expression aggOutput : cvs.get_aggFunctionOutputs().getParamExpressions()){
			ids.addVariable(((DataIdentifier)aggOutput).getName(), new DataIdentifier((DataIdentifier)aggOutput));
		}
		
		return ids;	
	}
	
	public ArrayList<Hops> get_hops() {
		
//		if (this._hops == null){
//			
//			CVStatement statement = (CVStatement) getStatement(0) ;
//			
//			// read each input file to be partitioned
//			ArrayList<Hops> cvHops = new ArrayList<Hops>() ;
//			for (String input : pp.getInputs()){
//				
//				DataIdentifier var = this.liveIn().getVariables().get(input);
//				
//				long actualDim1 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim1() : var.getDim1();
//				long actualDim2 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim2() : var.getDim2();
//				
//				DataOp read = new DataOp(var.getName(), var.getDataType(), var.getValueType(), 
//					DataOpTypes.TRANSIENTREAD, null, actualDim1, actualDim2,var.getNnz(), 
//					var.getRowsInBlock(), var.getColumnsInBlock());
//				read.setAllPositions(var.getBeginLine(), var.getBeginColumn(), var.getEndLine(), var.getEndColumn());
//				read.set_rows_in_block( var.getRowsInBlock()) ;
//				read.set_cols_in_block( var.getColumnsInBlock()) ;
//			
//				// create the partition HOP
//				partitionHop = new PartitionOp("B", DataType.MATRIX, ValueType.DOUBLE, pp, read) ;
//				partitionHop.setAllPositions(statement.getBeginLine(), statement.getBeginColumn(), statement.getEndLine(), statement.getEndColumn());
//			
//				// create the CV HOP
//				//crossvalHop = new CrossvalOp("CVHop", DataType.MATRIX, ValueType.DOUBLE, partitionHop, params) ;
//			
//				cvHops.add(partitionHop) ;
//				set_hops(cvHops) ;
//			}
//		} // if (this._hops == null)
		
		return this._hops;
		
	}
}