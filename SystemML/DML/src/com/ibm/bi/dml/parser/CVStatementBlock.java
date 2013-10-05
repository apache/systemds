/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.Hop;


public class CVStatementBlock extends StatementBlock 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
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
		// NOTE: assume each partition output is DataIdentifier with datatype matrix (and valuetype double)
		VariableSet partitionOutputList = new VariableSet();
		for (Expression output : cvs.get_partitionOutputs().getParamExpressions()){
			try {
				((DataIdentifier)output).setTypeInfo("double", "matrix");
			} catch (ParseException e){
				throw new LanguageException(e);
			}
			partitionOutputList.addVariable( ((DataIdentifier)output).getName(),(DataIdentifier)output);
			
		}
		
		// check train function exists
		FunctionCallIdentifier trainFCI = cvs.get_trainFunctionCall();
		HashMap<String, FunctionStatementBlock> tempNS = dmlProg.getFunctionStatementBlocks(trainFCI.getNamespace());
		if (tempNS == null){
			LOG.error(this.printBlockErrorLocation() + "Namespace " + trainFCI.getNamespace() + " is undefined ");
			throw new LanguageException(this.printBlockErrorLocation() + "Namespace " + trainFCI.getNamespace() + " is undefined ");
		}
		FunctionStatementBlock trainFSB = tempNS.get(trainFCI.getName());
		if (trainFSB == null){
			LOG.error(this.printBlockErrorLocation() + "CV training function " + trainFCI.getNamespace() + "::" + trainFCI.getName() + " is not available ");
			throw new LanguageException(this.printBlockErrorLocation() + "CV training function " + trainFCI.getNamespace() + "::" + trainFCI.getName() + " is not available ");
		}
		
		// check train function has correct number of parameters AND
		// train function inputs are available from either:
		//	 1) already defined variables
		//	 2) partition function outputs
		VariableSet tempIds = new VariableSet();
		tempIds.addVariables(ids);
		tempIds.addVariables(partitionOutputList);
		try {
			trainFCI.validateExpression(dmlProg,tempIds.getVariables(), constVars);
		} catch(IOException e){
			throw new LanguageException(e);
		}
		
		// build list of training function outputs
		// check training function call has correct number of parameters		
		FunctionStatement trainFS = (FunctionStatement)trainFSB.getStatement(0);
		if (cvs.get_trainFunctionOutputs().getParamExpressions().size() != trainFS.getOutputParams().size()){
			LOG.error(this.printBlockErrorLocation() + "CV training function " + trainFCI.getNamespace() + "::" + trainFCI.getName() + " has wrong number parameters ");
			throw new LanguageException(this.printBlockErrorLocation() + "CV training function " + trainFCI.getNamespace() + "::" + trainFCI.getName() + " has wrong number parameters ");

		}
		
		// for each train function parameter, set datatype and valuetype
		VariableSet trainOutputList = new VariableSet();
		for (int i = 0; i< cvs.get_trainFunctionOutputs().getParamExpressions().size(); i++){
			DataIdentifier output = (DataIdentifier)cvs.get_trainFunctionOutputs().getParamExpressions().get(i);
			try {
				output.setTypeInfo(trainFS.getOutputParams().get(i).getValueType().toString(), trainFS.getOutputParams().get(i).getDataType().toString());
			} catch (ParseException e) {
				throw new LanguageException(e);
			}
			trainOutputList.addVariable( ((DataIdentifier)output).getName(),(DataIdentifier)output);
		}
		
		////////////////////////////////////////////////////////////////////////
		// validate test function
		////////////////////////////////////////////////////////////////////////
		
		// check test function exists
		FunctionCallIdentifier testFCI = cvs.get_testFunctionCall();
		tempNS = dmlProg.getFunctionStatementBlocks(testFCI.getNamespace());
		if (tempNS == null){
			LOG.error(this.printBlockErrorLocation() + "Namespace " + testFCI.getNamespace() + " is undefined ");
			throw new LanguageException(this.printBlockErrorLocation() + "Namespace " + testFCI.getNamespace() + " is undefined ");
		}
		FunctionStatementBlock testFSB = tempNS.get(testFCI.getName());
		if (testFSB == null){
			LOG.error(this.printBlockErrorLocation() + "CV training function " + testFCI.getNamespace() + "::" + testFCI.getName() + " is not available ");
			throw new LanguageException(this.printBlockErrorLocation() + "CV training function " + testFCI.getNamespace() + "::" + testFCI.getName() + " is not available ");
		}
		
		// check test function has correct number of parameters AND
		// test function inputs are available from either:
		//	 1) already defined variables
		//	 2) partition function outputs
		//	 3) train function outputs
		tempIds = new VariableSet();
		tempIds.addVariables(ids);
		tempIds.addVariables(partitionOutputList);
		tempIds.addVariables(trainOutputList);
		try {
			testFCI.validateExpression(dmlProg,tempIds.getVariables(), constVars);
		} catch(IOException e){
			throw new LanguageException(e);
		}
		
		// build list of test function outputs
		// check test function call has correct number of parameters		
		FunctionStatement testFS = (FunctionStatement)testFSB.getStatement(0);
		if (cvs.get_testFunctionOutputs().getParamExpressions().size() != testFS.getOutputParams().size()){
			LOG.error(this.printBlockErrorLocation() + "CV test function " + testFCI.getNamespace() + "::" + testFCI.getName() + " has wrong number parameters ");
			throw new LanguageException(this.printBlockErrorLocation() + "CV training function " + testFCI.getNamespace() + "::" + testFCI.getName() + " has wrong number parameters ");

		}
		
		// for each test function parameter, set datatype and valuetype
		VariableSet testOutputList = new VariableSet();
		for (int i = 0; i< cvs.get_testFunctionOutputs().getParamExpressions().size(); i++){
			DataIdentifier output = (DataIdentifier)cvs.get_testFunctionOutputs().getParamExpressions().get(i);
			try {
				output.setTypeInfo(testFS.getOutputParams().get(i).getValueType().toString(), testFS.getOutputParams().get(i).getDataType().toString());
			} catch (ParseException e) {
				throw new LanguageException(e);
			}
			testOutputList.addVariable( ((DataIdentifier)output).getName(),(DataIdentifier)output);
				
		}
		
		///////////////////////////////////////////////////////////////////////
		// handle agg function
		///////////////////////////////////////////////////////////////////////
	
		// check agg function exists
		Identifier aggFCI = cvs.get_aggFunctionCall();
		
		
		
		if (aggFCI instanceof FunctionCallIdentifier){
		
			tempNS = dmlProg.getFunctionStatementBlocks(((FunctionCallIdentifier)aggFCI).getNamespace());
			if (tempNS == null){
				LOG.error(this.printBlockErrorLocation() + "Namespace " + ((FunctionCallIdentifier)aggFCI).getNamespace() + " is undefined ");
				
				throw new LanguageException(this.printBlockErrorLocation() + "Namespace " + testFCI.getNamespace() + " is undefined ");
			}
			FunctionStatementBlock aggFSB = tempNS.get(((FunctionCallIdentifier)aggFCI).getName());
			if (aggFSB == null){
				LOG.error(this.printBlockErrorLocation() + "CV aggregation function " + ((FunctionCallIdentifier)aggFCI).getNamespace() + "::" + ((FunctionCallIdentifier)aggFCI).getName() + " is not available ");
				throw new LanguageException(this.printBlockErrorLocation() + "CV aggregation function " + ((FunctionCallIdentifier)aggFCI).getNamespace() + "::" + ((FunctionCallIdentifier)aggFCI).getName() + " is not available ");
			}
		}
		
		// check agg function has correct number of parameters AND
		// agg function inputs are available from either:
		//	 1) already defined variables
		//	 2) partition function outputs
		//	 3) train function outputs
		//	 4) test function outputs
		tempIds = new VariableSet();
		tempIds.addVariables(ids);
		tempIds.addVariables(partitionOutputList);
		tempIds.addVariables(trainOutputList);
		tempIds.addVariables(testOutputList);
		
		try {
			if (aggFCI instanceof FunctionCallIdentifier){
				FunctionCallIdentifier aggFCI_fci = ((FunctionCallIdentifier)aggFCI);
				aggFCI_fci.validateExpression(dmlProg,tempIds.getVariables(), constVars);
				FunctionStatementBlock aggFSB = dmlProg.getFunctionStatementBlock(aggFCI_fci.getNamespace(), aggFCI_fci.getName());
				FunctionStatement aggFS = (FunctionStatement)aggFSB.getStatement(0);
				if (cvs.get_aggFunctionOutputs().getParamExpressions().size() != aggFS.getOutputParams().size()){
					LOG.error(this.printBlockErrorLocation() + "CV aggregate function " + ((FunctionCallIdentifier)aggFCI).getNamespace() + "::" + ((FunctionCallIdentifier)aggFCI).getName() + " has wrong number parameters ");
					throw new LanguageException(this.printBlockErrorLocation() + "CV aggregate function " + ((FunctionCallIdentifier)aggFCI).getNamespace() + "::" + ((FunctionCallIdentifier)aggFCI).getNamespace()  + " has wrong number parameters ");
			
				}
				
				for (int i = 0; i < cvs.get_aggFunctionOutputs().getParamExpressions().size(); i++){
					DataIdentifier output = (DataIdentifier)cvs.get_aggFunctionOutputs().getParamExpressions().get(i);
					try {
						output.setTypeInfo(aggFS.getOutputParams().get(i).getValueType().toString(), aggFS.getOutputParams().get(i).getDataType().toString());
					} catch (ParseException e) {
						throw new LanguageException(e);
					}
					ids.addVariable(output.getName(), new DataIdentifier(output));
				}
			}
			else {
				BuiltinFunctionExpression aggBFE = (BuiltinFunctionExpression)aggFCI;
				aggBFE.validateExpression(tempIds.getVariables(), constVars);
				DataIdentifier output = (DataIdentifier)cvs.get_aggFunctionOutputs().getParamExpressions().get(0);
				try {
					output.setTypeInfo(aggBFE.getOutput().getValueType().toString(), aggBFE.getOutput().getDataType().toString());
				} catch (ParseException e) {
					throw new LanguageException(e);
				}
				ids.addVariable(output.getName(), new DataIdentifier(output));
				
			}
		} catch(IOException e){
			throw new LanguageException(e);
		}

	
		
		// add error aggregate output to returned variables 
		
		for (Expression aggOutput : cvs.get_aggFunctionOutputs().getParamExpressions()){
			
		}
		
		return ids;	
	}
	
	public ArrayList<Hop> get_hops() {
		
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