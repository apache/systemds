/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

//package com.ibm.bi.dml.parser;
//
//import java.util.ArrayList;
//import java.util.HashMap;
//
//import com.ibm.bi.dml.hops.DataOp;
//import com.ibm.bi.dml.hops.Hops;
//import com.ibm.bi.dml.hops.Hops.DataOpTypes;
//import com.ibm.bi.dml.hops.PartitionOp;
//import com.ibm.bi.dml.meta.PartitionParams;
//import com.ibm.bi.dml.parser.Expression.DataType;
//import com.ibm.bi.dml.parser.Expression.ValueType;
//import com.ibm.bi.dml.utils.LanguageException;
//
//
//public class ELUseStatementBlock extends StatementBlock {
//	
//	private Hops _partitionHop ;
//	
//	public String toString() {
//		StringBuffer sb = new StringBuffer();
//		ELUseStatement statement = (ELUseStatement) getStatement(0) ;
//		sb.append(statement);
//		return sb.toString();
//	}
//	
//	public boolean mergeable(){
//		return false;
//	}
//	
//	public PartitionParams getPartitionParams() {
//		CVStatement statement = (CVStatement) getStatement(0) ;
//		return statement.getPartitionParams() ;
//	}
//			
//	public Hops getPartitionHop() {
//		return _partitionHop ;
//	}
//	
//	@Override
//	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String, ConstIdentifier> constVars) throws LanguageException {
//
//		if (this.getNumStatements() > 1){
//			LOG.error(this.printBlockErrorLocation() + "Use Ensemble statement block can only have single statement");
//			throw new LanguageException(this.printBlockErrorLocation() + "Use Ensemble statement block can only have single statement");
//		}
//		ELUseStatement eustmt = (ELUseStatement) this.getStatement(0);
//		
//		// check the ensemble is available
//		if (!ids.containsVariable(eustmt.getEnsembleName())){
//			LOG.error(eustmt.printErrorLocation() + "Ensemble " + eustmt.getEnsembleName() + " in Use Ensemble statement not available");
//			throw new LanguageException(eustmt.printErrorLocation() + "Ensemble " + eustmt.getEnsembleName() + " in Use Ensemble statement not available");
//		}
//		
//		// check the input datasets are available
//		for (String input : eustmt.getInputNames()){
//			if (!ids.containsVariable(input)){
//				LOG.error(eustmt.printErrorLocation() + "Use Ensemble statement input dataset " + input + " is not available ");
//				throw new LanguageException(eustmt.printErrorLocation() + "Use Ensemble statement input dataset " + input + " is not available ");
//			}
//		}
//		
//		// check the test function is available
//		if (!dmlProg.getFunctionStatementBlocks(null).containsKey(eustmt.getFunctionParameters().getTestFunctionName())) {
//			LOG.error(eustmt.printErrorLocation() + "use ensemble test function " + eustmt.getFunctionParameters().getTestFunctionName() + " is not available ");
//			throw new LanguageException(eustmt.printErrorLocation() + "use ensemble test function " + eustmt.getFunctionParameters().getTestFunctionName() + " is not available ");
//		}
//		
//		// check the test function parameters are available
//		for (String input : eustmt.getFunctionParameters().getErrorAggFormalParams()){
//			
//			// check if the parameter already exists as variable
//			if (!ids.containsVariable(input)) {
//				LOG.error(eustmt.printErrorLocation() + "In Use Ensemble statement, variable " + input + " is not available but required by error aggregation function");
//				throw new LanguageException(eustmt.printErrorLocation() + "In Use Ensemble statement, variable " + input + " is not available but required by error aggregation function");
//			}
//		}
//		
//		// check the agg function inputs are either generated as test outputs or available as variables
//		for (String input : eustmt.getFunctionParameters().getErrorAggFormalParams()){
//			
//			boolean found = false;
//			if (eustmt.getFunctionParameters().getTestFunctionReturnParams() != null){
//				for (String input2 : eustmt.getFunctionParameters().getTestFunctionReturnParams()){
//					if (input2.equals(input)) found = true;
//				}
//			}
//			
//			if (ids.containsVariable(input)) found = true;
//			
//			if (found == false) {
//				LOG.error(eustmt.printErrorLocation() + "In Use Ensemble statement, variable " + input + " is not available but required by error aggregation function");
//				throw new LanguageException(eustmt.printErrorLocation() + "In Use Ensemble statement, variable " + input + " is not available but required by error aggregation function");
//			}
//		}
//		
//		// add aggrFunction outputs to set of available variables
//		for (String aggOutput : eustmt.getFunctionParameters().getErrorAggReturnParams()){
//			ids.addVariable(aggOutput, new DataIdentifier(aggOutput));
//		}
//		
//		return ids;
//		
//	}
//	
//	public ArrayList<Hops> get_hops() {
//		ELStatement stmt = (ELStatement) getStatement(0) ;
//		PartitionParams pp = stmt.getPartitionParams() ;
//		MetaLearningFunctionParameters params = stmt.getFunctionParameters() ;
//		
//		// read the input files to be partitioned
//		ArrayList<Hops> cvHops = new ArrayList<Hops>() ;
//		for (String input : pp.getInputs()){
//		
//			DataIdentifier var = this.liveIn().getVariables().get(input);
//			
//			long actualDim1 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim1() : var.getDim1();
//			long actualDim2 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim2() : var.getDim2();
//			
//			DataOp read = new DataOp(var.getName(), var.getDataType(), var.getValueType(), 
//					DataOpTypes.TRANSIENTREAD, null, actualDim1, actualDim2, var.getNnz(),
//					var.getRowsInBlock(), var.getColumnsInBlock());
//			read.setAllPositions(var.getBeginLine(), var.getBeginColumn(), var.getEndLine(), var.getEndColumn());
//			read.set_rows_in_block( var.getRowsInBlock()) ;
//			read.set_cols_in_block( var.getColumnsInBlock()) ;
//			
//			// create the partition HOP
//			_partitionHop = new PartitionOp("B", DataType.MATRIX, ValueType.DOUBLE, pp, read);
//			cvHops.add(_partitionHop) ;
//		}
//		
//		set_hops(cvHops);
//		return cvHops;
//	}
//
//	
//}