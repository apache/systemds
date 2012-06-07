package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.Hops.DataOpTypes;
import com.ibm.bi.dml.hops.PartitionOp;
import com.ibm.bi.dml.meta.PartitionParams;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LanguageException;


public class CVStatementBlock extends StatementBlock {
	
	Hops partitionHop ;
	
	public String toString() {
		StringBuffer sb = new StringBuffer();
		CVStatement statement = (CVStatement) getStatement(0) ;
		sb.append(statement);
		return sb.toString();
	}
	
	public boolean mergeable(){
		return false;
	}
	
	public PartitionParams getPartitionParams() {
		CVStatement statement = (CVStatement) getStatement(0) ;
		return statement.getPartitionParams() ;
	}
			
	public Hops getPartitionHop() {
		return partitionHop ;
	}
	
	@Override
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String, ConstIdentifier> constVars) throws LanguageException {

		if (this.getNumStatements() > 1)
			throw new LanguageException("CV statement block can only have single statement");
		CVStatement cvs = (CVStatement) this.getStatement(0);
		
		// check the input datasets are available
		for (String input : cvs.getInputNames()){
			if (!ids.containsVariable(input)){
				throw new LanguageException("CV statement input dataset " + input + " is not available ");
			}
		}	
		
		// check train function exists and train function inputs are available
		if (!dmlProg.getFunctionStatementBlocks(null).containsKey(cvs.getFunctionParameters().getTrainFunctionName()))
			throw new LanguageException("CV training function " + cvs.getFunctionParameters().getTrainFunctionName() + " is not available ");
		
		for (String input : cvs.getFunctionParameters().getTrainFunctionFormalParams()){
			boolean found = false;
			
			if (ids.containsVariable(input)) found = true;
			
			for (String input2 : cvs.getFunctionParameters().getPartitionReturnParams()){
				if (input2.equals(input)) found = true;
			}
			
			if (found == false)
				throw new LanguageException("In CV Statement, variable " + input + " is not available");
		}
		
			
		// check test function exists and test function inputs are available
		if (!dmlProg.getFunctionStatementBlocks(null).containsKey(cvs.getFunctionParameters().getTestFunctionName()))
			throw new LanguageException("CV test function " + cvs.getFunctionParameters().getTestFunctionName() + " is not available ");
							
		for (String input : cvs.getFunctionParameters().getTestFunctionFormalParams()){
			boolean found = false;
			
			if (ids.containsVariable(input)) found = true;
			
			for (String input2 : cvs.getFunctionParameters().getPartitionReturnParams()){
				if (input2.equals(input)) found = true;
			}
			
			for (String input2 : cvs.getFunctionParameters().getTrainFunctionReturnParams()){
				if (input2.equals(input)) found = true;
			}
			
			if (found == false)
				throw new LanguageException("In CV Statement, variable " + input + " is not available");
		}
			
		// add error aggregate output to returned variables 
		for (String aggOutput : cvs.getFunctionParameters().getErrorAggReturnParams()){
			ids.addVariable(aggOutput, new DataIdentifier(aggOutput));
		}
		
		return ids;	
	}
	
	public ArrayList<Hops> get_hops() {
		
		if (this._hops == null){
			
			CVStatement statement = (CVStatement) getStatement(0) ;
			PartitionParams pp = statement.getPartitionParams() ;
			MetaLearningFunctionParameters params = statement.getFunctionParameters() ;
			
			// read each input file to be partitioned
			ArrayList<Hops> cvHops = new ArrayList<Hops>() ;
			for (String input : pp.getInputs()){
				
				DataIdentifier var = this.liveIn().getVariables().get(input);
				DataOp read = new DataOp(var.getName(), var.getDataType(), var.getValueType(), 
					DataOpTypes.TRANSIENTREAD, null, var.getDim1(), var.getDim2(),var.getNnz(), 
					var.getRowsInBlock(), var.getColumnsInBlock());
				read.set_rows_in_block( var.getRowsInBlock()) ;
				read.set_cols_in_block( var.getColumnsInBlock()) ;
			
				// create the partition HOP
				partitionHop = new PartitionOp("B", DataType.MATRIX, ValueType.DOUBLE, pp, read) ;
			
				// create the CV HOP
				//crossvalHop = new CrossvalOp("CVHop", DataType.MATRIX, ValueType.DOUBLE, partitionHop, params) ;
			
				cvHops.add(partitionHop) ;
				set_hops(cvHops) ;
			}
		} // if (this._hops == null)
		
		return this._hops;
		
	}
}