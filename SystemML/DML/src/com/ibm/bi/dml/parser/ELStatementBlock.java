package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.PartitionOp;
import com.ibm.bi.dml.hops.Hops.DataOpTypes;
import com.ibm.bi.dml.meta.PartitionParams;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LanguageException;


public class ELStatementBlock extends StatementBlock {
	
	private Hops _partitionHop ;
	
	public String toString() {
		StringBuffer sb = new StringBuffer();
		ELStatement statement = (ELStatement) getStatement(0) ;
		sb.append(statement);
		return sb.toString();
	}
	
	public boolean mergeable(){
		return false;
	}
	
	public PartitionParams getPartitionParams() {
		ELStatement statement = (ELStatement) getStatement(0) ;
		return statement.getPartitionParams() ;
	}
		
	public MetaLearningFunctionParameters getELFunctionParameters() {
		return ((ELStatement) getStatement(0)).getFunctionParameters();
	}
	
	public Hops getPartitionHop() {
		return _partitionHop ;
	}
	
	public void setPartitionHop(Hops p) {
		_partitionHop = p;
	}
	
	
	@Override
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String, ConstIdentifier> constVars) throws LanguageException {

		if (this.getNumStatements() > 1)
			throw new LanguageException("Ensemble statement block can only have single statement");
		ELStatement estmt = (ELStatement) this.getStatement(0);
		
		// add ensemble to available variables set
		ids.addVariable(estmt.getEnsembleName(), new DataIdentifier(estmt.getEnsembleName()));
			
		// check the input datasets are available
		for (String input : estmt.getInputNames()){
			if (!ids.containsVariable(input))
				throw new LanguageException("Ensemble statement input dataset " + input + " is not available ");
		}
		
		// check the training function is available
		if (!dmlProg.getFunctionStatementBlocks(null).containsKey(estmt.getFunctionParameters().getTrainFunctionName()))
			throw new LanguageException("Ensemble training function " + estmt.getFunctionParameters().getTrainFunctionName() + " is not available ");
		
		// check training function inputs are available
		for (String input : estmt.getFunctionParameters().getTrainFunctionFormalParams()){
		
			boolean found = false;
			
			// check if input is available from existing variable
			if (ids.containsVariable(input)) found = true;
			
			// check if input is available as output of partition
			for (String input2 : estmt.getFunctionParameters().getPartitionReturnParams()){
				if (input2.equals(input)) found = true;
			}
			
			if (found == false)
				throw new LanguageException("In Build Ensemble statement, variable " + input + " is not available but required by train function");
			
		}
		
		// if specified, check test function is available
		if (estmt.getFunctionParameters().getTestFunctionName() != null){
			
			// validate test function has been specified
			if (!dmlProg.getFunctionStatementBlocks(null).containsKey(estmt.getFunctionParameters().getTestFunctionName())){
				throw new LanguageException("Ensemble test function " + estmt.getFunctionParameters().getTestFunctionName() + " is not available ");
			}
			
			// validate test function parameters available 
			
			for (String input : estmt.getFunctionParameters().getTestFunctionFormalParams()){
				boolean found = false;
				
				if (ids.containsVariable(input)) found = true;
				
				for (String input2 : estmt.getFunctionParameters().getPartitionReturnParams()){
					if (input2.equals(input)) found = true;
				}
				
				for (String input2 : estmt.getFunctionParameters().getTrainFunctionReturnParams()){
					if (input2.equals(input)) found = true;
				}
				
				if (found == false)
					throw new LanguageException("In Build Ensemble statement, variable " + input + " is not available but required by train function");
					
			}
			
		}
		return ids;	
	}
	
	public ArrayList<Hops> get_hops() {
		ELStatement statement = (ELStatement) getStatement(0) ;
		PartitionParams pp = statement.getPartitionParams() ;
		MetaLearningFunctionParameters params = statement.getFunctionParameters() ;
		
		// read the input file to be partitioned
		ArrayList<Hops> eHops = new ArrayList<Hops>();
		for (String input : pp.getInputs()){
		
			DataIdentifier var = this.liveIn().getVariables().get(input);
			DataOp read = new DataOp(var.getName(), var.getDataType(), var.getValueType(), 
				DataOpTypes.TRANSIENTREAD, null, var.getDim1(), var.getDim2(), 
				(int)var.getRowsInBlock(), (int)var.getColumnsInBlock());
			read.set_rows_per_block((int) var.getRowsInBlock()) ;
			read.set_cols_per_block((int) var.getColumnsInBlock()) ;
		
			// create the partition HOP
			_partitionHop = new PartitionOp("B", DataType.MATRIX, ValueType.DOUBLE, pp, read) ;
			eHops.add(_partitionHop) ;
		}
		
		set_hops(eHops) ;
		return eHops ;
	}

	
}