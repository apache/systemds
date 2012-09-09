package com.ibm.bi.dml.parser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LanguageException;


public class FunctionStatementBlock extends StatementBlock {
	
	/**
	 *  TODO: DRB:  This needs to be changed to reflect:
	 *  
	 *    1)  Default values for variables -- need to add R styled check here to make sure that once vars with 
	 *    default values start, they keep going to the right
	 *    
	 *    2)  The other parameters for External Functions
	 * @throws IOException 
	 */
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String,ConstIdentifier> constVars) 
		throws LanguageException, ParseException, IOException 
	{
		
		if (_statements.size() > 1)
			throw new LanguageException(this.printBlockErrorLocation() + "FunctionStatementBlock should have only 1 statement (FunctionStatement)");
		
		FunctionStatement fstmt = (FunctionStatement) _statements.get(0);
			
		if (!(fstmt instanceof ExternalFunctionStatement)){
			
			fstmt.setBody(StatementBlock.mergeFunctionCalls(fstmt.getBody(), dmlProg));
			
			// perform validate for function body
			this._dmlProg = dmlProg;
			for(StatementBlock sb : fstmt.getBody())
			{
				ids = sb.validate(dmlProg, ids, constVars);
				constVars = sb.getConstOut();
			}
			_constVarsIn.putAll(fstmt.getBody().get(0).getConstIn());
			_constVarsOut.putAll(fstmt.getBody().get(fstmt.getBody().size()-1).getConstOut());
		}
		else {
			//validate specified attributes and attribute values
			ExternalFunctionStatement efstmt = (ExternalFunctionStatement) fstmt;
			efstmt.validateParameters();
			
			//validate child statements
			this._dmlProg = dmlProg;
			for(StatementBlock sb : efstmt.getBody()) //TODO MB: Is this really necessary? Can an ExternalFunction, implemented in Java, really have child statement blocks?
			{
				ids = sb.validate(dmlProg, ids, constVars);
				constVars = sb.getConstOut();
			}
		}
		
		
		return ids;
	}

	public VariableSet initializeforwardLV(VariableSet activeInPassed) throws LanguageException {
		
		FunctionStatement fstmt = (FunctionStatement)_statements.get(0);
		if (_statements.size() > 1)
			throw new LanguageException(this.printBlockErrorLocation() + "FunctionStatementBlock should have only 1 statement (while statement)");
		
		
		_read = new VariableSet();
		_gen = new VariableSet();
				
		VariableSet current = new VariableSet();
		current.addVariables(activeInPassed);
		
		for (int  i = 0; i < fstmt.getBody().size(); i++){
			
			StatementBlock sb = fstmt.getBody().get(i);
			current = sb.initializeforwardLV(current);	
			
			// for each generated variable in this block, check variable not killed
			// in prior statement block in while stmt blody
			for (String varName : sb._gen.getVariableNames()){
				
				// IF the variable is NOT set in the while loop PRIOR to this stmt block, 
				// THEN needs to be generated
				if (!_kill.getVariableNames().contains(varName)){
					_gen.addVariable(varName, sb._gen.getVariable(varName));	
				}
			}
			
			_read.addVariables(sb._read);
			_updated.addVariables(sb._updated);
		
			// only add kill variables for statement blocks guaranteed to execute
			if (!(sb instanceof WhileStatementBlock) && !(sb instanceof IfStatementBlock) && !(sb instanceof ForStatementBlock) ){
				_kill.addVariables(sb._kill);
			}	
		}
		
		// activeOut includes variables from passed live in and updated in the while body
		_liveOut = new VariableSet();
		_liveOut.addVariables(current);
		_liveOut.addVariables(_updated);
		return _liveOut;
	}

	public VariableSet initializebackwardLV(VariableSet loPassed) throws LanguageException{
		
		FunctionStatement wstmt = (FunctionStatement)_statements.get(0);
			
		VariableSet lo = new VariableSet();
		lo.addVariables(loPassed);
		
		// calls analyze for each statement block in while stmt body
		int numBlocks = wstmt.getBody().size();
		for (int i = numBlocks - 1; i >= 0; i--){
			lo = wstmt.getBody().get(i).analyze(lo);
		}	
		
		VariableSet loReturn = new VariableSet();
		loReturn.addVariables(lo);
		return loReturn;
	
	}
	
	
	public ArrayList<Hops> get_hops() throws HopsException {
		
		if (_hops != null && _hops.size() > 0){
			throw new HopsException(this.printBlockErrorLocation() + "there should be no HOPs associated with the FunctionStatementBlock");
		}
		
		return _hops;
	}
	
	public VariableSet analyze(VariableSet loPassed) throws LanguageException{
		throw new LanguageException(this.printBlockErrorLocation() + "Both liveIn and liveOut variables need to be specified for liveness analysis for FunctionStatementBlock");
		
	}
	
	public VariableSet analyze(VariableSet liPassed, VariableSet loPassed) throws LanguageException{
 		
		VariableSet candidateLO = new VariableSet();
		candidateLO.addVariables(loPassed);
		candidateLO.addVariables(_gen);
		
		VariableSet origLiveOut = new VariableSet();
		origLiveOut.addVariables(_liveOut);
		
		_liveOut = new VariableSet();
	 	for (String name : candidateLO.getVariableNames()){
	 		if (origLiveOut.containsVariable(name)){
	 			_liveOut.addVariable(name, candidateLO.getVariable(name));
	 		}
	 	}
	 	
		initializebackwardLV(_liveOut);
		
		// Cannot remove kill variables
		_liveIn = new VariableSet();
		_liveIn.addVariables(liPassed);
		
		for(String key : _gen.getVariableNames()){
			if (_liveIn.containsVariable(key) == false){
				throw new LanguageException(this.getStatement(0).printErrorLocation() + "function " + ((FunctionStatement)this.getStatement(0)).getName() + " requires variable " + key + " to be passed as formal parameter");
			}
		}
		
		VariableSet liveInReturn = new VariableSet();
		liveInReturn.addVariables(_liveIn);
		
		return liveInReturn;
	}
}