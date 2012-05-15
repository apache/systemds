package dml.parser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import dml.hops.Hops;
import dml.lops.Lops;
import dml.utils.HopsException;
import dml.utils.LanguageException;

public class ForStatementBlock extends StatementBlock {
	
	protected Hops _fromHops        = null;
	protected Hops _toHops          = null;
	protected Hops _incrementHops   = null;
	
	protected Lops _fromLops        = null;
	protected Lops _toLops          = null;
	protected Lops _incrementLops   = null;

	public IterablePredicate getIterPredicate(){
		return ((ForStatement)_statements.get(0)).getIterablePredicate();
	}

	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String,ConstIdentifier> constVars) throws LanguageException, IOException {
		
		if (_statements.size() > 1)
			throw new LanguageException("ForStatementBlock should have only 1 statement (for statement)");
		
		ForStatement fs = (ForStatement) _statements.get(0);
		fs.setBody(StatementBlock.mergeFunctionCalls(fs.getBody(), dmlProg));
		
		IterablePredicate predicate = fs.getIterablePredicate();
		
		// process the statement blocks in the body of the for statement
		predicate.validateExpression(ids.getVariables());
		ArrayList<StatementBlock> body = fs.getBody();
		
		this._dmlProg = dmlProg;
		for(StatementBlock sb : body)
		{
			ids = sb.validate(dmlProg, ids, constVars);
			constVars = sb.getConstOut();
		}
		_constVarsIn.putAll(body.get(0).getConstIn());
		_constVarsOut.putAll(body.get(body.size()-1).getConstOut());
		
		return ids;
	}
	
	public VariableSet initializeforwardLV(VariableSet activeInPassed) throws LanguageException {
		
		ForStatement fstmt = (ForStatement)_statements.get(0);
		if (_statements.size() > 1)
			throw new LanguageException("ForStatementBlock should have only 1 statement (for statement)");
		
		
		_read = new VariableSet();
		_read.addVariables(fstmt.getIterablePredicate().variablesRead());
		_updated.addVariables(fstmt.getIterablePredicate().variablesUpdated());
		
		_gen = new VariableSet();
		_gen.addVariables(fstmt.getIterablePredicate().variablesRead());

		// add the iterVar from iterable predicate to kill set 
		_kill.addVariables(fstmt.getIterablePredicate().variablesUpdated());
		
		VariableSet current = new VariableSet();
		current.addVariables(activeInPassed);
		current.addVariables(_updated);
		
		
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
		
		// set preliminary "warn" set -- variables that if used later may cause runtime error
		// if the loop is not executed
		// warnSet = (updated MINUS (updatedIfBody INTERSECT updatedElseBody)) MINUS current
		for (String varName : _updated.getVariableNames()){
			if (!activeInPassed.containsVariable(varName)) {
				_warnSet.addVariable(varName, _updated.getVariable(varName));
			}
		}
		
		// activeOut includes variables from passed live in and updated in the while body
		_liveOut = new VariableSet();
		_liveOut.addVariables(current);
		_liveOut.addVariables(_updated);
		return _liveOut;
	}

	public VariableSet initializebackwardLV(VariableSet loPassed) throws LanguageException{
		
		ForStatement fstmt = (ForStatement)_statements.get(0);
			
		VariableSet lo = new VariableSet();
		lo.addVariables(loPassed);
		
		// calls analyze for each statement block in while stmt body
		int numBlocks = fstmt.getBody().size();
		for (int i = numBlocks - 1; i >= 0; i--){
			lo = fstmt.getBody().get(i).analyze(lo);
		}	
		
		VariableSet loReturn = new VariableSet();
		loReturn.addVariables(lo);
		return loReturn;
	
	}

	public ArrayList<Hops> get_hops() throws HopsException {
		
		if (_hops != null && _hops.size() > 0){
			throw new HopsException("there should be no HOPs associated with the ForStatementBlock");
		}
		
		return _hops;
	}

	public void setFromHops(Hops hops) { _fromHops = hops; }
	public void setToHops(Hops hops) { _toHops = hops; }
	public void setIncrementHops(Hops hops) { _incrementHops = hops; }
	
	public Hops getFromHops()      { return _fromHops; }
	public Hops getToHops()        { return _toHops; }
	public Hops getIncrementHops() { return _incrementHops; }

	public void setFromLops(Lops lops) { _fromLops = lops; }
	public void setToLops(Lops lops) { _toLops = lops; }
	public void setIncrementLops(Lops lops) { _incrementLops = lops; }
	
	public Lops getFromLops()      { return _fromLops; }
	public Lops getToLops()        { return _toLops; }
	public Lops getIncrementLops() { return _incrementLops; }

	
	
	public VariableSet analyze(VariableSet loPassed) throws LanguageException{
 		
		VariableSet predVars = new VariableSet();
		IterablePredicate ip = ((ForStatement)_statements.get(0)).getIterablePredicate(); 
		
		predVars.addVariables(ip.variablesRead());
		predVars.addVariables(ip.variablesUpdated());
		
		VariableSet candidateLO = new VariableSet();
		candidateLO.addVariables(loPassed);
		candidateLO.addVariables(_gen);
		candidateLO.addVariables(predVars);
		
		VariableSet origLiveOut = new VariableSet();
		origLiveOut.addVariables(_liveOut);
		origLiveOut.addVariables(predVars);
		origLiveOut.addVariables(_gen);
		
		_liveOut = new VariableSet();
	 	for (String name : candidateLO.getVariableNames()){
	 		if (origLiveOut.containsVariable(name)){
	 			_liveOut.addVariable(name, candidateLO.getVariable(name));
	 		}
	 	}
	 	
		initializebackwardLV(_liveOut);
		
		// set final warnSet: remove variables NOT in live out
		VariableSet finalWarnSet = new VariableSet();
		for (String varName : _warnSet.getVariableNames()){
			if (_liveOut.containsVariable(varName)){
				finalWarnSet.addVariable(varName,_warnSet.getVariable(varName));
			}
		}
		_warnSet = finalWarnSet;
		
		// for now just print the warn set
		for (String varName : _warnSet.getVariableNames()){
			if( !ip.getIterVar().getName().equals( varName)  )
				System.out.println("***** WARNING: Initialization of " + varName + " on line " + _warnSet.getVariable(varName).getDefinedLine() + " depends on for execution");
		}
		
		// Cannot remove kill variables
		_liveIn = new VariableSet();
		_liveIn.addVariables(_liveOut);
		_liveIn.addVariables(_gen);
		
		VariableSet liveInReturn = new VariableSet();
		liveInReturn.addVariables(_liveIn);
		
		return liveInReturn;
	}
}