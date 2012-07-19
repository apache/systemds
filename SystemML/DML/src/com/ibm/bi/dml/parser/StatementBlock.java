package com.ibm.bi.dml.parser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.FormatType;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.FunctionCallCPInstruction;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LanguageException;


public class StatementBlock extends LiveVariableAnalysis{

	protected DMLProgram _dmlProg; 
	protected ArrayList<Statement> _statements;
	ArrayList<Hops> _hops = null;
	ArrayList<Lops> _lops = null;
	HashMap<String,ConstIdentifier> _constVarsIn;
	HashMap<String,ConstIdentifier> _constVarsOut;
	
	// this stores the function call instruction 
	private FunctionCallCPInstruction _functionCallInst;
	
	public StatementBlock(){
		_dmlProg = null;
		_statements = new ArrayList<Statement>();
		_read = new VariableSet();
		_updated = new VariableSet(); 
		_gen = new VariableSet();
		_kill = new VariableSet();
		_warnSet = new VariableSet();
		_initialized = true;
		_constVarsIn = new HashMap<String,ConstIdentifier>();
		_constVarsOut = new HashMap<String,ConstIdentifier>();
		_functionCallInst = null;
	}
	
	public void setFunctionCallInst(FunctionCallCPInstruction fci){
		_functionCallInst = fci;
	}
	
	public FunctionCallCPInstruction getFunctionCallInst(){
		return _functionCallInst;
	}
	
	
	public void setDMLProg(DMLProgram dmlProg){
		_dmlProg = dmlProg;
	}
	
	public DMLProgram getDMLProg(){
		return _dmlProg;
	}
	
	public void addStatement(Statement s){
		_statements.add(s);
	}
	
	/**
	 * replace statement 
	 */
	public void replaceStatement(int index, Statement passedStmt){
		this._statements.set(index, passedStmt);
	}
	
	public void addStatementBlock(StatementBlock s){
		for (int i = 0; i < s.getNumStatements(); i++){
			_statements.add(s.getStatement(i));
		}
	}
	
	public int getNumStatements(){
		return _statements.size();
	}

	public Statement getStatement(int i){
		return _statements.get(i);
	}
	
	public ArrayList<Statement> getStatements()
	{
		return _statements;
	}

	public ArrayList<Hops> get_hops() throws HopsException {
		return _hops;
	}

	public ArrayList<Lops> get_lops() {
		return _lops;
	}

	public void set_hops(ArrayList<Hops> hops) {
		_hops = hops;
	}

	public void set_lops(ArrayList<Lops> lops) {
		_lops = lops;
	}

	public boolean mergeable(){
		for (Statement s : _statements){	
			if (s.controlStatement())
				return false;
		}
		return true;
	}

	
    public boolean isMergeableFunctionCallBlock(DMLProgram dmlProg) throws LanguageException{
		
		// check whether targetIndex stmt block is for a mergable function call 
		Statement stmt = this.getStatement(0);
		
		// Check whether targetIndex block is: control stmt block or stmt block for un-mergable function call
		if (stmt instanceof WhileStatement || stmt instanceof IfStatement || stmt instanceof ForStatement || 
				stmt instanceof FunctionStatement || stmt instanceof CVStatement || stmt instanceof ELStatement)
			return false;
		
		// for regular stmt block, check if this is a function call stmt block
		if (stmt instanceof AssignmentStatement || stmt instanceof MultiAssignmentStatement){
			Expression sourceExpr = null;
			if (stmt instanceof AssignmentStatement)
				sourceExpr = ((AssignmentStatement)stmt).getSource();
			else
				sourceExpr = ((MultiAssignmentStatement)stmt).getSource();
			
			if (sourceExpr instanceof FunctionCallIdentifier){
				FunctionCallIdentifier fcall = (FunctionCallIdentifier) sourceExpr;
				FunctionStatementBlock fblock = dmlProg.getFunctionStatementBlock(fcall.getNamespace(), fcall.getName());
				if (fblock == null)
					throw new LanguageException("function " + fcall.getName() + " is undefined");
			
				if (fblock.getStatement(0) instanceof ExternalFunctionStatement  ||  ((FunctionStatement)fblock.getStatement(0)).getBody().size() > 1 ){
					return false;
				}
			}
		}
		// regular function block
		return true;
	}

    public boolean isRewritableFunctionCall(Statement stmt, DMLProgram dmlProg) throws LanguageException{
			
		// for regular stmt, check if this is a function call stmt block
		if (stmt instanceof AssignmentStatement || stmt instanceof MultiAssignmentStatement){
			Expression sourceExpr = null;
			if (stmt instanceof AssignmentStatement)
				sourceExpr = ((AssignmentStatement)stmt).getSource();
			else
				sourceExpr = ((MultiAssignmentStatement)stmt).getSource();
			
			if (sourceExpr instanceof FunctionCallIdentifier){
				FunctionCallIdentifier fcall = (FunctionCallIdentifier) sourceExpr;
				FunctionStatementBlock fblock = dmlProg.getFunctionStatementBlock(fcall.getNamespace(),fcall.getName());
				if (fblock == null)
					throw new LanguageException("function " + fcall.getName() + " is undefined");
			
				if (fblock.getStatement(0) instanceof ExternalFunctionStatement  ||  ((FunctionStatement)fblock.getStatement(0)).getBody().size() > 1 ){
					return false;
				}
				else {
					return true;
				}
			}
		}
		
		// regular statement
		return false;
	}
    
    public boolean isNonRewritableFunctionCall(Statement stmt, DMLProgram dmlProg) throws LanguageException{
		
		// for regular stmt, check if this is a function call stmt block
		if (stmt instanceof AssignmentStatement || stmt instanceof MultiAssignmentStatement){
			Expression sourceExpr = null;
			if (stmt instanceof AssignmentStatement)
				sourceExpr = ((AssignmentStatement)stmt).getSource();
			else
				sourceExpr = ((MultiAssignmentStatement)stmt).getSource();
			
			if (sourceExpr instanceof FunctionCallIdentifier){
				FunctionCallIdentifier fcall = (FunctionCallIdentifier) sourceExpr;
				FunctionStatementBlock fblock = dmlProg.getFunctionStatementBlock(fcall.getNamespace(), fcall.getName());
				if (fblock == null)
					throw new LanguageException("function " + fcall.getName() + " is undefined");
			
				if (fblock.getStatement(0) instanceof ExternalFunctionStatement  ||  ((FunctionStatement)fblock.getStatement(0)).getBody().size() > 1 ){
					return true;
				}
				else {
					return false;
				}
			}
		}
		
		// regular statement
		return false;
	}
    
	
	public static ArrayList<StatementBlock> mergeFunctionCalls(ArrayList<StatementBlock> body, DMLProgram dmlProg) throws LanguageException
	{
		for(int i = 0; i <body.size(); i++){
			
			StatementBlock currBlock = body.get(i);
			
			// recurse to children function statement blocks
			if (currBlock instanceof WhileStatementBlock){
				WhileStatement wstmt = (WhileStatement)((WhileStatementBlock)currBlock).getStatement(0);
				wstmt.setBody(mergeFunctionCalls(wstmt.getBody(),dmlProg));		
			}
			
			else if (currBlock instanceof ForStatementBlock){
				ForStatement fstmt = (ForStatement)((ForStatementBlock)currBlock).getStatement(0);
				fstmt.setBody(mergeFunctionCalls(fstmt.getBody(),dmlProg));		
			}
			
			else if (currBlock instanceof IfStatementBlock){
				IfStatement ifstmt = (IfStatement)((IfStatementBlock)currBlock).getStatement(0);
				ifstmt.setIfBody(mergeFunctionCalls(ifstmt.getIfBody(),dmlProg));		
				ifstmt.setElseBody(mergeFunctionCalls(ifstmt.getElseBody(),dmlProg));
			}
			
			else if (currBlock instanceof FunctionStatementBlock){
				FunctionStatement functStmt = (FunctionStatement)((FunctionStatementBlock)currBlock).getStatement(0);
				functStmt.setBody(mergeFunctionCalls(functStmt.getBody(),dmlProg));		
			}
		}
		
		ArrayList<StatementBlock> result = new ArrayList<StatementBlock>();

		StatementBlock currentBlock = null;

		for (int i = 0; i < body.size(); i++){
			StatementBlock current = body.get(i);
			if (current.isMergeableFunctionCallBlock(dmlProg)){
				if (currentBlock != null) {
					currentBlock.addStatementBlock(current);
				} else {
					currentBlock = current;
				}
			} else {
				if (currentBlock != null) {
					result.add(currentBlock);
				}
				result.add(current);
				currentBlock = null;
			}
		}

		if (currentBlock != null) {
			result.add(currentBlock);
		}
		return result;		
	}
	
	public String toString(){
		StringBuffer sb = new StringBuffer();
		sb.append("statements\n");
		for (Statement s : _statements){
			sb.append(s);
			sb.append("\n");
		}
		if (_liveOut != null) sb.append("liveout " + _liveOut.toString() + "\n");
		if (_liveIn!= null) sb.append("livein " + _liveIn.toString()+ "\n");
		if (_gen != null) sb.append("gen " + _gen.toString()+ "\n");
		if (_kill != null) sb.append("kill " + _kill.toString()+ "\n");
		if (_read != null) sb.append("read " + _read.toString()+ "\n");
		if (_updated != null) sb.append("updated " + _updated.toString()+ "\n");
		return sb.toString();
	}

	public static ArrayList<StatementBlock> mergeStatementBlocks(ArrayList<StatementBlock> sb){

		ArrayList<StatementBlock> result = new ArrayList<StatementBlock>();

		if (sb.size() == 0) {
			return new ArrayList<StatementBlock>();
		}

		StatementBlock currentBlock = null;

		for (int i = 0; i < sb.size(); i++){
			StatementBlock current = sb.get(i);
			if (current.mergeable()){
				if (currentBlock != null) {
					currentBlock.addStatementBlock(current);
				} else {
					currentBlock = current;
				}
			} else {
				if (currentBlock != null) {
					result.add(currentBlock);
				}
				result.add(current);
				currentBlock = null;
			}
		}

		if (currentBlock != null) {
			result.add(currentBlock);
		}
		return result;

	}

	
	public ArrayList<Statement> rewriteFunctionCallStatements (DMLProgram dmlProg, ArrayList<Statement> statements) throws LanguageException {
		
		ArrayList<Statement> newStatements = new ArrayList<Statement>();
		for (Statement current : statements){
			if (isRewritableFunctionCall(current, dmlProg)){
	
				Expression sourceExpr = null;
				if (current instanceof AssignmentStatement)
					sourceExpr = ((AssignmentStatement)current).getSource();
				else
					sourceExpr = ((MultiAssignmentStatement)current).getSource();
					
				FunctionCallIdentifier fcall = (FunctionCallIdentifier) sourceExpr;
				FunctionStatementBlock fblock = dmlProg.getFunctionStatementBlock(fcall.getNamespace(), fcall.getName());
				FunctionStatement fstmt = (FunctionStatement)fblock.getStatement(0);
				String prefix = new Integer(fblock.hashCode()).toString() + "_";
				
				if (fstmt.getBody().size() > 1)
					throw new LanguageException("rewritable function can only have 1 statement block");
				StatementBlock sblock = fstmt.getBody().get(0);
				
				for (int i =0; i < fstmt.getInputParams().size(); i++){
					
					DataIdentifier currFormalParam = fstmt.getInputParams().get(i);
					
					// create new assignment statement
					String newFormalParameterName = prefix + currFormalParam.getName();
					DataIdentifier newTarget = new DataIdentifier(currFormalParam);
					newTarget.setName(newFormalParameterName);
					
					Expression currCallParam = null;
					if (fcall.getParamExpressions().size() > i){
						// function call has value for parameter
						currCallParam = fcall.getParamExpressions().get(i);
					}
					else {
						// use default value for parameter
						if (fstmt.getInputParams().get(i).getDefaultValue() == null)
							throw new LanguageException("line " + currFormalParam.getDefinedLine() + ": default parameter for " + currFormalParam + " is undefined");
						currCallParam = new DataIdentifier(fstmt.getInputParams().get(i).getDefaultValue(),fstmt.getInputParams().get(i).getDefinedLine(), fstmt.getInputParams().get(i).getDefinedCol());
					}
					
					// create the assignment statement to bind the call parameter to formal parameter
					AssignmentStatement binding = new AssignmentStatement(newTarget, currCallParam);
					newStatements.add(binding);
				}
				
				for (Statement stmt : sblock._statements){
					
					// rewrite the statement to use the "rewritten" name					
					Statement rewrittenStmt = stmt.rewriteStatement(prefix);
					newStatements.add(rewrittenStmt);		
				}
				
				// handle the return values
				for (int i = 0; i < fstmt.getOutputParams().size(); i++){
					
					// get the target (return parameter from function)
					DataIdentifier currReturnParam = fstmt.getOutputParams().get(i);
					String newSourceName = prefix + currReturnParam.getName();
					DataIdentifier newSource = new DataIdentifier(currReturnParam);
					newSource.setName(newSourceName);
				
					// get binding 
					DataIdentifier newTarget = null;
					if (current instanceof AssignmentStatement){
						if (i > 0) throw new LanguageException("Assignment statement cannot return multiple values");
						newTarget = new DataIdentifier(((AssignmentStatement)current).getTarget());
					}
					else{
						newTarget = new DataIdentifier(((MultiAssignmentStatement)current).getTargetList().get(i));
					}
					// create the assignment statement to bind the call parameter to formal parameter
					AssignmentStatement binding = new AssignmentStatement(newTarget, newSource);
					newStatements.add(binding);
				}
								
			} // end if (isRewritableFunctionCall(current, dmlProg)
				
//			else if (isNonRewritableFunctionCall(current, dmlProg)){
//				
//				FunctionCallIdentifier fcall = null;
//				if (current instanceof AssignmentStatement)
//					fcall = (FunctionCallIdentifier)((AssignmentStatement)current).getSource();
//				else
//					fcall = (FunctionCallIdentifier)((MultiAssignmentStatement)current).getSource();
//					
//				FunctionStatementBlock fblock = dmlProg.getFunctionStatementBlock(fcall.getName());
//				FunctionStatement fstmt = (FunctionStatement)fblock.getStatement(0);
//				
			//	ArrayList<Expression> newfcallExprs = new ArrayList<Expression>();
				
				// create a new assignment statement for each input
			//	for (int i=0; i < fstmt.inputParams.size(); i++){
					
			//		DataIdentifier currFormalParam = fstmt.inputParams.get(i);
			//		newfcallExprs.add(new DataIdentifier(currFormalParam));
			//		
			//		Expression currCallParam = null;
			//		if (fcall.getParamExpressions().size() > i){
			//			// function call has value for parameter
			//			currCallParam = fcall.getParamExpressions().get(i);
			//		}
			//		else {
			//			// use default value for parameter
			//			if (fstmt.inputParams.get(i).getDefaultValue() == null)
			//				throw new LanguageException("line " + currFormalParam.getDefinedLine() + ": default parameter for " + currFormalParam + " is undefined");
			//			currCallParam = new DataIdentifier(fstmt.inputParams.get(i).getDefaultValue(),fstmt.inputParams.get(i).getDefinedLine(), fstmt.inputParams.get(i).getDefinedCol());
			//		}
			//		
			//		// create the assignment statement to bind the call parameter to formal parameter
			//		AssignmentStatement binding = new AssignmentStatement(currFormalParam, currCallParam);
			//		newStatements.add(binding);
			//	}
			//	
			//	// rewrite current assignment statement to have function call using the given bindings
			//	FunctionCallIdentifier newfcall = new FunctionCallIdentifier(fcall.getName(), newfcallExprs);
			//	
			//	if (current instanceof AssignmentStatement)
			//		((AssignmentStatement)current).setSource(newfcall);
			//	else
			//	((MultiAssignmentStatement)current).setSource(newfcall);
			//	
			//	
			//	newStatements.add(current);
			//
			//}
			else {
				newStatements.add(current);
			}
		}
		
		return newStatements;
	}
	
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String, ConstIdentifier> constVars) throws LanguageException, ParseException, IOException {

		_constVarsIn.putAll(constVars);
		HashMap<String, ConstIdentifier> currConstVars = new HashMap<String,ConstIdentifier>();
		currConstVars.putAll(constVars);
			
		_statements = rewriteFunctionCallStatements(dmlProg, _statements);
		_dmlProg = dmlProg;
		
		for (Statement current : _statements){
			
			if (current instanceof InputStatement){
				InputStatement is = (InputStatement)current;	
				DataIdentifier target = is.getIdentifier(); 
				
				Expression source = is.getSource();
				source.setOutput(target);
				source.validateExpression(ids.getVariables(), currConstVars);
				
				setStatementFormatType(is);
				
				// use existing size and properties information for LHS IndexedIdentifier
				if (target instanceof IndexedIdentifier){
					DataIdentifier targetAsSeen = ids.getVariable(target.getName());
					if (targetAsSeen == null)
						throw new LanguageException("ERROR: cannot assign value to indexed identifier " + target.toString() + " without initializing " + is.getIdentifier().getName());
					target.setProperties(targetAsSeen);
				}
							
				ids.addVariable(target.getName(),target);
			}
			
			else if (current instanceof OutputStatement){
				OutputStatement os = (OutputStatement)current;
				
				// validate variable being written by output statement exists
				DataIdentifier target = (DataIdentifier)os.getIdentifier();
				if (ids.getVariable(target.getName()) == null)
					throwUndefinedVar ( target.getName(), os.toString() );
				
				if ( ids.getVariable(target.getName()).getDataType() == DataType.SCALAR) {
					boolean paramsOkay = true;
					for (String key : os._paramsExpr.getVarParams().keySet()){
						if (!key.equals(Statement.IO_FILENAME)) 
							paramsOkay = false;
					}
					if (paramsOkay == false)
						throw new LanguageException("Invalid parameters in write statement: " + os.toString());
				}
				
				
				Expression source = os.getSource();
				source.setOutput(target);
				source.validateExpression(ids.getVariables(), currConstVars);
				
				setStatementFormatType(os);
				target.setDimensionValueProperties(ids.getVariable(target.getName()));
			}
			
			else if (current instanceof AssignmentStatement){
				AssignmentStatement as = (AssignmentStatement)current;
				DataIdentifier target = as.getTarget(); 
				Expression source = as.getSource();
				
				if (source instanceof FunctionCallIdentifier)			
					((FunctionCallIdentifier) source).validateExpression(dmlProg, ids.getVariables());
				else
					source.validateExpression(ids.getVariables());
				
				// Handle const vars: Basic Constant propagation 
				currConstVars.remove(target.getName());
				if (source instanceof ConstIdentifier && !(target instanceof IndexedIdentifier)){
					currConstVars.put(target.getName(), (ConstIdentifier)source);
				}
			
				if (source instanceof BuiltinFunctionExpression){
					BuiltinFunctionExpression bife = (BuiltinFunctionExpression)source;
					if ((bife.getOpCode() == Expression.BuiltinFunctionOp.NROW) ||
							(bife.getOpCode() == Expression.BuiltinFunctionOp.NCOL)){
						DataIdentifier id = (DataIdentifier)bife.getFirstExpr();
						DataIdentifier currVal = ids.getVariable(id.getName());
						if (currVal == null){
							throwUndefinedVar ( id.getName(), bife.toString() );
						}
						IntIdentifier intid = null;
						if (bife.getOpCode() == Expression.BuiltinFunctionOp.NROW){
							intid = new IntIdentifier((int)currVal.getDim1());
						} else {
							intid = new IntIdentifier((int)currVal.getDim2());
						}
						currConstVars.put(target.getName(), intid);
					}
				}
				// CASE: target NOT indexed identifier
				if (!(target instanceof IndexedIdentifier)){
					target.setProperties(source.getOutput());
				}
				// CASE: target is indexed identifier
				else{
					// process the "target" being indexed
					DataIdentifier targetAsSeen = ids.getVariable(target.getName());
					if (targetAsSeen == null)
						throw new LanguageException("cannot assign value to indexed identifier " + target.toString() + " without first initializing " + target.getName());
					target.setProperties(targetAsSeen);
					
					// process the expressions for the indexing
					if ( ((IndexedIdentifier)target).getRowLowerBound() != null  )
						((IndexedIdentifier)target).getRowLowerBound().validateExpression(ids.getVariables());
					if ( ((IndexedIdentifier)target).getRowUpperBound() != null  )
						((IndexedIdentifier)target).getRowUpperBound().validateExpression(ids.getVariables());
					if ( ((IndexedIdentifier)target).getColLowerBound() != null  )
						((IndexedIdentifier)target).getColLowerBound().validateExpression(ids.getVariables());
					if ( ((IndexedIdentifier)target).getColUpperBound() != null  )
						((IndexedIdentifier)target).getColUpperBound().validateExpression(ids.getVariables());
					
				}
				ids.addVariable(target.getName(), target);
				
			}
			
			else if (current instanceof MultiAssignmentStatement){
				MultiAssignmentStatement mas = (MultiAssignmentStatement) current;
				ArrayList<DataIdentifier> targetList = mas.getTargetList(); 
				
				// perform validation of source expression
				Expression source = mas.getSource();
				if (!(source instanceof FunctionCallIdentifier)){
					throw new LanguageException("can only use user-defined functions with multi-assignment statement");
				}
				else {
					FunctionCallIdentifier fci = (FunctionCallIdentifier)source;
					fci.validateExpression(dmlProg, ids.getVariables());
				}
				
		
				for (int j =0; j< targetList.size(); j++){
					
					// set target properties (based on type info in function call statement return params)
					DataIdentifier target = targetList.get(j);
					FunctionCallIdentifier fci = (FunctionCallIdentifier)source;
					FunctionStatement fstmt = (FunctionStatement)_dmlProg.getFunctionStatementBlock(fci.getNamespace(), fci.getName()).getStatement(0);
					if (!(target instanceof IndexedIdentifier)){
						target.setProperties(fstmt.getOutputParams().get(j));
					}
					else{
						DataIdentifier targetAsSeen = ids.getVariable(target.getName());
						if (targetAsSeen == null)
							throw new LanguageException("cannot assign value to indexed identifier " + target.toString() + " without first initializing " + target.getName());
						target.setProperties(targetAsSeen);
					}
					
					ids.addVariable(target.getName(), target);
				}
					
			}
			
			else if(current instanceof RandStatement)
			{
				RandStatement rs = (RandStatement) current;
				
				// perform constant propagation by replacing exprParams which are DataIdentifier (but NOT IndexedIdentifier) variables with constant values. 
				// Also perform "best-effort" validation of parameter values (i.e., for parameter values which are constant expressions) 
				rs.performConstantPropagation(currConstVars);
							
				
				
				// update properties of RandStatement target identifier
				rs.setIdentifierProperties();
				
				// add RandStatement target to available variables list
				ids.addVariable(rs.getIdentifier().getName(), rs.getIdentifier());
				
			}
				
			else if(current instanceof CVStatement || current instanceof ELStatement || 
					current instanceof ForStatement || current instanceof IfStatement || current instanceof WhileStatement ){
				throw new LanguageException("control statement (CVStatement, ELStatement, WhileStatement, IfStatement, ForStatement) should not be in genreric statement block.  Likely a parsing error");
			}
				
			else if (current instanceof PrintStatement){
				PrintStatement pstmt = (PrintStatement) current;
				Expression expr = pstmt.getExpression();	
				expr.validateExpression(ids.getVariables());
				
				// check that variables referenced in print statement expression are scalars
				if (expr.getOutput().getDataType() != Expression.DataType.SCALAR){
					throw new LanguageException("print statement can only print scalars");
				}
			}
			
			// no work to perform for PathStatement or ImportStatement
			else if (current instanceof PathStatement){}
			else if (current instanceof ImportStatement){}
			
			
			else {
				throw new LanguageException("cannot process statement of type " + current.getClass().getSimpleName());
			}
			
		} // end for (Statement current : _statements){
		_constVarsOut.putAll(currConstVars);
		return ids;

	}
	
	public void setStatementFormatType(IOStatement s) throws LanguageException, ParseException{
		if (s.getExprParam(Statement.FORMAT_TYPE)!= null ){
		 	
	 		Expression formatTypeExpr = s.getExprParam(Statement.FORMAT_TYPE);  
			if (!(formatTypeExpr instanceof StringIdentifier))
				throw new LanguageException("ERROR: input statement parameter " + Statement.FORMAT_TYPE 
						+ " can only be a string with one of following values: binary, text", 
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
	 		
			String ft = formatTypeExpr.toString();
			if (ft.equalsIgnoreCase("binary")){
				s._id.setFormatType(FormatType.BINARY);
			} else if (ft.equalsIgnoreCase("text")){
				s._id.setFormatType(FormatType.TEXT);
			} else throw new LanguageException("ERROR: input statement parameter " + Statement.FORMAT_TYPE 
					+ " can only be a string with one of following values: binary, text", 
					LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		} else {
			s.addExprParam(Statement.FORMAT_TYPE, new StringIdentifier(FormatType.TEXT.toString()));
			s._id.setFormatType(FormatType.TEXT);
		}
	}
	
	/**
	 * For each statement:
	 * 
	 * gen rule: for each variable read in current statement but not updated in any PRIOR statement, add to gen
	 * Handles case where variable both read and updated in same statement (i = i + 1, i needs to be added to gen)
	 * 
	 * kill rule:  for each variable updated in current statement but not read in this or any PRIOR statement,
	 * add to kill. 
	 *  
	 */
	public VariableSet initializeforwardLV(VariableSet activeIn) throws LanguageException {
		
		for (Statement s : _statements){
			s.initializeforwardLV(activeIn);
			VariableSet read = s.variablesRead();
			VariableSet updated = s.variablesUpdated();
			
			if (s instanceof WhileStatement || s instanceof IfStatement || s instanceof ForStatement){
				throw new LanguageException("control statement (while / for / if) cannot be in generic statement block");
			}
	
			if (read != null){
				// for each variable read in this statement but not updated in 
				// 		any prior statement, add to sb._gen
				
				for (String var : read.getVariableNames()) {
					if (!_updated.containsVariable(var)){
						_gen.addVariable(var, read.getVariable(var));
					}
				}
			}

			_read.addVariables(read);
			_updated.addVariables(updated);

			if (updated != null) {
				// for each updated variable that is not read
				for (String var : updated.getVariableNames()){
					if (!_read.containsVariable(var)) {
						_kill.addVariable(var, _updated.getVariable(var));
					}
				}
			}
		}
		_liveOut = new VariableSet();
		_liveOut.addVariables(activeIn);
		_liveOut.addVariables(_updated);
		return _liveOut;
	}
	
	
	public VariableSet initializebackwardLV(VariableSet loPassed) throws LanguageException{
		int numStatements = _statements.size();

		VariableSet lo = new VariableSet();
		lo.addVariables(loPassed);
		
		for (int i = numStatements-1; i>=0; i--){
			lo =  _statements.get(i).initializebackwardLV(lo);
		}
		
		VariableSet loReturn = new VariableSet();
		loReturn.addVariables(lo);
		return loReturn;
	}

	public HashMap<String, ConstIdentifier> getConstIn(){
		return _constVarsIn;
	}
	
	public HashMap<String, ConstIdentifier> getConstOut(){
		return _constVarsOut;
	}
	
	
	public VariableSet analyze(VariableSet loPassed) 
		throws LanguageException{
		
		VariableSet candidateLO = new VariableSet();
		candidateLO.addVariables(loPassed);
		//candidateLO.addVariables(_gen);
		
		VariableSet origLiveOut = new VariableSet();
		origLiveOut.addVariables(_liveOut);
		
		_liveOut = new VariableSet();
	 	for (String name : candidateLO.getVariableNames()){
	 		if (origLiveOut.containsVariable(name)){
	 			_liveOut.addVariable(name, candidateLO.getVariable(name));
	 		}
	 	}
	 	
		initializebackwardLV(_liveOut);
		
		_liveIn = new VariableSet();
		_liveIn.addVariables(_liveOut);
		_liveIn.removeVariables(_kill);
		_liveIn.addVariables(_gen);
			
		VariableSet liveInReturn = new VariableSet();
		liveInReturn.addVariables(_liveIn);
		return liveInReturn;
	}
	
}  // end class
