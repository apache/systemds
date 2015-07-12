/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.recompile.Recompiler;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.FormatType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.parser.LanguageException.LanguageErrorCodes;
import com.ibm.bi.dml.parser.PrintStatement.PRINTTYPE;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;


public class StatementBlock extends LiveVariableAnalysis
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected static final Log LOG = LogFactory.getLog(StatementBlock.class.getName());
	protected static IDSequence _seq = new IDSequence();
		
	protected DMLProgram _dmlProg; 
	protected ArrayList<Statement> _statements;
	ArrayList<Hop> _hops = null;
	ArrayList<Lop> _lops = null;
	HashMap<String,ConstIdentifier> _constVarsIn;
	HashMap<String,ConstIdentifier> _constVarsOut;
	
	private boolean _requiresRecompile = false;
	
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
	}
	
	public void setDMLProg(DMLProgram dmlProg){
		_dmlProg = dmlProg;
	}
	
	public DMLProgram getDMLProg(){
		return _dmlProg;
	}
	
	public void addStatement(Statement s){
		_statements.add(s);
		
		if (_statements.size() == 1){
			this._filename      = s.getFilename();
			this._beginLine 	= s.getBeginLine(); 
			this._beginColumn 	= s.getBeginColumn();
		}
		
		this._endLine 		= s.getEndLine();
		this._endColumn		= s.getEndColumn();
		
	}
	
	/**
	 * replace statement 
	 */
	public void replaceStatement(int index, Statement passedStmt){
		this._statements.set(index, passedStmt);
		
		if (index == 0){
			this._beginLine 	= passedStmt.getBeginLine(); 
			this._beginColumn 	= passedStmt.getBeginColumn();
		}
		
		else if (index == this._statements.size() -1){
			this._endLine 		= passedStmt.getEndLine();
			this._endColumn		= passedStmt.getEndColumn();	
		}
	}
	
	public void addStatementBlock(StatementBlock s){
		for (int i = 0; i < s.getNumStatements(); i++){
			_statements.add(s.getStatement(i));
		}
		
		this._beginLine 	= _statements.get(0).getBeginLine(); 
		this._beginColumn 	= _statements.get(0).getBeginColumn();
		
		this._endLine 		= _statements.get(_statements.size() - 1).getEndLine();
		this._endColumn		= _statements.get(_statements.size() - 1).getEndColumn();
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
	
	public void setStatements( ArrayList<Statement> s )
	{
		_statements = s;
	}

	public ArrayList<Hop> get_hops() throws HopsException {
		return _hops;
	}

	public ArrayList<Lop> getLops() {
		return _lops;
	}

	public void set_hops(ArrayList<Hop> hops) {
		_hops = hops;
	}

	public void setLops(ArrayList<Lop> lops) {
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
		
//    	if (DMLScript.ENABLE_DEBUG_MODE && !DMLScript.ENABLE_DEBUG_OPTIMIZER)
    	if (DMLScript.ENABLE_DEBUG_MODE)
			return false;
		
		// check whether targetIndex stmt block is for a mergable function call 
		Statement stmt = this.getStatement(0);
		
		// Check whether targetIndex block is: control stmt block or stmt block for un-mergable function call
		if (   stmt instanceof WhileStatement || stmt instanceof IfStatement || stmt instanceof ForStatement 
			|| stmt instanceof FunctionStatement || ( stmt instanceof PrintStatement && ((PrintStatement)stmt).getType() == PRINTTYPE.STOP )/*|| stmt instanceof ELStatement*/ )
		{
			return false;
		}
		
		if (stmt instanceof AssignmentStatement || stmt instanceof MultiAssignmentStatement){
			Expression sourceExpr = null;
			if (stmt instanceof AssignmentStatement) {
				AssignmentStatement astmt = (AssignmentStatement)stmt;
				// for now, ensure that an assignment statement containing a read from csv ends up in own statement block
				if(astmt.getSource().toString().contains(DataExpression.FORMAT_TYPE + "=" + DataExpression.FORMAT_TYPE_VALUE_CSV) && astmt.getSource().toString().contains("read"))
					return false;
				sourceExpr = astmt.getSource();
			}
			else
				sourceExpr = ((MultiAssignmentStatement)stmt).getSource();
			if ( sourceExpr instanceof BuiltinFunctionExpression && ((BuiltinFunctionExpression)sourceExpr).multipleReturns() )
				return false;
			
			//function calls (only mergable if inlined dml-bodied function)
			if (sourceExpr instanceof FunctionCallIdentifier){
				FunctionCallIdentifier fcall = (FunctionCallIdentifier) sourceExpr;
				FunctionStatementBlock fblock = dmlProg.getFunctionStatementBlock(fcall.getNamespace(), fcall.getName());
				if (fblock == null){
					LOG.error(sourceExpr.printErrorLocation() + "function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
					throw new LanguageException(sourceExpr.printErrorLocation() + "function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
				}
				if( !rIsInlineableFunction(fblock, dmlProg) )
					return false;
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
				if (fblock == null){
					LOG.error(sourceExpr.printErrorLocation() + "function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
					throw new LanguageException(sourceExpr.printErrorLocation() + "function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
				}
				
				if( rIsInlineableFunction(fblock, dmlProg) )
					return true;
			}
		}
		
		// regular statement
		return false;
	}
   

    /**
     * 
     * @param fblock
     * @param prog
     * @return
     */
    private boolean rIsInlineableFunction( FunctionStatementBlock fblock, DMLProgram prog )
    {
    	boolean ret = true;
    	
    	//reject external functions and function bodies with multiple blocks
    	if(    fblock.getStatements().isEmpty() //empty blocks
    		|| fblock.getStatement(0) instanceof ExternalFunctionStatement  
    		|| ((FunctionStatement)fblock.getStatement(0)).getBody().size() > 1 )
    	{
			return false;
		}
		
    	//reject control flow and non-inlinable functions
    	if(!fblock.getStatements().isEmpty() && !((FunctionStatement)fblock.getStatement(0)).getBody().isEmpty()) 
    	{
    		StatementBlock stmtBlock = ((FunctionStatement)fblock.getStatement(0)).getBody().get(0);
			
    		//reject control flow blocks
        	if (stmtBlock instanceof IfStatementBlock || stmtBlock instanceof WhileStatementBlock || stmtBlock instanceof ForStatementBlock)
				 return false;
        	
        	//recursively check that functions are inlinable
        	for( Statement s : stmtBlock.getStatements() ){
        		if( s instanceof AssignmentStatement && ((AssignmentStatement)s).getSource() instanceof FunctionCallIdentifier )
        		{
        			AssignmentStatement as = (AssignmentStatement)s;
        			FunctionCallIdentifier fcall = (FunctionCallIdentifier) as.getSource();
    				FunctionStatementBlock fblock2 = prog.getFunctionStatementBlock(fcall.getNamespace(), fcall.getName());
    				ret &= rIsInlineableFunction(fblock2, prog);
    				if( as.getSource().toString().contains(DataExpression.FORMAT_TYPE + "=" + DataExpression.FORMAT_TYPE_VALUE_CSV) && as.getSource().toString().contains("read"))
    					return false;
    				
    				if( !ret ) return false;
        		}
        		if( s instanceof MultiAssignmentStatement && ((MultiAssignmentStatement)s).getSource() instanceof FunctionCallIdentifier )
        		{
        			FunctionCallIdentifier fcall = (FunctionCallIdentifier) ((MultiAssignmentStatement)s).getSource();
    				FunctionStatementBlock fblock2 = prog.getFunctionStatementBlock(fcall.getNamespace(), fcall.getName());
    				ret &= rIsInlineableFunction(fblock2, prog);
    				if( !ret ) return false;
        		}
        	}
		}
    	
    	return ret;
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
		StringBuilder sb = new StringBuilder();
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

		if (sb == null || sb.isEmpty()) {
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
				if (fblock == null){
					LOG.error(fcall.printErrorLocation() + "function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
					throw new LanguageException(fcall.printErrorLocation() + "function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
				}
				FunctionStatement fstmt = (FunctionStatement)fblock.getStatement(0);
				
				// recursive inlining (no memo required because update-inplace of function statement blocks, so no redundant inlining)
				if( rIsInlineableFunction(fblock, dmlProg) ){
					fstmt.getBody().get(0).setStatements(rewriteFunctionCallStatements(dmlProg, fstmt.getBody().get(0).getStatements()));
				}
				
				//MB: we cannot use the hash since multiple interleaved inlined functions should be independent.
				//String prefix = new Integer(fblock.hashCode()).toString() + "_";
				String prefix = _seq.getNextID() + "_";
				
				if (fstmt.getBody().size() > 1){
					LOG.error(sourceExpr.printErrorLocation() + "rewritable function can only have 1 statement block");
					throw new LanguageException(sourceExpr.printErrorLocation() + "rewritable function can only have 1 statement block");
				}
				StatementBlock sblock = fstmt.getBody().get(0);
				
				for (int i =0; i < fstmt.getInputParams().size(); i++){
					
					DataIdentifier currFormalParam = fstmt.getInputParams().get(i);
					
					// create new assignment statement
					String newFormalParameterName = prefix + currFormalParam.getName();
					DataIdentifier newTarget = new DataIdentifier(currFormalParam);
					newTarget.setName(newFormalParameterName);
					
					Expression currCallParam = null;
					if (fcall.getParamExprs().size() > i){
						// function call has value for parameter
						currCallParam = fcall.getParamExprs().get(i).getExpr();
					}
					else {
						// use default value for parameter
						if (fstmt.getInputParams().get(i).getDefaultValue() == null){
							LOG.error(currFormalParam.printErrorLocation() + "default parameter for " + currFormalParam + " is undefined");
							throw new LanguageException(currFormalParam.printErrorLocation() + "default parameter for " + currFormalParam + " is undefined");
						}
						currCallParam = new DataIdentifier(fstmt.getInputParams().get(i).getDefaultValue());
						currCallParam.setAllPositions( 	fstmt.getInputParams().get(i).getFilename(),
														fstmt.getInputParams().get(i).getBeginLine(), 
														fstmt.getInputParams().get(i).getBeginColumn(),
														fstmt.getInputParams().get(i).getEndLine(),
														fstmt.getInputParams().get(i).getEndColumn());
					}
					
					//auto casting of inputs on inlining (if required) TODO discuss with Doug
					ValueType targetVT = newTarget.getValueType();
					if( newTarget.getDataType()==DataType.SCALAR && targetVT != currCallParam.getOutput().getValueType() && targetVT != ValueType.STRING ){
						currCallParam = new BuiltinFunctionExpression(BuiltinFunctionExpression.getValueTypeCastOperator(targetVT), new Expression[] {currCallParam}, 
												newTarget.getFilename(), newTarget.getBeginLine(), newTarget.getBeginColumn(), newTarget.getEndLine(), newTarget.getEndColumn());
						
					}
					
					// create the assignment statement to bind the call parameter to formal parameter
					AssignmentStatement binding = new AssignmentStatement(newTarget, currCallParam, newTarget.getBeginLine(), newTarget.getBeginColumn(), newTarget.getEndLine(), newTarget.getEndColumn());
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
						if (i > 0) {
							LOG.error(current.printErrorLocation() + "Assignment statement cannot return multiple values");
							throw new LanguageException(current.printErrorLocation() + "Assignment statement cannot return multiple values");
						}
						newTarget = new DataIdentifier(((AssignmentStatement)current).getTarget());
					}
					else{
						newTarget = new DataIdentifier(((MultiAssignmentStatement)current).getTargetList().get(i));
					}
					
					//auto casting of inputs on inlining (always, redundant cast removed during Hop Rewrites) TODO discuss with Doug
					ValueType sourceVT = newSource.getValueType();
					if( newSource.getDataType()==DataType.SCALAR && sourceVT != ValueType.STRING ){
						newSource = new BuiltinFunctionExpression(BuiltinFunctionExpression.getValueTypeCastOperator(sourceVT), new Expression[] {newSource},
								newTarget.getFilename(), newTarget.getBeginLine(), newTarget.getBeginColumn(), newTarget.getEndLine(), newTarget.getEndColumn());
					}
					
					// create the assignment statement to bind the call parameter to formal parameter
					AssignmentStatement binding = new AssignmentStatement(newTarget, newSource, newTarget.getBeginLine(), newTarget.getBeginColumn(), newTarget.getEndLine(), newTarget.getEndColumn());
					
					newStatements.add(binding);
				}
								
			} // end if (isRewritableFunctionCall(current, dmlProg)
				
			else {
				newStatements.add(current);
			}
		}
		
		return newStatements;
	}
	
	/**
	 * 
	 * @param dmlProg
	 * @param ids
	 * @param constVars
	 * @param conditional
	 * @return
	 * @throws LanguageException
	 * @throws ParseException
	 * @throws IOException
	 */
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String, ConstIdentifier> constVars, boolean conditional) 
		throws LanguageException, ParseException, IOException 
	{
		_constVarsIn.putAll(constVars);
		HashMap<String, ConstIdentifier> currConstVars = new HashMap<String,ConstIdentifier>();
		currConstVars.putAll(constVars);
			
		_statements = rewriteFunctionCallStatements(dmlProg, _statements);
		_dmlProg = dmlProg;
		
		for (Statement current : _statements){
						
			if (current instanceof OutputStatement){
				OutputStatement os = (OutputStatement)current;
				
				// validate variable being written by output statement exists
				DataIdentifier target = (DataIdentifier)os.getIdentifier();
				if (ids.getVariable(target.getName()) == null){
					raiseValidateError("Undefined Variable (" + target.getName() + ") used in statement", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
				}
				
				if ( ids.getVariable(target.getName()).getDataType() == DataType.SCALAR) {
					boolean paramsOkay = true;
					for (String key : os.getSource().getVarParams().keySet()){
						if (! (key.equals(DataExpression.IO_FILENAME) || key.equals(DataExpression.FORMAT_TYPE))) 
							paramsOkay = false;
					}
					if( !paramsOkay ) {
						raiseValidateError("Invalid parameters in write statement: " + os.toString(), conditional);
					}
				}
					
				Expression source = os.getSource();
				source.setOutput(target);
				source.validateExpression(ids.getVariables(), currConstVars, conditional);
				
				setStatementFormatType(os, conditional);
				target.setDimensionValueProperties(ids.getVariable(target.getName()));
			}
			
			else if (current instanceof AssignmentStatement){
				AssignmentStatement as = (AssignmentStatement)current;
				DataIdentifier target = as.getTarget(); 
			 	Expression source = as.getSource();
				
				if (source instanceof FunctionCallIdentifier)			
					((FunctionCallIdentifier) source).validateExpression(dmlProg, ids.getVariables(),currConstVars, conditional);
				else
					source.validateExpression(ids.getVariables(), currConstVars, conditional);
		
				if (source instanceof DataExpression && ((DataExpression)source).getOpCode() == Expression.DataOp.READ)
					setStatementFormatType(as, conditional);
				
				// Handle const vars: (a) basic constant propagation, and (b) transitive constant propagation over assignments 
				currConstVars.remove(target.getName());
				if(source instanceof ConstIdentifier && !(target instanceof IndexedIdentifier)){ //basic
					currConstVars.put(target.getName(), (ConstIdentifier)source);
				}
				if( source instanceof DataIdentifier && !(target instanceof IndexedIdentifier) ){ //transitive
					DataIdentifier diSource = (DataIdentifier) source;
					if( currConstVars.containsKey(diSource.getName()) ){
						currConstVars.put(target.getName(), currConstVars.get(diSource.getName()));
					}
				}
			
				if (source instanceof BuiltinFunctionExpression){
					BuiltinFunctionExpression bife = (BuiltinFunctionExpression)source;
					if ((bife.getOpCode() == Expression.BuiltinFunctionOp.NROW) ||
							(bife.getOpCode() == Expression.BuiltinFunctionOp.NCOL)){
						DataIdentifier id = (DataIdentifier)bife.getFirstExpr();
						DataIdentifier currVal = ids.getVariable(id.getName());
						if (currVal == null){
							//throwUndefinedVar ( id.getName(), bife.toString() );
							bife.raiseValidateError("Undefined Variable (" + id.getName() + ") used in statement", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
						}
						IntIdentifier intid = null;
						if (bife.getOpCode() == Expression.BuiltinFunctionOp.NROW){
							intid = new IntIdentifier(currVal.getDim1(), bife.getFilename(), bife.getBeginLine(), bife.getBeginColumn(), bife.getEndLine(), bife.getEndColumn());
						} else {
							intid = new IntIdentifier(currVal.getDim2(), bife.getFilename(), bife.getBeginLine(), bife.getBeginColumn(), bife.getEndLine(), bife.getEndColumn());
						}
						
						// handle case when nrow / ncol called on variable with size unknown (dims == -1) 
						//	--> const prop NOT possible 
						if (intid.getValue() != -1){
							currConstVars.put(target.getName(), intid);
						}
					}
				}
				// CASE: target NOT indexed identifier
				if (!(target instanceof IndexedIdentifier)){
					target.setProperties(source.getOutput());
					if (source.getOutput() instanceof IndexedIdentifier){
						target.setDimensions(source.getOutput().getDim1(), source.getOutput().getDim2());
					}
					
				}
				// CASE: target is indexed identifier
				else 
				{
					// process the "target" being indexed
					DataIdentifier targetAsSeen = ids.getVariable(target.getName());
					if (targetAsSeen == null){
						target.raiseValidateError("cannot assign value to indexed identifier " + target.toString() + " without first initializing " + target.getName(), conditional);
					}
					target.setProperties(targetAsSeen);
					
					// process the expressions for the indexing
					if ( ((IndexedIdentifier)target).getRowLowerBound() != null  )
						((IndexedIdentifier)target).getRowLowerBound().validateExpression(ids.getVariables(), currConstVars, conditional);
					if ( ((IndexedIdentifier)target).getRowUpperBound() != null  )
						((IndexedIdentifier)target).getRowUpperBound().validateExpression(ids.getVariables(), currConstVars, conditional);
					if ( ((IndexedIdentifier)target).getColLowerBound() != null  )
						((IndexedIdentifier)target).getColLowerBound().validateExpression(ids.getVariables(), currConstVars, conditional);
					if ( ((IndexedIdentifier)target).getColUpperBound() != null  )
						((IndexedIdentifier)target).getColUpperBound().validateExpression(ids.getVariables(), currConstVars, conditional);
					
					// validate that LHS indexed identifier is being assigned a matrix value
//					if (source.getOutput().getDataType() != Expression.DataType.MATRIX){
//						LOG.error(target.printErrorLocation() + "Indexed expression " + target.toString() + " can only be assigned matrix value");
//						throw new LanguageException(target.printErrorLocation() + "Indexed expression " + target.toString() + " can only be assigned matrix value");
//					}
					
					// validate that size of LHS index ranges is being assigned:
					//	(a) a matrix value of same size as LHS
					//	(b) singleton value (semantics: initialize enitre submatrix with this value)
					IndexPair targetSize = ((IndexedIdentifier)target).calculateIndexedDimensions(ids.getVariables(), currConstVars, conditional);
							
					if (targetSize._row >= 1 && source.getOutput().getDim1() > 1 && targetSize._row != source.getOutput().getDim1()){
						target.raiseValidateError("Dimension mismatch. Indexed expression " + target.toString() + " can only be assigned matrix with dimensions " 
								+ targetSize._row + " rows and " + targetSize._col + " cols. Attempted to assign matrix with dimensions " 
								+ source.getOutput().getDim1() + " rows and " + source.getOutput().getDim2() + " cols ", conditional);
					}
					
					if (targetSize._col >= 1 && source.getOutput().getDim2() > 1 && targetSize._col != source.getOutput().getDim2()){
						target.raiseValidateError("Dimension mismatch. Indexed expression " + target.toString() + " can only be assigned matrix with dimensions " 
								+ targetSize._row + " rows and " + targetSize._col + " cols. Attempted to assign matrix with dimensions " 
								+ source.getOutput().getDim1() + " rows and " + source.getOutput().getDim2() + " cols ", conditional);
					}
					
					((IndexedIdentifier)target).setDimensions(targetSize._row, targetSize._col);		
				}
				
				ids.addVariable(target.getName(), target);
				
			}
			
			else if (current instanceof MultiAssignmentStatement){
				MultiAssignmentStatement mas = (MultiAssignmentStatement) current;
				ArrayList<DataIdentifier> targetList = mas.getTargetList(); 
				
				// perform validation of source expression
				Expression source = mas.getSource();
				/*
				 * MultiAssignmentStatments currently supports only External, 
				 * User-defined, and Multi-return Builtin function expressions
				 */
				if (!(source instanceof DataIdentifier) 
						|| (source instanceof DataIdentifier && !((DataIdentifier)source).multipleReturns()) ) {
				//if (!(source instanceof FunctionCallIdentifier) ) {
						//|| !(source instanceof BuiltinFunctionExpression && ((BuiltinFunctionExpression)source).isMultiReturnBuiltinFunction()) ){
					source.raiseValidateError("can only use user-defined functions with multi-assignment statement", conditional);
				}
				
				if ( source instanceof FunctionCallIdentifier) {
					FunctionCallIdentifier fci = (FunctionCallIdentifier)source;
					fci.validateExpression(dmlProg, ids.getVariables(), currConstVars, conditional);
				}
				else if ( source instanceof BuiltinFunctionExpression && ((DataIdentifier)source).multipleReturns()) {
					source.validateExpression(mas, ids.getVariables(), currConstVars, conditional);
				}
				else 
					throw new LanguageException("Unexpected error.");
				
		
				if ( source instanceof FunctionCallIdentifier ) {
					for (int j =0; j< targetList.size(); j++){
						
						DataIdentifier target = targetList.get(j);
							// set target properties (based on type info in function call statement return params)
							FunctionCallIdentifier fci = (FunctionCallIdentifier)source;
							FunctionStatement fstmt = (FunctionStatement)_dmlProg.getFunctionStatementBlock(fci.getNamespace(), fci.getName()).getStatement(0);
							if (fstmt == null){
								fci.raiseValidateError(" function " + fci.getName() + " is undefined in namespace " + fci.getNamespace(), conditional);
							}
							if (!(target instanceof IndexedIdentifier)){
								target.setProperties(fstmt.getOutputParams().get(j));
							}
							else{
								DataIdentifier targetAsSeen = ids.getVariable(target.getName());
								if (targetAsSeen == null){
									raiseValidateError(target.printErrorLocation() + "cannot assign value to indexed identifier " + target.toString() + " without first initializing " + target.getName(), conditional);
								}
								target.setProperties(targetAsSeen);
							}
							ids.addVariable(target.getName(), target);
					}
				}
				else if ( source instanceof BuiltinFunctionExpression ) {
					Identifier[] outputs = source.getOutputs();
					for (int j=0; j < targetList.size(); j++) {
						ids.addVariable(targetList.get(j).getName(), (DataIdentifier)outputs[j]);
					}
				}
			}
				
			else if(current instanceof ForStatement || current instanceof IfStatement || current instanceof WhileStatement ){
				raiseValidateError("control statement (CVStatement, ELStatement, WhileStatement, IfStatement, ForStatement) should not be in genreric statement block.  Likely a parsing error", conditional);
			}
				
			else if (current instanceof PrintStatement){
				PrintStatement pstmt = (PrintStatement) current;
				Expression expr = pstmt.getExpression();	
				expr.validateExpression(ids.getVariables(), currConstVars, conditional);
				
				// check that variables referenced in print statement expression are scalars
				if (expr.getOutput().getDataType() != Expression.DataType.SCALAR){
				   raiseValidateError("print statement can only print scalars", conditional);
				}
			}
			
			// no work to perform for PathStatement or ImportStatement
			else if (current instanceof PathStatement){}
			else if (current instanceof ImportStatement){}
			
			
			else {
				raiseValidateError("cannot process statement of type " + current.getClass().getSimpleName(), conditional);
			}
			
		} // end for (Statement current : _statements){
		_constVarsOut.putAll(currConstVars);
		return ids;

	}
	
	public void setStatementFormatType(OutputStatement s, boolean conditionalValidate) 
		throws LanguageException, ParseException
	{
		if (s.getExprParam(DataExpression.FORMAT_TYPE)!= null ){
		 	
	 		Expression formatTypeExpr = s.getExprParam(DataExpression.FORMAT_TYPE);  
			if (!(formatTypeExpr instanceof StringIdentifier)){
				raiseValidateError("IO statement parameter " + DataExpression.FORMAT_TYPE 
						+ " can only be a string with one of following values: binary, text", conditionalValidate, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			String ft = formatTypeExpr.toString();
			if (ft.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_BINARY)){
				s.getIdentifier().setFormatType(FormatType.BINARY);
			} else if (ft.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_TEXT)){
				s.getIdentifier().setFormatType(FormatType.TEXT);
			} else if (ft.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_MATRIXMARKET)){
				s.getIdentifier().setFormatType(FormatType.MM);
			} else if (ft.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_CSV)){
				s.getIdentifier().setFormatType(FormatType.CSV);
			} else{ 
				raiseValidateError("IO statement parameter " + DataExpression.FORMAT_TYPE 
						+ " can only be a string with one of following values: binary, text, mm, csv", conditionalValidate, LanguageErrorCodes.INVALID_PARAMETERS);
			}
		} else {
			s.addExprParam(DataExpression.FORMAT_TYPE, 
					new StringIdentifier(FormatType.TEXT.toString(), s.getFilename(), s.getBeginLine(), s.getBeginColumn(), s.getEndLine(), s.getEndColumn()),
					true);
			s.getIdentifier().setFormatType(FormatType.TEXT);
		}
	}
	
	public void setStatementFormatType(AssignmentStatement s, boolean conditionalValidate) 
		throws LanguageException, ParseException
	{
		
		if (!(s.getSource() instanceof DataExpression))
			return;
		DataExpression dataExpr = (DataExpression)s.getSource();
		
		if (dataExpr.getVarParam(DataExpression.FORMAT_TYPE)!= null ){
		 	
	 		Expression formatTypeExpr = dataExpr.getVarParam(DataExpression.FORMAT_TYPE);  
			if (!(formatTypeExpr instanceof StringIdentifier)){
				raiseValidateError("IO statement parameter " + DataExpression.FORMAT_TYPE 
						+ " can only be a string with one of following values: binary, text", conditionalValidate, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			String ft = formatTypeExpr.toString();
			if (ft.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_BINARY)){
				s.getTarget().setFormatType(FormatType.BINARY);
			} else if (ft.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_TEXT)){
				s.getTarget().setFormatType(FormatType.TEXT);
			} else if (ft.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_MATRIXMARKET)){
				s.getTarget().setFormatType(FormatType.MM);
			} else if (ft.equalsIgnoreCase(DataExpression.FORMAT_TYPE_VALUE_CSV)){
				s.getTarget().setFormatType(FormatType.CSV);
			} else{ 
				raiseValidateError("IO statement parameter " + DataExpression.FORMAT_TYPE 
						+ " can only be a string with one of following values: binary, text, mm, csv", conditionalValidate, LanguageErrorCodes.INVALID_PARAMETERS);
			}
		} else {
			dataExpr.addVarParam(DataExpression.FORMAT_TYPE, new StringIdentifier(FormatType.TEXT.toString(),
					dataExpr.getFilename(), dataExpr.getBeginLine(), dataExpr.getBeginColumn(), dataExpr.getEndLine(), dataExpr.getEndColumn()));
			s.getTarget().setFormatType(FormatType.TEXT);
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
				LOG.error(s.printErrorLocation() + "control statement (while / for / if) cannot be in generic statement block");
				throw new LanguageException(s.printErrorLocation() + "control statement (while / for / if) cannot be in generic statement block");
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
	
	///////////////////////////////////////////////////////////////
	// validate error handling (consistent for all expressions)
	
	/**
	* 
	* @param msg
	* @param conditional
	* @throws LanguageException
	*/
	public void raiseValidateError( String msg, boolean conditional ) 
		throws LanguageException
	{
		raiseValidateError(msg, conditional, null);
	}
	
	/**
	* 
	* @param msg
	* @param conditional
	* @param code
	* @throws LanguageException
	*/
	public void raiseValidateError( String msg, boolean conditional, String errorCode ) 
		throws LanguageException
	{
		if( conditional )  //warning if conditional
		{
			String fullMsg = this.printWarningLocation() + msg;
			
			LOG.warn( fullMsg );
		}
		else  //error and exception if unconditional
		{
			String fullMsg = this.printErrorLocation() + msg;
			
			LOG.error( fullMsg );			
			if( errorCode != null )
				throw new LanguageException( fullMsg, errorCode );
			else 
				throw new LanguageException( fullMsg );
		}
	}

	///////////////////////////////////////////////////////////////////////////
	// store position information for statement blocks
	///////////////////////////////////////////////////////////////////////////
	private String _filename = "MAIN SCRIPT";
	private int _beginLine = 0, _beginColumn = 0;
	private int _endLine = 0, _endColumn = 0;
	
	public void setFilename (String fname)  { _filename = fname;	}
	public void setBeginLine(int passed)    { _beginLine = passed;  }
	public void setBeginColumn(int passed) 	{ _beginColumn = passed; }
	public void setEndLine(int passed) 		{ _endLine = passed;   }
	public void setEndColumn(int passed)	{ _endColumn = passed; }
	
	public void setAllPositions(String fname, int blp, int bcp, int elp, int ecp){
		_filename    = fname;
		_beginLine	 = blp; 
		_beginColumn = bcp; 
		_endLine 	 = elp;
		_endColumn 	 = ecp;
	}

	public String getFilename() { return _filename;	   }
	public int getBeginLine()	{ return _beginLine;   }
	public int getBeginColumn() { return _beginColumn; }
	public int getEndLine() 	{ return _endLine;   }
	public int getEndColumn()	{ return _endColumn; }
	
	public String printErrorLocation(){
		return "ERROR: " + _filename + " -- line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	public String printBlockErrorLocation(){
		return "ERROR: "  + _filename + " -- statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
	
	public String printWarningLocation(){
		return "WARNING: " + _filename + " -- line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	

	/////////
	// materialized hops recompilation flags
	////
	
	public void updateRecompilationFlag() 
		throws HopsException
	{
		_requiresRecompile =   OptimizerUtils.ALLOW_DYN_RECOMPILATION 
			                   && OptimizerUtils.isHybridExecutionMode() 	
			                   && Recompiler.requiresRecompilation(get_hops());
	}
	
	public boolean requiresRecompilation()
	{
		return _requiresRecompile;
	}
	
	
}  // end class
