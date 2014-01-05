/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.compile.Recompiler;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.FormatType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;


public class StatementBlock extends LiveVariableAnalysis
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
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

	public ArrayList<Hop> get_hops() throws HopsException {
		return _hops;
	}

	public ArrayList<Lop> get_lops() {
		return _lops;
	}

	public void set_hops(ArrayList<Hop> hops) {
		_hops = hops;
	}

	public void set_lops(ArrayList<Lop> lops) {
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
		if (   stmt instanceof WhileStatement || stmt instanceof IfStatement || stmt instanceof ForStatement 
			|| stmt instanceof FunctionStatement || stmt instanceof CVStatement /*|| stmt instanceof ELStatement*/ )
		{
			return false;
		}
		
		// for regular stmt block, check if this is a function call stmt block
		if (stmt instanceof AssignmentStatement || stmt instanceof MultiAssignmentStatement){
			Expression sourceExpr = null;
			if (stmt instanceof AssignmentStatement) {
				AssignmentStatement astmt = (AssignmentStatement)stmt;
				if( astmt.containsIndividualStatementBlockOperations() )
					return false;
				sourceExpr = astmt.getSource();
			}
			else
				sourceExpr = ((MultiAssignmentStatement)stmt).getSource();
			if ( sourceExpr instanceof BuiltinFunctionExpression && ((BuiltinFunctionExpression)sourceExpr).multipleReturns() )
				return false;
			
			if (sourceExpr instanceof FunctionCallIdentifier){
				FunctionCallIdentifier fcall = (FunctionCallIdentifier) sourceExpr;
				FunctionStatementBlock fblock = dmlProg.getFunctionStatementBlock(fcall.getNamespace(), fcall.getName());
				if (fblock == null){
					LOG.error(sourceExpr.printErrorLocation() + "function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
					throw new LanguageException(sourceExpr.printErrorLocation() + "function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
				}
				if (fblock.getStatements().size() > 0 && fblock.getStatement(0) instanceof ExternalFunctionStatement  ||  ((FunctionStatement)fblock.getStatement(0)).getBody().size() > 1 ){
					return false;
				}
				else {
					 // check if statement block is a control block
					 if (fblock.getStatements().size() > 0 && ((FunctionStatement)fblock.getStatement(0)).getBody().size() > 0){
						 StatementBlock stmtBlock = ((FunctionStatement)fblock.getStatement(0)).getBody().get(0);
						 if (stmtBlock instanceof IfStatementBlock || stmtBlock instanceof WhileStatementBlock || stmtBlock instanceof ForStatementBlock){
							 return false;
						 }
						 else {
							 return true; 
						 }
					 }
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
				if (fblock == null){
					LOG.error(sourceExpr.printErrorLocation() + "function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
					throw new LanguageException(sourceExpr.printErrorLocation() + "function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
				}
				if (fblock.getStatement(0) instanceof ExternalFunctionStatement  ||  ((FunctionStatement)fblock.getStatement(0)).getBody().size() > 1){
					return false;
				}
				else {
					// check if statement block is a control block
					if (fblock.getStatements().size() > 0 && ((FunctionStatement)fblock.getStatement(0)).getBody().size() > 0){
						StatementBlock stmtBlock = ((FunctionStatement)fblock.getStatement(0)).getBody().get(0);
						if (stmtBlock instanceof IfStatementBlock || stmtBlock instanceof WhileStatementBlock || stmtBlock instanceof ForStatementBlock)
							return false;
						else
							return true;
					}
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
				if (fblock == null){
					LOG.error(sourceExpr.printErrorLocation() + "function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
					throw new LanguageException(sourceExpr.printErrorLocation() + "function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
				}
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
				if (fblock == null){
					LOG.error(fcall.printErrorLocation() + "function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
					throw new LanguageException(fcall.printErrorLocation() + "function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
				}
				FunctionStatement fstmt = (FunctionStatement)fblock.getStatement(0);
				
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
					if (fcall.getParamExpressions().size() > i){
						// function call has value for parameter
						currCallParam = fcall.getParamExpressions().get(i);
					}
					else {
						// use default value for parameter
						if (fstmt.getInputParams().get(i).getDefaultValue() == null){
							LOG.error(currFormalParam.printErrorLocation() + "default parameter for " + currFormalParam + " is undefined");
							throw new LanguageException(currFormalParam.printErrorLocation() + "default parameter for " + currFormalParam + " is undefined");
						}
						currCallParam = new DataIdentifier(fstmt.getInputParams().get(i).getDefaultValue());
						currCallParam.setAllPositions( fstmt.getInputParams().get(i).getBeginLine(), 
														fstmt.getInputParams().get(i).getBeginColumn(),
														fstmt.getInputParams().get(i).getEndLine(),
														fstmt.getInputParams().get(i).getEndColumn());
					}
					
					// create the assignment statement to bind the call parameter to formal parameter
					AssignmentStatement binding = new AssignmentStatement(newTarget, currCallParam, newTarget._beginLine, newTarget._beginColumn, newTarget._endLine, newTarget._endColumn);
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
					// create the assignment statement to bind the call parameter to formal parameter
					AssignmentStatement binding = new AssignmentStatement(newTarget, newSource, newTarget._beginLine, newTarget._beginColumn, newTarget._endLine, newTarget._endColumn);
					
					newStatements.add(binding);
				}
								
			} // end if (isRewritableFunctionCall(current, dmlProg)
				
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
					if (targetAsSeen == null){
						LOG.error(target.printErrorLocation() + "cannot assign value to indexed identifier " + target.toString() + " without initializing " + is.getIdentifier().getName());
						throw new LanguageException(target.printErrorLocation() + "cannot assign value to indexed identifier " + target.toString() + " without initializing " + is.getIdentifier().getName());
					}
					target.setProperties(targetAsSeen);
				}
							
				ids.addVariable(target.getName(),target);
			}
			
			else if (current instanceof OutputStatement){
				OutputStatement os = (OutputStatement)current;
				
				// validate variable being written by output statement exists
				DataIdentifier target = (DataIdentifier)os.getIdentifier();
				if (ids.getVariable(target.getName()) == null){
					//throwUndefinedVar ( target.getName(), os );
					
					LOG.error(os.printErrorLocation() + "Undefined Variable (" + target.getName() + ") used in statement");
					throw new LanguageException(os.printErrorLocation() + "Undefined Variable (" + target.getName() + ") used in statement",
							LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
				}
				
				if ( ids.getVariable(target.getName()).getDataType() == DataType.SCALAR) {
					boolean paramsOkay = true;
					for (String key : os._paramsExpr.getVarParams().keySet()){
						if (! (key.equals(Statement.IO_FILENAME) || key.equals(Statement.FORMAT_TYPE))) 
							paramsOkay = false;
					}
					if (paramsOkay == false){
						LOG.error(os.printErrorLocation() + "Invalid parameters in write statement: " + os.toString());
						throw new LanguageException(os.printErrorLocation() + "Invalid parameters in write statement: " + os.toString());
					}
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
					((FunctionCallIdentifier) source).validateExpression(dmlProg, ids.getVariables(),currConstVars);
				else
					source.validateExpression(ids.getVariables(), currConstVars);
				
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
							//throwUndefinedVar ( id.getName(), bife.toString() );
							LOG.error(bife.printErrorLocation() + "Undefined Variable (" + id.getName() + ") used in statement");
							throw new LanguageException(bife.printErrorLocation() + "Undefined Variable (" + id.getName() + ") used in statement",
									LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
						}
						IntIdentifier intid = null;
						if (bife.getOpCode() == Expression.BuiltinFunctionOp.NROW){
							intid = new IntIdentifier((int)currVal.getDim1());
						} else {
							intid = new IntIdentifier((int)currVal.getDim2());
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
				else {
					// process the "target" being indexed
					DataIdentifier targetAsSeen = ids.getVariable(target.getName());
					if (targetAsSeen == null){
						LOG.error(target.printErrorLocation() + "cannot assign value to indexed identifier " + target.toString() + " without first initializing " + target.getName());
						throw new LanguageException(target.printErrorLocation() + "cannot assign value to indexed identifier " + target.toString() + " without first initializing " + target.getName());
					}
					target.setProperties(targetAsSeen);
					
					// process the expressions for the indexing
					if ( ((IndexedIdentifier)target).getRowLowerBound() != null  )
						((IndexedIdentifier)target).getRowLowerBound().validateExpression(ids.getVariables(), currConstVars);
					if ( ((IndexedIdentifier)target).getRowUpperBound() != null  )
						((IndexedIdentifier)target).getRowUpperBound().validateExpression(ids.getVariables(), currConstVars);
					if ( ((IndexedIdentifier)target).getColLowerBound() != null  )
						((IndexedIdentifier)target).getColLowerBound().validateExpression(ids.getVariables(), currConstVars);
					if ( ((IndexedIdentifier)target).getColUpperBound() != null  )
						((IndexedIdentifier)target).getColUpperBound().validateExpression(ids.getVariables(), currConstVars);
					
					// validate that LHS indexed identifier is being assigned a matrix value
//					if (source.getOutput().getDataType() != Expression.DataType.MATRIX){
//						LOG.error(target.printErrorLocation() + "Indexed expression " + target.toString() + " can only be assigned matrix value");
//						throw new LanguageException(target.printErrorLocation() + "Indexed expression " + target.toString() + " can only be assigned matrix value");
//					}
					
					// validate that size of LHS index ranges is being assigned:
					//	(a) a matrix value of same size as LHS
					//	(b) singleton value (semantics: initialize enitre submatrix with this value)
					IndexPair targetSize = ((IndexedIdentifier)target).calculateIndexedDimensions(ids.getVariables(), currConstVars);
							
					if (targetSize._row >= 1 && source.getOutput().getDim1() > 1 && targetSize._row != source.getOutput().getDim1()){
						
						LOG.error(target.printErrorLocation() + "Dimension mismatch. Indexed expression " + target.toString() + " can only be assigned matrix with dimensions " 
								+ targetSize._row + " rows and " + targetSize._col + " cols. Attempted to assign matrix with dimensions " 
								+ source.getOutput().getDim1() + " rows and " + source.getOutput().getDim2() + " cols " );
						
						throw new LanguageException(target.printErrorLocation() + "Dimension mismatch. Indexed expression " + target.toString() + " can only be assigned matrix with dimensions " 
										+ targetSize._row + " rows and " + targetSize._col + " cols. Attempted to assign matrix with dimensions " 
										+ source.getOutput().getDim1() + " rows and " + source.getOutput().getDim2() + " cols " );
					}
					
					if (targetSize._col >= 1 && source.getOutput().getDim2() > 1 && targetSize._col != source.getOutput().getDim2()){
						
						LOG.error(target.printErrorLocation() + "Dimension mismatch. Indexed expression " + target.toString() + " can only be assigned matrix with dimensions " 
								+ targetSize._row + " rows and " + targetSize._col + " cols. Attempted to assign matrix with dimensions " 
								+ source.getOutput().getDim1() + " rows and " + source.getOutput().getDim2() + " cols " );
						
						throw new LanguageException(target.printErrorLocation() + "Dimension mismatch. Indexed expression " + target.toString() + " can only be assigned matrix with dimensions " 
										+ targetSize._row + " rows and " + targetSize._col + " cols. Attempted to assign matrix with dimensions " 
										+ source.getOutput().getDim1() + " rows and " + source.getOutput().getDim2() + " cols " );
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
					LOG.error(source.printErrorLocation() + "can only use user-defined functions with multi-assignment statement");
					throw new LanguageException(source.printErrorLocation() + "can only use user-defined functions with multi-assignment statement");
				}
				
				if ( source instanceof FunctionCallIdentifier) {
					FunctionCallIdentifier fci = (FunctionCallIdentifier)source;
					fci.validateExpression(dmlProg, ids.getVariables(), currConstVars);
				}
				else if ( source instanceof BuiltinFunctionExpression && ((DataIdentifier)source).multipleReturns()) {
					source.validateExpression(mas, ids.getVariables(), currConstVars);
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
								LOG.error(fci.printErrorLocation() + " function " + fci.getName() + " is undefined in namespace " + fci.getNamespace());
								throw new LanguageException(fci.printErrorLocation() + " function " + fci.getName() + " is undefined in namespace " + fci.getNamespace());
							}
							if (!(target instanceof IndexedIdentifier)){
								target.setProperties(fstmt.getOutputParams().get(j));
							}
							else{
								DataIdentifier targetAsSeen = ids.getVariable(target.getName());
								if (targetAsSeen == null){
									LOG.error(target.printErrorLocation() + "cannot assign value to indexed identifier " + target.toString() + " without first initializing " + target.getName());
									throw new LanguageException(target.printErrorLocation() + "cannot assign value to indexed identifier " + target.toString() + " without first initializing " + target.getName());
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
			else if(current instanceof RandStatement)
			{
				RandStatement rs = (RandStatement) current;
				
				DataIdentifier target = rs.getIdentifier(); 
				Expression source = rs.getSource();
				source.setOutput(target);
				
				// validate Rand Statement
				source.validateExpression(ids.getVariables(), currConstVars);
				
				
				
				// use existing size and properties information for LHS IndexedIdentifier
				// Do we want to support this? if not, throw an exception, if yes, copy from Assignment part 
				// CASE: target NOT indexed identifier
				if (!(target instanceof IndexedIdentifier)){
					target.setProperties(source.getOutput());
					
					if (source.getOutput() instanceof IndexedIdentifier){
						target.setDimensions(source.getOutput().getDim1(), source.getOutput().getDim2());
						rs.getIdentifier().setDimensions(source.getOutput().getDim1(), source.getOutput().getDim2());
					}
					
				}
				// CASE: target is indexed identifier
				else {
					// process the "target" being indexed
					DataIdentifier targetAsSeen = ids.getVariable(target.getName());
					if (targetAsSeen == null){
						LOG.error(target.printErrorLocation() + "cannot assign value to indexed identifier " + target.toString() + " without first initializing " + target.getName());
						throw new LanguageException(target.printErrorLocation() + "cannot assign value to indexed identifier " + target.toString() + " without first initializing " + target.getName());
					}
					
					//target.setProperties(targetAsSeen);
					
					// process the expressions for the indexing
					if ( ((IndexedIdentifier)target).getRowLowerBound() != null  )
						((IndexedIdentifier)target).getRowLowerBound().validateExpression(ids.getVariables(), currConstVars);
					if ( ((IndexedIdentifier)target).getRowUpperBound() != null  )
						((IndexedIdentifier)target).getRowUpperBound().validateExpression(ids.getVariables(), currConstVars);
					if ( ((IndexedIdentifier)target).getColLowerBound() != null  )
						((IndexedIdentifier)target).getColLowerBound().validateExpression(ids.getVariables(), currConstVars);
					if ( ((IndexedIdentifier)target).getColUpperBound() != null  )
						((IndexedIdentifier)target).getColUpperBound().validateExpression(ids.getVariables(), currConstVars);
					
					// validate that LHS indexed identifier is being assigned a matrix value
					if (source.getOutput().getDataType() != Expression.DataType.MATRIX){
						LOG.error(target.printErrorLocation() + "Indexed expression " + target.toString() + " can only be assigned matrix value");
						throw new LanguageException(target.printErrorLocation() + "Indexed expression " + target.toString() + " can only be assigned matrix value");
					}
					
					// validate that size of LHS index ranges is being assigned:
					//	(a) a matrix value of same size as LHS
					//	(b) singleton value (semantics: initialize enitre submatrix with this value)
					IndexPair targetSize = ((IndexedIdentifier)target).calculateIndexedDimensions(ids.getVariables(), currConstVars);
							
					if (targetSize._row >= 1 && source.getOutput().getDim1() > 1 && targetSize._row != source.getOutput().getDim1()){
						
						LOG.error(target.printErrorLocation() + "Dimension mismatch. Indexed expression " + target.toString() + " can only be assigned matrix with dimensions " 
								+ targetSize._row + " rows and " + targetSize._col + " cols. Attempted to assign matrix with dimensions " 
								+ source.getOutput().getDim1() + " rows and " + source.getOutput().getDim2() + " cols " );
						
						throw new LanguageException(target.printErrorLocation() + "Dimension mismatch. Indexed expression " + target.toString() + " can only be assigned matrix with dimensions " 
										+ targetSize._row + " rows and " + targetSize._col + " cols. Attempted to assign matrix with dimensions " 
										+ source.getOutput().getDim1() + " rows and " + source.getOutput().getDim2() + " cols " );
					}
					
					if (targetSize._col >= 1 && source.getOutput().getDim2() > 1 && targetSize._col != source.getOutput().getDim2()){
					
						LOG.error(target.printErrorLocation() + "Dimension mismatch. Indexed expression " + target.toString() + " can only be assigned matrix with dimensions " 
								+ targetSize._row + " rows and " + targetSize._col + " cols. Attempted to assign matrix with dimensions " 
								+ source.getOutput().getDim1() + " rows and " + source.getOutput().getDim2() + " cols " );
						
						throw new LanguageException(target.printErrorLocation() + "Dimension mismatch. Indexed expression " + target.toString() + " can only be assigned matrix with dimensions " 
										+ targetSize._row + " rows and " + targetSize._col + " cols. Attempted to assign matrix with dimensions " 
										+ source.getOutput().getDim1() + " rows and " + source.getOutput().getDim2() + " cols " );
					}
					
					((IndexedIdentifier)target).setDimensions(targetSize._row, targetSize._col);
					
						
				}
				
				// add RandStatement target to available variables list
				ids.addVariable(target.getName(),target);
			
			}
				
			else if(current instanceof CVStatement /*|| current instanceof ELStatement*/ 
					|| current instanceof ForStatement || current instanceof IfStatement || current instanceof WhileStatement ){
				LOG.error(current.printErrorLocation() + "control statement (CVStatement, ELStatement, WhileStatement, IfStatement, ForStatement) should not be in genreric statement block.  Likely a parsing error");
				throw new LanguageException(current.printErrorLocation() + "control statement (CVStatement, ELStatement, WhileStatement, IfStatement, ForStatement) should not be in genreric statement block.  Likely a parsing error");
			}
				
			else if (current instanceof PrintStatement){
				PrintStatement pstmt = (PrintStatement) current;
				Expression expr = pstmt.getExpression();	
				expr.validateExpression(ids.getVariables(), currConstVars);
				
				// check that variables referenced in print statement expression are scalars
				if (expr.getOutput().getDataType() != Expression.DataType.SCALAR){
					LOG.error(current.printErrorLocation() + "print statement can only print scalars");
					throw new LanguageException(current.printErrorLocation() + "print statement can only print scalars");
				}
			}
			
			// no work to perform for PathStatement or ImportStatement
			else if (current instanceof PathStatement){}
			else if (current instanceof ImportStatement){}
			
			
			else {
				LOG.error(current.printErrorLocation() + "cannot process statement of type " + current.getClass().getSimpleName());
				throw new LanguageException(current.printErrorLocation() + "cannot process statement of type " + current.getClass().getSimpleName());
			}
			
		} // end for (Statement current : _statements){
		_constVarsOut.putAll(currConstVars);
		return ids;

	}
	
	public void setStatementFormatType(IOStatement s) throws LanguageException, ParseException{
		if (s.getExprParam(Statement.FORMAT_TYPE)!= null ){
		 	
	 		Expression formatTypeExpr = s.getExprParam(Statement.FORMAT_TYPE);  
			if (!(formatTypeExpr instanceof StringIdentifier)){
				
				LOG.error(s.printErrorLocation() + "IO statement parameter " + Statement.FORMAT_TYPE 
						+ " can only be a string with one of following values: binary, text");
				
				throw new LanguageException(s.printErrorLocation() + "IO statement parameter " + Statement.FORMAT_TYPE 
						+ " can only be a string with one of following values: binary, text", 
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			String ft = formatTypeExpr.toString();
			if (ft.equalsIgnoreCase(Statement.FORMAT_TYPE_VALUE_BINARY)){
				s._id.setFormatType(FormatType.BINARY);
			} else if (ft.equalsIgnoreCase(Statement.FORMAT_TYPE_VALUE_TEXT)){
				s._id.setFormatType(FormatType.TEXT);
			} else if (ft.equalsIgnoreCase(Statement.FORMAT_TYPE_VALUE_MATRIXMARKET)){
				s._id.setFormatType(FormatType.MM);
			} else if (ft.equalsIgnoreCase(Statement.FORMAT_TYPE_VALUE_CSV)){
				s._id.setFormatType(FormatType.CSV);
			} else{ 
				
				LOG.error(s.printErrorLocation() + "IO statement parameter " + Statement.FORMAT_TYPE 
						+ " can only be a string with one of following values: binary, text, mm, csv");
				
				throw new LanguageException(s.printErrorLocation() + "IO statement parameter " + Statement.FORMAT_TYPE 
					+ " can only be a string with one of following values: binary, text, mm, csv", 
					LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
		} else {
			s.addExprParam(Statement.FORMAT_TYPE, new StringIdentifier(FormatType.TEXT.toString()),true);
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

	///////////////////////////////////////////////////////////////////////////
	// store position information for statement blocks
	///////////////////////////////////////////////////////////////////////////
	public int _beginLine = 0, _beginColumn = 0;
	public int _endLine = 0, _endColumn = 0;
	
	public void setBeginLine(int passed)    { _beginLine = passed;   }
	public void setBeginColumn(int passed) 	{ _beginColumn = passed; }
	public void setEndLine(int passed) 		{ _endLine = passed;   }
	public void setEndColumn(int passed)	{ _endColumn = passed; }
	
	public void setAllPositions(int blp, int bcp, int elp, int ecp){
		_beginLine	 = blp; 
		_beginColumn = bcp; 
		_endLine 	 = elp;
		_endColumn 	 = ecp;
	}

	public int getBeginLine()	{ return _beginLine;   }
	public int getBeginColumn() { return _beginColumn; }
	public int getEndLine() 	{ return _endLine;   }
	public int getEndColumn()	{ return _endColumn; }
	
	public String printErrorLocation(){
		return "ERROR: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	public String printBlockErrorLocation(){
		return "ERROR: statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
	
	public String printWarningLocation(){
		return "WARNING: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	/**
	 * MB: This method was used to remove updated vars from constant propagation when
	 * live-variable-analysis was executed AFTER validate. Since now, we execute
	 * live-variable-analysis BEFORE validate, this is redundant and should not be used anymore.
	 * 
	 * @param asb
	 * @param upVars
	 */
	@Deprecated
	public void rFindUpdatedVariables( ArrayList<StatementBlock> asb, HashSet<String> upVars )
	{
		for(StatementBlock sb : asb ) // foreach statementblock
			for( Statement s : sb._statements ) // foreach statement in statement block
			{
				if( s instanceof ForStatement || s instanceof ParForStatement )
				{
					rFindUpdatedVariables(((ForStatement)s).getBody(), upVars);
				}
				else if( s instanceof WhileStatement ) 
				{
					rFindUpdatedVariables(((WhileStatement)s).getBody(), upVars);
				}
				else if( s instanceof IfStatement ) 
				{
					rFindUpdatedVariables(((IfStatement)s).getIfBody(), upVars);
					rFindUpdatedVariables(((IfStatement)s).getElseBody(), upVars);
				}
				else if( s instanceof FunctionStatement ) 
				{
					rFindUpdatedVariables(((FunctionStatement)s).getBody(), upVars);
				}
				else
				{
					//evaluate assignment statements
					Collection<DataIdentifier> tmp = null; 
					if( s instanceof AssignmentStatement )
					{
						tmp = ((AssignmentStatement)s).getTargetList();	
					}
					else if (s instanceof FunctionStatement)
					{
						tmp = ((FunctionStatement)s).getOutputParams();
					}
					else if (s instanceof MultiAssignmentStatement)
					{
						tmp = ((MultiAssignmentStatement)s).getTargetList();
					}
					/* FIXME at Doug
					else if (s instanceof RandStatement)
					{
						tmp = new ArrayList<DataIdentifier>();
						tmp.add(((RandStatement)s).getIdentifier());
					}*/
					
					//add names of updated data identifiers to results
					if( tmp!=null )
						for( DataIdentifier di : tmp )
							upVars.add( di.getName() );
				}
			}
	}

	/////////
	// materialized hops recompilation flags
	////
	
	public void updateRecompilationFlag() 
		throws HopsException
	{
		_requiresRecompile =   OptimizerUtils.ALLOW_DYN_RECOMPILATION 
			                   && DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID	
			                   && Recompiler.requiresRecompilation(get_hops());
	}
	
	public boolean requiresRecompilation()
	{
		return _requiresRecompile;
	}
	
	
}  // end class
