/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.parser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.sysds.parser.Expression;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.hops.rewrite.StatementBlockRewriteRule;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.common.Builtins;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.parser.LanguageException.LanguageErrorCodes;
import org.apache.sysds.parser.PrintStatement.PRINTTYPE;
import org.apache.sysds.parser.dml.DmlSyntacticValidator;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.utils.MLContextProxy;


public class StatementBlock extends LiveVariableAnalysis implements ParseInfo
{
	protected static final Log LOG = LogFactory.getLog(StatementBlock.class.getName());
	protected static IDSequence _seq = new IDSequence();
	private static IDSequence _seqSBID = new IDSequence();
	protected final long _ID;
	protected final String _name;
	
	protected DMLProgram _dmlProg;
	protected ArrayList<Statement> _statements;
	ArrayList<Hop> _hops = null;
	ArrayList<Lop> _lops = null;
	HashMap<String,ConstIdentifier> _constVarsIn;
	HashMap<String,ConstIdentifier> _constVarsOut;

	private boolean _recompileOnce = false;
	private ArrayList<String> _updateInPlaceVars = null;
	private boolean _requiresRecompile = false;
	private boolean _splitDag = false;
	private boolean _nondeterministic = false;
	private HashMap<Lop.Type, List<Lop.Type>> _checkpointPositions = null;

	protected double repetitions = 1;
	private double loopDepRatio = 0; //ratio of loop dependent HOP dags
	public final static double DEFAULT_LOOP_REPETITIONS = 10;

	public StatementBlock() {
		_ID = getNextSBID();
		_name = "SB"+_ID;
		_dmlProg = null;
		_statements = new ArrayList<>();
		_read = new VariableSet();
		_updated = new VariableSet();
		_gen = new VariableSet();
		_kill = new VariableSet();
		_warnSet = new VariableSet();
		_initialized = true;
		_constVarsIn = new HashMap<>();
		_constVarsOut = new HashMap<>();
		_updateInPlaceVars = new ArrayList<>();
	}
	
	public StatementBlock(StatementBlock sb) {
		this();
		setParseInfo(sb);
		_dmlProg = sb._dmlProg;
		_nondeterministic = sb.isNondeterministic();
	}

	public void setDMLProg(DMLProgram dmlProg){
		_dmlProg = dmlProg;
	}
	
	private static long getNextSBID() {
		return _seqSBID.getNextID();
	}

	public DMLProgram getDMLProg(){
		return _dmlProg;
	}
	
	public long getSBID() {
		return _ID;
	}
	
	public String getName() {
		return _name;
	}

	public void addStatement(Statement s) {
		_statements.add(s);
		if (_statements.size() == 1){
			_filename    = s.getFilename();
			_beginLine   = s.getBeginLine();
			_beginColumn = s.getBeginColumn();
		}
		_endLine   = s.getEndLine();
		_endColumn = s.getEndColumn();
	}

	public void addStatementBlock(StatementBlock s){
		for (int i = 0; i < s.getNumStatements(); i++)
			_statements.add(s.getStatement(i));
		_beginLine   = _statements.get(0).getBeginLine();
		_beginColumn = _statements.get(0).getBeginColumn();
		_endLine     = _statements.get(_statements.size() - 1).getEndLine();
		_endColumn   = _statements.get(_statements.size() - 1).getEndColumn();
	}

	public int getNumStatements(){
		return _statements.size();
	}

	public Statement getStatement(int i){
		return _statements.get(i);
	}

	public ArrayList<Statement> getStatements() {
		return _statements;
	}

	public void setStatements( ArrayList<Statement> s ) {
		_statements = s;
	}

	public ArrayList<Hop> getHops() {
		return _hops;
	}

	public ArrayList<Lop> getLops() {
		return _lops;
	}

	public void setHops(ArrayList<Hop> hops) {
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
	
	public void setSplitDag(boolean flag) {
		_splitDag = flag;
	}
	
	public boolean isSplitDag() {
		return _splitDag;
	}

	public double getLoopDepRatio() {
		return loopDepRatio;
	}

	// maintain the ration of loop-dependent HOP dags in this block
	public void setLoopDepRatio(double dep) {
		loopDepRatio = dep;
	}

	private static boolean isMergeablePrintStatement(Statement stmt) {
		return ( stmt instanceof PrintStatement &&
			(((PrintStatement)stmt).getType() == PRINTTYPE.STOP || ((PrintStatement)stmt).getType() == PRINTTYPE.ASSERT) );
	}

	public boolean isMergeableFunctionCallBlock(DMLProgram dmlProg) {
		// check whether targetIndex stmt block is for a mergable function call
		Statement stmt = this.getStatement(0);
		
		// Check whether targetIndex block is: control stmt block or stmt block for un-mergable function call
		if (   stmt instanceof WhileStatement || stmt instanceof IfStatement || stmt instanceof ForStatement
			|| stmt instanceof FunctionStatement || isMergeablePrintStatement(stmt) )
		{
			return false;
		}

		if (stmt instanceof AssignmentStatement || stmt instanceof MultiAssignmentStatement){
			Expression sourceExpr = null;
			if (stmt instanceof AssignmentStatement) {
				AssignmentStatement astmt = (AssignmentStatement)stmt;
				// for now, ensure that an assignment statement containing a read from csv ends up in own statement block
				if(astmt.getSource().toString().contains(DataExpression.FORMAT_TYPE + "=" + FileFormat.CSV.toString()) 
					&& astmt.getSource().toString().contains("read"))
					return false;
				if (astmt.controlStatement())
					return false;
				sourceExpr = astmt.getSource();
			}
			else
				sourceExpr = ((MultiAssignmentStatement)stmt).getSource();
			if ( (sourceExpr instanceof BuiltinFunctionExpression && ((BuiltinFunctionExpression)sourceExpr).multipleReturns())
				|| (sourceExpr instanceof ParameterizedBuiltinFunctionExpression && ((ParameterizedBuiltinFunctionExpression)sourceExpr).multipleReturns()))
				return false;

			// function calls (only mergable if inlined dml-bodied function)
			if (sourceExpr instanceof FunctionCallIdentifier) {
				FunctionCallIdentifier fcall = (FunctionCallIdentifier) sourceExpr;
				FunctionStatementBlock fblock = dmlProg.getFunctionStatementBlock(fcall.getNamespace(),
						fcall.getName());
				if (fblock == null) {
					//special-handling builtin functions that are not yet type-customized
					if( Builtins.contains(fcall.getName(), true, false) )
						return false;
					
					if (DMLProgram.DEFAULT_NAMESPACE.equals(fcall.getNamespace())) {
						throw new LanguageException(
								sourceExpr.printErrorLocation() + "Function " + fcall.getName() + "() is undefined.");
					} else {
						throw new LanguageException(sourceExpr.printErrorLocation() + "Function " + fcall.getName()
								+ "() is undefined in namespace '" + fcall.getNamespace() + "'.");
					}
				}
				if (!rIsInlineableFunction(fblock, dmlProg))
					return false;
			}
		}

		// regular statement block
		return true;
	}

	public boolean isRewritableFunctionCall(Statement stmt, DMLProgram dmlProg) {

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
				if (fblock == null) {
					if( Builtins.contains(fcall.getName(), true, false) 
						|| DMLProgram.isInternalNamespace(fcall.getNamespace()))
						return false;
					throw new LanguageException(sourceExpr.printErrorLocation() + "function " 
						+ fcall.getName() + " is undefined in namespace " + fcall.getNamespace());
				}

				//check for unsupported target indexed identifiers (for consistent error handling)
				if( stmt instanceof AssignmentStatement
					&& ((AssignmentStatement)stmt).getTarget() instanceof IndexedIdentifier ) {
					return false;
				}

				//check if function can be inlined
				if( rIsInlineableFunction(fblock, dmlProg) ) {
					return true;
				}
			}
		}

		// regular statement
		return false;
	}


	private boolean rIsInlineableFunction( FunctionStatementBlock fblock, DMLProgram prog )
	{
		boolean ret = true;
	
		//reject external functions and function bodies with multiple blocks
		if(    fblock.getStatements().isEmpty() //empty blocks
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
					if( as.getSource().toString().contains(DataExpression.FORMAT_TYPE + "=" + FileFormat.CSV.toString())
						&& as.getSource().toString().contains("read"))
						return false;
			
					if( !ret ) return false;
				}
				else if( s instanceof MultiAssignmentStatement ) {
					MultiAssignmentStatement mas = (MultiAssignmentStatement)s;
					if( mas.getSource() instanceof FunctionCallIdentifier ) {
						FunctionCallIdentifier fcall = (FunctionCallIdentifier) ((MultiAssignmentStatement)s).getSource();
						FunctionStatementBlock fblock2 = prog.getFunctionStatementBlock(fcall.getNamespace(), fcall.getName());
						ret &= rIsInlineableFunction(fblock2, prog);
						if( !ret ) return false;
					}
					else if( mas.getSource() instanceof BuiltinFunctionExpression
						&& ((BuiltinFunctionExpression)mas.getSource()).multipleReturns() ) {
						return false;
					}
				}
			}
		}
	
		return ret;
	}

	public static ArrayList<StatementBlock> mergeFunctionCalls(List<StatementBlock> body, DMLProgram dmlProg) 
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

		ArrayList<StatementBlock> result = new ArrayList<>();
		StatementBlock currentBlock = null;

		for (int i = 0; i < body.size(); i++) {
			StatementBlock current = body.get(i);
			if (current.isMergeableFunctionCallBlock(dmlProg)){
				if (currentBlock != null)
					currentBlock.addStatementBlock(current);
				else
					currentBlock = current;
			}
			else {
				if (currentBlock != null)
					result.add(currentBlock);
				result.add(current);
				currentBlock = null;
			}
		}

		if (currentBlock != null) {
			result.add(currentBlock);
		}

		return result;
	}

	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append("statements\n");
		for (Statement s : _statements){
			sb.append(s);
			sb.append("\n");
		}
		if (_liveOut != null) sb.append("liveout " + _liveOut.toString() + "\n");
		if (_liveIn!= null) sb.append("livein " + _liveIn.toString()+ "\n");
		if (_gen != null && !_gen.getVariables().isEmpty()) sb.append("gen " + _gen.toString()+ "\n");
		if (_kill != null && !_kill.getVariables().isEmpty()) sb.append("kill " + _kill.toString()+ "\n");
		if (_read != null && !_read.getVariables().isEmpty()) sb.append("read " + _read.toString()+ "\n");
		if (_updated != null && !_updated.getVariables().isEmpty()) sb.append("updated " + _updated.toString()+ "\n");
		return sb.toString();
	}
	
	public ArrayList<String> getInputstoSB() {
		ArrayList<String> inputs = _liveIn != null && _read != null ? new ArrayList<>() : null;
		if (_liveIn != null && _read != null) {
			for (String varName : _read.getVariables().keySet()) {
				if (_liveIn.containsVariable(varName))
					inputs.add(varName);
			}
		}
		return inputs;
	}

	public ArrayList<String> getOutputNamesofSB() {
		ArrayList<String> outputs = _liveOut != null 
			&& _updated != null ? new ArrayList<>() : null;
		if (_liveOut != null && _updated != null) {
			for (String varName : _updated.getVariables().keySet()) {
				if (_liveOut.containsVariable(varName))
					outputs.add(varName);
			}
		}
		return outputs;
	}
	
	public ArrayList<DataIdentifier> getOutputsofSB() {
		ArrayList<DataIdentifier> outputs = _liveOut != null 
			&& _updated != null ? new ArrayList<>() : null;
		if (_liveOut != null && _updated != null) {
			for (String varName : _updated.getVariables().keySet()) {
				if (_liveOut.containsVariable(varName))
					outputs.add(_liveOut.getVariable(varName));
			}
		}
		return outputs;
	}

	public static ArrayList<StatementBlock> mergeStatementBlocks(List<StatementBlock> sb){
		if (sb == null || sb.isEmpty())
			return new ArrayList<>();

		ArrayList<StatementBlock> result = new ArrayList<>();
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
	
	public static List<StatementBlock> rHoistFunctionCallsFromExpressions(StatementBlock current, DMLProgram prog) {
		if (current instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock)current;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			ArrayList<StatementBlock> tmp = new ArrayList<>();
			for (StatementBlock sb : fstmt.getBody())
				tmp.addAll(rHoistFunctionCallsFromExpressions(sb, prog));
			fstmt.setBody(tmp);
		}
		else if (current instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) current;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			//TODO handle predicates
			ArrayList<StatementBlock> tmp = new ArrayList<>();
			for (StatementBlock sb : wstmt.getBody())
				tmp.addAll(rHoistFunctionCallsFromExpressions(sb, prog));
			wstmt.setBody(tmp);
		}
		else if (current instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) current;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			//TODO handle predicates
			ArrayList<StatementBlock> tmp = new ArrayList<>();
			for (StatementBlock sb : istmt.getIfBody())
				tmp.addAll(rHoistFunctionCallsFromExpressions(sb, prog));
			istmt.setIfBody(tmp);
			if( istmt.getElseBody() != null && !istmt.getElseBody().isEmpty() ) {
				ArrayList<StatementBlock> tmp2 = new ArrayList<>();
				for (StatementBlock sb : istmt.getElseBody())
					tmp2.addAll(rHoistFunctionCallsFromExpressions(sb, prog));
				istmt.setElseBody(tmp2);
			}
		}
		else if (current instanceof ForStatementBlock) { //incl parfor
			ForStatementBlock fsb = (ForStatementBlock) current;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			//TODO handle predicates
			ArrayList<StatementBlock> tmp = new ArrayList<>();
			for (StatementBlock sb : fstmt.getBody())
				tmp.addAll(rHoistFunctionCallsFromExpressions(sb, prog));
			fstmt.setBody(tmp);
		}
		else { //generic (last-level)
			ArrayList<Statement> tmp = new ArrayList<>();
			for(Statement stmt : current.getStatements())
				tmp.addAll(rHoistFunctionCallsFromExpressions(stmt, prog));
			if( current.getStatements().size() != tmp.size() )
				return createStatementBlocks(current, tmp);
		}
		return Arrays.asList(current);
	}

	public static List<Statement> rHoistFunctionCallsFromExpressions(Statement stmt, DMLProgram prog) {
		ArrayList<Statement> tmp = new ArrayList<>();
		if( stmt instanceof AssignmentStatement ) {
			AssignmentStatement astmt = (AssignmentStatement)stmt;
			boolean ix = (astmt.getTargetList().get(0) instanceof IndexedIdentifier);
			rHoistFunctionCallsFromExpressions(astmt.getSource(), !ix, tmp, prog);
			if( ix && astmt.getSource() instanceof FunctionCallIdentifier ) {
				AssignmentStatement lstmt = (AssignmentStatement) tmp.get(tmp.size()-1);
				astmt.setSource(copy(lstmt.getTarget()));
			}
		}
		else if( stmt instanceof MultiAssignmentStatement ) {
			MultiAssignmentStatement mstmt = (MultiAssignmentStatement)stmt;
			rHoistFunctionCallsFromExpressions(mstmt.getSource(), true, tmp, prog);
		}
		else if( stmt instanceof PrintStatement ) {
			PrintStatement pstmt = (PrintStatement)stmt;
			for(int i=0; i<pstmt.expressions.size(); i++) {
				Expression lexpr = pstmt.getExpressions().get(i);
				rHoistFunctionCallsFromExpressions(lexpr, false, tmp, prog);
				if( lexpr instanceof FunctionCallIdentifier ) {
					AssignmentStatement lstmt = (AssignmentStatement) tmp.get(tmp.size()-1);
					pstmt.getExpressions().set(i, copy(lstmt.getTarget()));
				}
			}
		}
		
		//most statements will be returned unchanged, while expressions with
		//function calls are split into potentially many statements
		List<Statement> ret = tmp.isEmpty() ? Arrays.asList(stmt) : tmp;
		if( !tmp.isEmpty() ) {
			for( Statement ltmp : tmp )
				ltmp.setParseInfo(stmt);
			tmp.add(stmt);
		}
		return ret;
	}
	
	public static Expression rHoistFunctionCallsFromExpressions(Expression expr, boolean root, ArrayList<Statement> tmp, DMLProgram prog) {
		if( expr == null || expr instanceof ConstIdentifier )
			return expr; //do nothing
		if( expr instanceof BinaryExpression ) {
			BinaryExpression lexpr = (BinaryExpression) expr;
			lexpr.setLeft(rHoistFunctionCallsFromExpressions(lexpr.getLeft(), false, tmp, prog));
			lexpr.setRight(rHoistFunctionCallsFromExpressions(lexpr.getRight(), false, tmp, prog));
		}
		else if( expr instanceof RelationalExpression ) {
			RelationalExpression lexpr = (RelationalExpression) expr;
			lexpr.setLeft(rHoistFunctionCallsFromExpressions(lexpr.getLeft(), false, tmp, prog));
			lexpr.setRight(rHoistFunctionCallsFromExpressions(lexpr.getRight(), false, tmp, prog));
		}
		else if( expr instanceof BooleanExpression ) {
			BooleanExpression lexpr = (BooleanExpression) expr;
			lexpr.setLeft(rHoistFunctionCallsFromExpressions(lexpr.getLeft(), false, tmp, prog));
			lexpr.setRight(rHoistFunctionCallsFromExpressions(lexpr.getRight(), false, tmp, prog));
		}
		else if( expr instanceof BuiltinFunctionExpression ) {
			BuiltinFunctionExpression lexpr = (BuiltinFunctionExpression) expr;
			Expression[] clexpr = lexpr.getAllExpr();
			for( int i=0; i<clexpr.length; i++ )
				clexpr[i] = rHoistFunctionCallsFromExpressions(clexpr[i], false, tmp, prog);
			if( !root && lexpr.getOpCode()==Builtins.TIME ) { //core time hoisting
				String varname = StatementBlockRewriteRule.createCutVarName(true);
				DataIdentifier di = new DataIdentifier(varname);
				di.setDataType(lexpr.getDataType());
				di.setValueType(lexpr.getValueType());
				tmp.add(new AssignmentStatement(di, lexpr, di));
			}
		}
		else if( expr instanceof ParameterizedBuiltinFunctionExpression ) {
			ParameterizedBuiltinFunctionExpression lexpr = (ParameterizedBuiltinFunctionExpression) expr;
			HashMap<String, Expression> clexpr = lexpr.getVarParams();
			for( String key : clexpr.keySet() )
				clexpr.put(key, rHoistFunctionCallsFromExpressions(clexpr.get(key), false, tmp, prog));
		}
		else if( expr instanceof DataExpression ) {
			DataExpression lexpr = (DataExpression) expr;
			HashMap<String, Expression> clexpr = lexpr.getVarParams();
			for( String key : clexpr.keySet() )
				clexpr.put(key, rHoistFunctionCallsFromExpressions(clexpr.get(key), false, tmp, prog));
		}
		else if( expr instanceof FunctionCallIdentifier ) {
			FunctionCallIdentifier fexpr = (FunctionCallIdentifier) expr;
			for( ParameterExpression pexpr : fexpr.getParamExprs() )
				pexpr.setExpr(rHoistFunctionCallsFromExpressions(pexpr.getExpr(), false, tmp, prog));
			if( !root ) { //core fcall hoisting
				String varname = StatementBlockRewriteRule.createCutVarName(true);
				DataIdentifier di = new DataIdentifier(varname);
				di.setDataType(fexpr.getDataType());
				di.setValueType(fexpr.getValueType());
				tmp.add(new AssignmentStatement(di, fexpr, di));
				//add hoisted dml-bodied builtin function to program (if not already loaded)
				FunctionDictionary<FunctionStatementBlock> fdict = prog.getBuiltinFunctionDictionary();
				if( Builtins.contains(fexpr.getName(), true, false) && (fdict == null ||
					(!fdict.containsFunction(Builtins.getInternalFName(fexpr.getName(), DataType.SCALAR))
					&& !fdict.containsFunction(Builtins.getInternalFName(fexpr.getName(), DataType.MATRIX)))) )
				{
					fdict = prog.createNamespace(DMLProgram.BUILTIN_NAMESPACE);
					Map<String,FunctionStatementBlock> fsbs = DmlSyntacticValidator
						.loadAndParseBuiltinFunction(fexpr.getName(), DMLProgram.BUILTIN_NAMESPACE, false);
					for( Entry<String,FunctionStatementBlock> fsb : fsbs.entrySet() ) {
						if( !fdict.containsFunction(fsb.getKey()) )
							fdict.addFunction(fsb.getKey(), fsb.getValue());
						fsb.getValue().setDMLProg(prog);
					}
				}
				return di;
			}
		}
		//note: all remaining expressions data identifiers remain unchanged
		return expr;
	}
	
	private static DataIdentifier copy(DataIdentifier di) {
		return new DataIdentifier(di);
	}
	
	private static List<StatementBlock> createStatementBlocks(StatementBlock sb, List<Statement> stmts) {
		List<StatementBlock> ret = new ArrayList<>();
		StatementBlock current = new StatementBlock(sb);
		for(Statement stmt : stmts) {
			//cut the statement block before and after the current function
			//(cut before is precondition for subsequent merge steps which 
			//assume function statements as the first statement in the block)
			boolean cut = stmt instanceof AssignmentStatement
				&& ((AssignmentStatement)stmt).getSource() instanceof FunctionCallIdentifier;
			if( cut && current.getNumStatements() > 0 ) { //before
				ret.add(current);
				current = new StatementBlock(sb);
			}
			current.addStatement(stmt);
			if( cut ) { //after
				ret.add(current);
				current = new StatementBlock(sb);
			}
		}
		if( current.getNumStatements() > 0 )
			ret.add(current);
		return ret;
	}
	
	public ArrayList<Statement> rewriteFunctionCallStatements (DMLProgram dmlProg, ArrayList<Statement> statements) {

		ArrayList<Statement> newStatements = new ArrayList<>();
		for (Statement current : statements) {
			if( !isRewritableFunctionCall(current, dmlProg) ) {
				newStatements.add(current);
				continue;
			}

			Expression sourceExpr = (current instanceof AssignmentStatement) ?
				((AssignmentStatement)current).getSource() :
				((MultiAssignmentStatement)current).getSource();
			FunctionCallIdentifier fcall = (FunctionCallIdentifier) sourceExpr;
			FunctionStatementBlock fblock = dmlProg.getFunctionStatementBlock(fcall.getNamespace(), fcall.getName());
			if( fblock == null )
				fcall.raiseValidateError("function " + fcall.getName() + " is undefined in namespace " + fcall.getNamespace(), false);
			FunctionStatement fstmt = (FunctionStatement)fblock.getStatement(0);

			// recursive inlining (no memo required because update-inplace of function statement blocks, so no redundant inlining)
			if( rIsInlineableFunction(fblock, dmlProg) ){
				fstmt.getBody().get(0).setStatements(
					rewriteFunctionCallStatements(dmlProg, fstmt.getBody().get(0).getStatements()));
			}

			//MB: we cannot use the hash since multiple interleaved inlined functions should be independent.
			String prefix = _seq.getNextID() + "_";

			if (fstmt.getBody().size() > 1){
				sourceExpr.raiseValidateError("rewritable function can only have 1 statement block", false);
			}
			StatementBlock sblock = fstmt.getBody().get(0);

			if( fcall.getParamExprs().size() != fstmt.getInputParams().size() ) {
				sourceExpr.raiseValidateError("Wrong number of function input arguments: "+
					fcall.getParamExprs().size() + " found, but " + fstmt.getInputParams().size()+" expected.");
			}

			for (int i =0; i < fcall.getParamExprs().size(); i++) {
				ParameterExpression inputArg = fcall.getParamExprs().get(i);
				DataIdentifier currFormalParam = (inputArg.getName()==null) ?
					fstmt.getInputParams().get(i) : fstmt.getInputParam(inputArg.getName());
				if( currFormalParam == null )
					throw new LanguageException("Non-existing named function argument '"
						+ inputArg.getName()+"' in call to "+fcall.getName()+".");
				
				// create new assignment statement
				String newFormalParameterName = prefix + currFormalParam.getName();
				DataIdentifier newTarget = new DataIdentifier(currFormalParam);
				newTarget.setName(newFormalParameterName);

				Expression currCallParam = inputArg.getExpr();

				//auto casting of inputs on inlining (if required)
				ValueType targetVT = newTarget.getValueType();
				if (newTarget.getDataType() == DataType.SCALAR && currCallParam.getOutput() != null
						&& targetVT != currCallParam.getOutput().getValueType() && targetVT != ValueType.STRING) {
					currCallParam = new BuiltinFunctionExpression(
						BuiltinFunctionExpression.getValueTypeCastOperator(targetVT),
						new Expression[] { currCallParam }, newTarget);
				}

				// create the assignment statement to bind the call parameter to formal parameter
				newStatements.add(new AssignmentStatement(newTarget, currCallParam, newTarget));
			}

			for (Statement stmt : sblock._statements){
				// rewrite the statement to use the "rewritten" name
				Statement rewrittenStmt = stmt.rewriteStatement(prefix);
				newStatements.add(rewrittenStmt);
			}

			if (current instanceof AssignmentStatement) {
				if (fstmt.getOutputParams().size() == 0) {
					AssignmentStatement as = (AssignmentStatement) current;
					if ((as.getTargetList().size() == 1) && (as.getTargetList().get(0) != null)) {
						raiseValidateError("Function '" + fcall.getName()
							+ "' does not return a value but is assigned to " + as.getTargetList().get(0), true);
					}
				}
			}
			else if (current instanceof MultiAssignmentStatement) {
				if (fstmt.getOutputParams().size() == 0) {
					MultiAssignmentStatement mas = (MultiAssignmentStatement) current;
					raiseValidateError("Function '" + fcall.getName()
						+ "' does not return a value but is assigned to " + mas.getTargetList(), true);
				}
			}
			
			// handle returns by appending name mappings, but with special handling of 
			// statements that contain function calls or multi-return builtin expressions (but disabled)
			appendOutputAssignments(current, prefix, fstmt, newStatements);
		}
		return newStatements;
	}
	
	@SuppressWarnings("unused")
	private static boolean isOutputBindingViaFunctionCall(Statement last, String prefix, FunctionStatement fstmt) {
		if( last instanceof AssignmentStatement ) {
			AssignmentStatement as = (AssignmentStatement) last;
			String newName = prefix + fstmt.getOutputParams().get(0).getName();
			return as.getSource() instanceof FunctionCallIdentifier
				&& as.getTarget().getName().equals(newName);
		}
		else if( last instanceof MultiAssignmentStatement ) {
			MultiAssignmentStatement mas = (MultiAssignmentStatement) last;
			List<DataIdentifier> tlist1 = mas.getTargetList();
			boolean ret = mas.getSource() instanceof FunctionCallIdentifier
				|| (mas.getSource() instanceof BuiltinFunctionExpression 
					&& ((BuiltinFunctionExpression)mas.getSource()).multipleReturns());
			for( DataIdentifier di : fstmt.getOutputParams() )
				ret &= tlist1.stream().anyMatch(d -> d.getName().equals(prefix+di.getName()));
			return ret;
		}
		return false; //default
	}
	
	@SuppressWarnings("unused")
	private static MultiAssignmentStatement createNewPartialMultiAssignment(Statement last, Statement current, String prefix, FunctionStatement fstmt) {
		MultiAssignmentStatement mas = (MultiAssignmentStatement) last;
		AssignmentStatement as = (AssignmentStatement) current;
		ArrayList<DataIdentifier> tlist = new ArrayList<>();
		String tmpStr = prefix+fstmt.getOutputParams().get(0).getName();
		for( DataIdentifier di : mas.getTargetList() )
			tlist.add( di.getName().equals(tmpStr) ? as.getTarget() : di );
		return new MultiAssignmentStatement(tlist, mas.getSource());
	}
	
	private static void appendOutputAssignments(Statement current, String prefix, FunctionStatement fstmt, List<Statement> newStatements) {
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
					fstmt.raiseValidateError("Assignment statement cannot return multiple values", false);
				}
				AssignmentStatement as = (AssignmentStatement) current;
				DataIdentifier targ = as.getTarget();
				if (targ == null) {
					Expression exp = as.getSource();
					FunctionCallIdentifier fci = (FunctionCallIdentifier) exp;
					String functionName = fci.getName();
					fstmt.raiseValidateError(functionName + " requires LHS value", false);
				} else {
					newTarget = new DataIdentifier(((AssignmentStatement)current).getTarget());
				}
			}
			else{
				newTarget = new DataIdentifier(((MultiAssignmentStatement)current).getTargetList().get(i));
			}

			//auto casting of inputs on inlining (always, redundant cast removed during Hop Rewrites)
			ValueType sourceVT = newSource.getValueType();
			if (newSource.getDataType() == DataType.SCALAR && sourceVT != ValueType.STRING) {
				newSource = new BuiltinFunctionExpression(
					BuiltinFunctionExpression.getValueTypeCastOperator(sourceVT),
					new Expression[] { newSource }, newTarget);
			}

			// create the assignment statement to bind the call parameter to formal parameter
			newStatements.add(new AssignmentStatement(newTarget, newSource, newTarget));
		}
	}

	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String, ConstIdentifier> constVars, boolean conditional)
	{
		_constVarsIn.putAll(constVars);
		_statements = rewriteFunctionCallStatements(dmlProg, _statements);
		_dmlProg = dmlProg;
		
		HashMap<String, ConstIdentifier> currConstVars = new HashMap<>(constVars);
		for (Statement current : _statements) {
			if (current instanceof OutputStatement) {
				OutputStatement os = (OutputStatement)current;
				// validate variable being written by output statement exists
				DataIdentifier target = os.getIdentifier();
				if (ids.getVariable(target.getName()) == null) {
					//undefined variables are always treated unconditionally as error in order to prevent common script-level bugs
					raiseValidateError("Undefined Variable (" + target.getName() + ") used in statement", false, LanguageErrorCodes.INVALID_PARAMETERS);
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
				validateAssignmentStatement(current, dmlProg, ids, currConstVars, conditional);
			}
			else if (current instanceof MultiAssignmentStatement){
				validateMultiAssignmentStatement(current, dmlProg, ids, currConstVars, conditional);
			}
			else if(current instanceof ForStatement || current instanceof IfStatement || current instanceof WhileStatement ){
				raiseValidateError("control statement (WhileStatement, IfStatement, ForStatement) should not be in generic statement block. Likely a parsing error", conditional);
			}
			else if (current instanceof PrintStatement) {
				PrintStatement pstmt = (PrintStatement) current;
				List<Expression> expressions = pstmt.getExpressions();
				for (Expression expression : expressions) {
					expression.validateExpression(ids.getVariables(), currConstVars, conditional);
					DataType outputDatatype = expression.getOutput().getDataType();
					switch (outputDatatype) {
						case SCALAR:
						case MATRIX:
						case FRAME:
						case LIST:
							break;
						case TENSOR:
							pstmt.raiseValidateError("Print statements can only print scalars. To print a " + outputDatatype + ", please wrap it in a toString() function.", conditional);
						default:
							pstmt.raiseValidateError("Print statements can only print scalars. Input datatype was: " + outputDatatype, conditional);
					}
				}
			}
			// no work to perform for PathStatement or ImportStatement
			else if (current instanceof PathStatement){}
			else if (current instanceof ImportStatement){}
			else {
				raiseValidateError("cannot process statement of type " + current.getClass().getSimpleName(), conditional);
			}
		}
		_constVarsOut.putAll(currConstVars);
		return ids;
	}
	
	private void validateAssignmentStatement(Statement current, DMLProgram dmlProg, 
		VariableSet ids, HashMap<String, ConstIdentifier> currConstVars, boolean conditional) 
	{
		AssignmentStatement as = (AssignmentStatement)current;
		DataIdentifier target = as.getTarget();
		Expression source = as.getSource();
		
		// check if target is builtin constant
		if (target != null && BuiltinConstant.contains(target.getName())) {
			target.raiseValidateError(String.format(
				"Cannot assign a value to the builtin constant %s.", target.getName()), false);
		}
		
		if (source instanceof FunctionCallIdentifier) {
			((FunctionCallIdentifier) source).validateExpression(
				dmlProg, ids.getVariables(),currConstVars, conditional);
		}
		else { //all builtin functions and expressions
			if( target == null  )
				raiseValidateError("Missing variable assignment.", false);
			
			if( MLContextProxy.isActive() )
				MLContextProxy.setAppropriateVarsForRead(source, target._name);
			
			source.validateExpression(ids.getVariables(), currConstVars, conditional);
		}
		
		if (source instanceof DataExpression && ((DataExpression)source).getOpCode() == Expression.DataOp.READ)
			setStatementFormatType(as, conditional);
		
		// Handle const vars: (a) basic constant propagation, and (b) transitive constant propagation over assignments
		if (target != null) {
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
		}
		
		if (source instanceof BuiltinFunctionExpression){
			BuiltinFunctionExpression bife = (BuiltinFunctionExpression)source;
			if (   bife.getOpCode() == Builtins.NROW
				|| bife.getOpCode() == Builtins.NCOL )
			{
				DataIdentifier id = (DataIdentifier)bife.getFirstExpr();
				DataIdentifier currVal = ids.getVariable(id.getName());
				if (currVal == null){
					//undefined variables are always treated unconditionally as error in order to prevent common script-level bugs
					bife.raiseValidateError("Undefined Variable (" + id.getName() + ") used in statement", false, LanguageErrorCodes.INVALID_PARAMETERS);
				}
				IntIdentifier intid = null;
				if (bife.getOpCode() == Builtins.NROW) {
					intid = new IntIdentifier((currVal instanceof IndexedIdentifier)
							? ((IndexedIdentifier) currVal).getOrigDim1() : currVal.getDim1(), bife);
				} else {
					intid = new IntIdentifier((currVal instanceof IndexedIdentifier)
							? ((IndexedIdentifier) currVal).getOrigDim2() : currVal.getDim2(), bife);
				}
				
				// handle case when nrow / ncol called on variable with size unknown (dims == -1)
				//	--> const prop NOT possible
				if (intid.getValue() != -1)
					currConstVars.put(target.getName(), intid);
			}
		}
		if (target == null) {
			// function has no return value
		}
		// CASE: target NOT indexed identifier
		else if (!(target instanceof IndexedIdentifier)){
			if( as.isAccumulator() && ids.containsVariable(target.getName()) )
				target.setProperties(ids.getVariable(target.getName()));
			else
				target.setProperties(source.getOutput());
			if (source.getOutput() instanceof IndexedIdentifier)
				target.setDimensions(source.getOutput().getDim1(), source.getOutput().getDim2());
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
			
			// validate that size of LHS index ranges is being assigned:
			//	(a) a matrix value of same size as LHS
			//	(b) singleton value (semantics: initialize entire submatrix with this value)
			IndexPair targetSize = ((IndexedIdentifier)target).calculateIndexedDimensions(ids.getVariables(), currConstVars, conditional);
			
			if( target.getDataType().isMatrixOrFrame() ) {
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
			}
			((IndexedIdentifier)target).setDimensions(targetSize._row, targetSize._col);
		}
		
		if (target != null)
			ids.addVariable(target.getName(), target);
	}
	
	private void validateMultiAssignmentStatement(Statement current, DMLProgram dmlProg, 
		VariableSet ids, HashMap<String, ConstIdentifier> currConstVars, boolean conditional) 
	{
		MultiAssignmentStatement mas = (MultiAssignmentStatement) current;
		ArrayList<DataIdentifier> targetList = mas.getTargetList();
		Expression source = mas.getSource();

		// check if target list contains builtin constant
		targetList.forEach(target -> {
			if (target != null && BuiltinConstant.contains(target.getName()))
				target.raiseValidateError(String.format(
					"Cannot assign a value to the builtin constant %s.", target.getName()), false);
		});
		
		//MultiAssignmentStatments currently supports only External,
		//User-defined, and Multi-return Builtin function expressions
		if (!(source instanceof DataIdentifier)
				|| (source instanceof DataIdentifier && !((DataIdentifier)source).multipleReturns()) ) {
			source.raiseValidateError("can only use user-defined functions with multi-assignment statement", conditional);
		}
		if ( source instanceof FunctionCallIdentifier) {
			FunctionCallIdentifier fci = (FunctionCallIdentifier)source;
			fci.validateExpression(dmlProg, ids.getVariables(), currConstVars, conditional);
		}
		else if ( (source instanceof BuiltinFunctionExpression || source instanceof ParameterizedBuiltinFunctionExpression)
				&& ((DataIdentifier)source).multipleReturns()) {
			source.validateExpression(mas, ids.getVariables(), currConstVars, conditional);
		}
		else
			throw new LanguageException("Unexpected error.");
		
		if ( source instanceof FunctionCallIdentifier ) {
			for (int j =0; j< targetList.size(); j++) {
				DataIdentifier target = targetList.get(j);
				// set target properties (based on type info in function call statement return params)
				FunctionCallIdentifier fci = (FunctionCallIdentifier)source;
				FunctionStatementBlock fblock = _dmlProg.getFunctionStatementBlock(fci.getNamespace(), fci.getName());
				if (fblock == null){
					fci.raiseValidateError(" function " + fci.getName() 
						+ " is undefined in namespace " + fci.getNamespace(), conditional);
					return;
				}
				FunctionStatement fstmt = (FunctionStatement)fblock.getStatement(0);
				if (!(target instanceof IndexedIdentifier)){
					target.setProperties(fstmt.getOutputParams().get(j));
				}
				else{
					DataIdentifier targetAsSeen = ids.getVariable(target.getName());
					if (targetAsSeen == null){
						raiseValidateError(target.printErrorLocation() + "cannot assign value to indexed identifier " 
							+ target.toString() + " without first initializing " + target.getName(), conditional);
					}
					target.setProperties(targetAsSeen);
				}
				ids.addVariable(target.getName(), target);
			}
		}
		else if ( source instanceof BuiltinFunctionExpression || source instanceof ParameterizedBuiltinFunctionExpression ) {
			Identifier[] outputs = source.getOutputs();
			for (int j=0; j < targetList.size(); j++) {
				ids.addVariable(targetList.get(j).getName(), (DataIdentifier)outputs[j]);
			}
		}
		
		// remove updated constant vars (for correctness)
		for(DataIdentifier target : targetList)
			currConstVars.remove(target.getName());
	}
	
	public void setStatementFormatType(OutputStatement s, boolean conditionalValidate)
	{
		//case of specified format parameter
		if (s.getExprParam(DataExpression.FORMAT_TYPE)!= null )
		{
	 		Expression formatTypeExpr = s.getExprParam(DataExpression.FORMAT_TYPE);
			if( formatTypeExpr instanceof StringIdentifier ) {
		 		String ft = formatTypeExpr.toString();
				try {
					s.getIdentifier().setFileFormat(FileFormat.safeValueOf(ft));
				}
				catch(Exception ex) {
					raiseValidateError("IO statement parameter " + DataExpression.FORMAT_TYPE
						+ " can only be a string with one of following values: binary, text, mm, csv, libsvm, jsonl;"
						+ " invalid format: '"+ft+"'.", false, LanguageErrorCodes.INVALID_PARAMETERS);
				}
			}
			else {
				s.getIdentifier().setFileFormat(FileFormat.UNKNOWN);
			}
		}
		//case of unspecified format parameter, use default
		else {
			s.addExprParam(DataExpression.FORMAT_TYPE, new StringIdentifier(FileFormat.TEXT.toString(), s), true);
			s.getIdentifier().setFileFormat(FileFormat.TEXT);
		}
	}

	public void setStatementFormatType(AssignmentStatement s, boolean conditionalValidate)
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
			try {
				s.getTarget().setFileFormat(FileFormat.safeValueOf(ft));
			}
			catch(Exception ex) {
				raiseValidateError("IO statement parameter " + DataExpression.FORMAT_TYPE
					+ " can only be a string with one of following values: binary, text, mm, csv, libsvm", conditionalValidate, LanguageErrorCodes.INVALID_PARAMETERS);
			}
		} else {
			dataExpr.addVarParam(DataExpression.FORMAT_TYPE,
				new StringIdentifier(FileFormat.TEXT.toString(), dataExpr));
			s.getTarget().setFileFormat(FileFormat.TEXT);
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
	@Override
	public VariableSet initializeforwardLV(VariableSet activeIn) {

		for (Statement s : _statements){
			s.initializeforwardLV(activeIn);
			VariableSet read = s.variablesRead();
			VariableSet updated = s.variablesUpdated();

			if (s instanceof WhileStatement || s instanceof IfStatement || s instanceof ForStatement){
				raiseValidateError("control statement (while / for / if) cannot be in generic statement block", false);
			}

			if (read != null){
				// for each variable read in this statement but not updated in
				// 		any prior statement, add to sb._gen

				for (String var : read.getVariableNames()) {
					if (!_updated.containsVariable(var)) {
						_gen.addVariable(var, read.getVariable(var));
					}
				}
			}

			_read.addVariables(read);
			_updated.addVariables(updated);

			if (updated != null) {
				// for each updated variable that is not read
				for (String var : updated.getVariableNames())
				{
					//NOTE MB: always add updated vars to kill (in order to prevent side effects
					//of implicitly updated statistics over common data identifiers, propagated from
					//downstream operators to its inputs due to 'livein = gen \cup (liveout-kill))'.
					_kill.addVariable(var, _updated.getVariable(var));

					//if (!_read.containsVariable(var)) {
					//	_kill.addVariable(var, _updated.getVariable(var));
					//}
				}
			}
		}
		_liveOut = new VariableSet();
		_liveOut.addVariables(activeIn);
		_liveOut.addVariables(_updated);
		return _liveOut;
	}

	@Override
	public VariableSet initializebackwardLV(VariableSet loPassed) {
		int numStatements = _statements.size();
		VariableSet lo = new VariableSet(loPassed);
		for (int i = numStatements-1; i>=0; i--)
			lo = _statements.get(i).initializebackwardLV(lo);
		return new VariableSet(lo);
	}

	public HashMap<String, ConstIdentifier> getConstIn(){
		return _constVarsIn;
	}

	public HashMap<String, ConstIdentifier> getConstOut(){
		return _constVarsOut;
	}

	@Override
	public VariableSet analyze(VariableSet loPassed) {
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

	public boolean hasHops(){
		return getHops() != null && !getHops().isEmpty();
	}

	///////////////////////////////////////////////////////////////
	// validate error handling (consistent for all expressions)

	public void raiseValidateError( String msg, boolean conditional ) {
		raiseValidateError(msg, conditional, null);
	}

	public void raiseValidateError( String msg, boolean conditional, String errorCode )
	{
		if( conditional )  //warning if conditional
		{
			String fullMsg = this.printWarningLocation() + msg;

			LOG.warn( fullMsg );
		}
		else  //error and exception if unconditional
		{
			String fullMsg = this.printErrorLocation() + msg;
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
	private String _text;

	@Override
	public void setFilename (String fname)  { _filename = fname; }
	@Override
	public void setBeginLine(int passed)    { _beginLine = passed; }
	@Override
	public void setBeginColumn(int passed)  { _beginColumn = passed; }
	@Override
	public void setEndLine(int passed)      { _endLine = passed; }
	@Override
	public void setEndColumn(int passed)    { _endColumn = passed; }
	@Override
	public void setText(String text)        { _text = text; }

	/**
	 * Set parse information.
	 *
	 * @param parseInfo
	 *            parse information, such as beginning line position, beginning
	 *            column position, ending line position, ending column position,
	 *            text, and filename
	 *            the DML filename (if it exists)
	 */
	public void setParseInfo(ParseInfo parseInfo) {
		_beginLine = parseInfo.getBeginLine();
		_beginColumn = parseInfo.getBeginColumn();
		_endLine = parseInfo.getEndLine();
		_endColumn = parseInfo.getEndColumn();
		_text = parseInfo.getText();
		_filename = parseInfo.getFilename();
	}

	@Override
	public String getFilename() { return _filename; }
	@Override
	public int getBeginLine() { return _beginLine; }
	@Override
	public int getBeginColumn() { return _beginColumn; }
	@Override
	public int getEndLine() { return _endLine; }
	@Override
	public int getEndColumn() { return _endColumn; }
	@Override
	public String getText() { return _text; }

	
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
	// materialized hops recompilation / updateinplace flags
	////

	public boolean updateRecompilationFlag() {
		return (_requiresRecompile =
			ConfigurationManager.isDynamicRecompilation()
			&& Recompiler.requiresRecompilation(getHops()));
	}

	public boolean requiresRecompilation() {
		return _requiresRecompile;
	}

	public ArrayList<String> getUpdateInPlaceVars() {
		return _updateInPlaceVars;
	}

	public void setUpdateInPlaceVars( ArrayList<String> vars ) {
		_updateInPlaceVars = vars;
	}
	
	public void setNondeterministic(boolean flag) {
		_nondeterministic = flag;
	}
	
	public boolean isNondeterministic() {
		return _nondeterministic;
	}
	
	public void setRecompileOnce( boolean flag ) {
		_recompileOnce = flag;
	}
	
	public boolean isRecompileOnce() {
		return _recompileOnce;
	}

	public void setCheckpointPosition(Lop input, List<Lop> outputs) {
		// FIXME: Type is not the best key as many Lops may have the same types
		Lop.Type inputT = input.getType();
		List<Lop.Type> outputsT = outputs.stream().map(Lop::getType).collect(Collectors.toList());

		if (_checkpointPositions == null)
			_checkpointPositions = new HashMap<>();
		if (!_checkpointPositions.containsKey(inputT)) {
			_checkpointPositions.put(inputT, outputsT);
		}
	}

	public HashMap<Lop.Type, List<Lop.Type>> getCheckpointPositions() {
		return _checkpointPositions;
	}

	/**
	 * Deep copy function for StatementBlock
	 * @param original Original StatementBlock to copy
	 * @return Deep copied StatementBlock
	 * // Todo Exclude Hop
	 */
	public StatementBlock deepCopy() {
		StatementBlock copy;
		if (this instanceof FunctionStatementBlock) {
			copy = new FunctionStatementBlock();
		} else if (this instanceof IfStatementBlock) {
			copy = new IfStatementBlock();
		} else if (this instanceof ForStatementBlock){
			copy = new ForStatementBlock();
		} else if (this instanceof WhileStatementBlock){
			copy = new WhileStatementBlock();
		} else {
			copy = new StatementBlock();
		}

		// Copy basic metadata
		copy.setFilename(this.getFilename());
		copy.setBeginLine(this.getBeginLine());
		copy.setBeginColumn(this.getBeginColumn());
		copy.setEndLine(this.getEndLine());
		copy.setEndColumn(this.getEndColumn());
		copy.setText(this.getText());

		// Copy DML program reference
		copy.setDMLProg(this.getDMLProg());

		// Copy LiveVariableAnalysis information
		if (this.liveIn() != null)
			copy.setLiveIn(this.liveIn());
		if (this.liveOut() != null)
			copy.setLiveOut(this.liveOut());
		if (this._gen != null)
			copy._gen.addVariables(this._gen);
		if (this._kill != null)
			copy._kill.addVariables(this._kill);
		if (this._read != null)
			copy._read.addVariables(this._read);
		if (this._updated != null)
			copy._updated.addVariables(this._updated);
		if (this._warnSet != null)
			copy._warnSet.addVariables(this._warnSet);

		// Copy constant variables
		copy._constVarsIn.putAll(this._constVarsIn);
		copy._constVarsOut.putAll(this._constVarsOut);

		// Copy DAG split flag
		copy.setSplitDag(false);
		// Deep copy statements
		if (this._statements != null && !this._statements.isEmpty()) {
			for (Statement stmt : this._statements) {
				Statement copyStmt = null;

				if (stmt instanceof AssignmentStatement) {
					AssignmentStatement as = (AssignmentStatement)stmt;
					AssignmentStatement newAs = new AssignmentStatement(new DataIdentifier(as.getTarget()), as.getSource());
					newAs.setParseInfo(as);
					newAs.setAccumulator(as.isAccumulator());
					copyStmt = newAs;
				} 
				else if (stmt instanceof MultiAssignmentStatement) {
					MultiAssignmentStatement mas = (MultiAssignmentStatement)stmt;
					MultiAssignmentStatement newMas = new MultiAssignmentStatement(mas.getTargetList(), mas.getSource());
					newMas.setParseInfo(mas);
					copyStmt = newMas;
				} 
				else if (stmt instanceof IfStatement) {
					IfStatement is = (IfStatement)stmt;
					IfStatement newIs = new IfStatement();
					newIs.setParseInfo(is);
					newIs.setConditionalPredicate(is.getConditionalPredicate());
					newIs.setIfBody(copyStatementBlocks(is.getIfBody()));
					newIs.setElseBody(copyStatementBlocks(is.getElseBody()));
					copyStmt = newIs;
				} 
				else if (stmt instanceof FunctionStatement) {
					FunctionStatement fs = (FunctionStatement)stmt;
					FunctionStatement newFs = new FunctionStatement();
					newFs.setParseInfo(fs);
					newFs.setName(fs.getName());
					newFs.setInputParams(fs.getInputParams());
					newFs.setInputDefaults(fs.getInputDefaults());
					newFs.setOutputParams(fs.getOutputParams());
					newFs.setBody(copyStatementBlocks(fs.getBody()));
					copyStmt = newFs;
				} 
				else if (stmt instanceof ForStatement) {
					ForStatement fs = (ForStatement)stmt;
					ForStatement newFs = new ForStatement();
					newFs.setParseInfo(fs);
					newFs.setPredicate(fs.getIterablePredicate());
					newFs.setBody(copyStatementBlocks(fs.getBody()));
					copyStmt = newFs;
				} 
				else if (stmt instanceof WhileStatement) {
					WhileStatement ws = (WhileStatement)stmt;
					WhileStatement newWs = new WhileStatement();
					newWs.setParseInfo(ws);
					newWs.setPredicate(ws.getConditionalPredicate());
					newWs.setBody(copyStatementBlocks(ws.getBody()));
					copyStmt = newWs;
				} 
				else if (stmt instanceof PrintStatement) {
					PrintStatement ps = (PrintStatement)stmt;
					PrintStatement newPs = new PrintStatement(ps.getType(), ps.getExpressions());
					newPs.setParseInfo(ps);
					copyStmt = newPs;
				}
				else if (stmt instanceof OutputStatement) {
					OutputStatement os = (OutputStatement)stmt;
					OutputStatement newOs = new OutputStatement(os.getIdentifier(), Expression.DataOp.WRITE, os);
					newOs.setExprParams(os.getSource());
					copyStmt = newOs;
				}
				else {
					copyStmt = stmt;
					copyStmt.setParseInfo(stmt);
				}

				// Add copied statement to new StatementBlock
				if (copyStmt != null) {
					copy.addStatement(copyStmt);
				}
			}
		}

		// Initialize _hops and _lops to null
		copy._hops = null;
		copy._lops = null;

		return copy;
	}

	/**
	 * Method to deep copy StatementBlock list
	 * @param body StatementBlock list to copy
	 * @return Deep copied StatementBlock list
	 */
	private ArrayList<StatementBlock> copyStatementBlocks(ArrayList<StatementBlock> body) {
		ArrayList<StatementBlock> newBody = new ArrayList<>();
		for (StatementBlock sb : body) {
			newBody.add(sb.deepCopy());
		}
		return newBody;
	}
}
