
package dml.parser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import dml.hops.AggBinaryOp;
import dml.hops.AggUnaryOp;
import dml.hops.BinaryOp;
import dml.hops.DataOp;
import dml.hops.Hops;
import dml.hops.LiteralOp;
import dml.hops.ParameterizedBuiltinOp;
import dml.hops.RandOp;
import dml.hops.ReorgOp;
import dml.hops.TertiaryOp;
import dml.hops.UnaryOp;
import dml.hops.Hops.AggOp;
import dml.hops.Hops.DataOpTypes;
import dml.hops.Hops.Direction;
import dml.hops.Hops.OpOp1;
import dml.hops.Hops.OpOp2;
import dml.hops.Hops.OpOp3;
import dml.hops.Hops.ParamBuiltinOp;
import dml.hops.Hops.VISIT_STATUS;
import dml.lops.Lops;
import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.runtime.instructions.CPInstructions.FunctionCallCPInstruction;
import dml.sql.sqlcontrolprogram.SQLBlockContainer;
import dml.sql.sqlcontrolprogram.SQLCleanup;
import dml.sql.sqlcontrolprogram.SQLContainerProgramBlock;
import dml.sql.sqlcontrolprogram.SQLCreateTable;
import dml.sql.sqlcontrolprogram.SQLDeclare;
import dml.sql.sqlcontrolprogram.SQLIfElseProgramBlock;
import dml.sql.sqlcontrolprogram.SQLOverwriteScalar;
import dml.sql.sqlcontrolprogram.SQLOverwriteTable;
import dml.sql.sqlcontrolprogram.SQLPrint;
import dml.sql.sqlcontrolprogram.SQLProcedureCall;
import dml.sql.sqlcontrolprogram.SQLProgram;
import dml.sql.sqlcontrolprogram.SQLProgramBlock;
import dml.sql.sqlcontrolprogram.SQLRenameTable;
import dml.sql.sqlcontrolprogram.SQLVariableAssignment;
import dml.sql.sqlcontrolprogram.SQLWhileProgramBlock;
import dml.sql.sqlcontrolprogram.SQLWithTable;
import dml.sql.sqllops.SQLLops;
import dml.sql.sqllops.SQLLops.GENERATES;
import dml.utils.HopsException;
import dml.utils.LanguageException;
import dml.utils.configuration.DMLConfig;

public class DMLTranslator {
	public static final int DMLBlockSize = 1000;
	public static DMLProgram _dmlProg;
		
	public DMLTranslator(DMLProgram dmlp) {
		_dmlProg = dmlp;
	}

	/**
	 * Validate parse tree
	 * 
	 * @throws LanguageException
	 * @throws IOException 
	 */
	public void validateParseTree(DMLProgram dmlp) throws LanguageException, IOException {
		
			
		// hnadle functions in namespaces (current program has "null" namespace)
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
		
			// for each function defined in the namespace
			for (String fname :  dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock fblock = dmlp.getFunctionStatementBlock(namespaceKey,fname);
				HashMap<String, ConstIdentifier> constVars = new HashMap<String, ConstIdentifier>();
				VariableSet vs = new VariableSet();
			
				// add the input variables for the function to input variable list
				FunctionStatement fstmt = (FunctionStatement)fblock.getStatement(0);
				if (fblock.getNumStatements() > 1){
					throw new LanguageException("FunctionStatementBlock can only have 1 FunctionStatement");
				}
			
				for (DataIdentifier currVar : fstmt.getInputParams()) {	
					vs.addVariable(currVar.getName(), currVar);
				}
				fblock.validate(dmlp, vs, constVars);
			} 
		
		}	
		
		// handle regular blocks -- "main" program
		VariableSet vs = new VariableSet();
		
		dmlp.setBlocks(StatementBlock.mergeFunctionCalls(dmlp.getBlocks(), dmlp));
		
		HashMap<String, ConstIdentifier> constVars = new HashMap<String, ConstIdentifier>();
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock sb = dmlp.getStatementBlock(i);
			vs = sb.validate(dmlp, vs, constVars);
			constVars = sb.getConstOut();
		}

		return;

	}

	public void liveVariableAnalysis(DMLProgram dmlp) throws LanguageException {
	
		// for each namespace, handle function program blocks -- forward direction
		for (String namespaceKey : dmlp.getNamespaces().keySet()) {
			for (String fname: dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock fsb = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
				
				VariableSet activeIn = new VariableSet();
				for (DataIdentifier id : fstmt.getInputParams()){
					activeIn.addVariable(id.getName(), id);
				}
				fsb.initializeforwardLV(activeIn);
			}
		}
		
		// for each namespace, handle function program blocks -- backward direction
		for (String namespaceKey : dmlp.getNamespaces().keySet()) {	
			for (String fname: dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
				
				// add output variables to liveout / activeout set
				FunctionStatementBlock fsb = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				VariableSet currentLiveOut = new VariableSet();
				VariableSet currentLiveIn = new VariableSet();
				FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
				
				for (DataIdentifier id : fstmt.getInputParams())
					currentLiveIn.addVariable(id.getName(), id);
				
				for (DataIdentifier id : fstmt.getOutputParams())
					currentLiveOut.addVariable(id.getName(), id);
					
				fsb._liveOut = currentLiveOut;
				fsb.analyze(currentLiveIn, currentLiveOut);	
			}
		} 
		
		
		// handle regular program blocks 
		VariableSet currentLiveOut = new VariableSet();
		int numBlocks = dmlp.getNumStatementBlocks();
		VariableSet activeIn = new VariableSet();
		VariableSet activeOut = new VariableSet();
		
		for (int i = 0; i < numBlocks; i++) {
			StatementBlock sb = dmlp.getStatementBlock(i);
			activeIn = sb.initializeforwardLV(activeIn);
		}

		if (numBlocks > 0){
			StatementBlock lastSb = dmlp.getStatementBlock(numBlocks - 1);
			lastSb._liveOut = new VariableSet();
			for (int i = numBlocks - 1; i >= 0; i--) {
				StatementBlock sb = dmlp.getStatementBlock(i);
				currentLiveOut = sb.analyze(currentLiveOut);
			}
		}
		return;

	}

	/**
	 * Construct Hops from parse tree
	 * 
	 * @throws ParseException
	 */
	public void constructHops(DMLProgram dmlp) throws ParseException, LanguageException {

		// for each namespace, handle function program blocks
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
		
			for (String fname: dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock current = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				FunctionStatement fstmt = (FunctionStatement)current.getStatement(0);
				
				//currentLiveIn = new VariableSet();
				//for (DataIdentifier id : fstmt.getInputParams()){
				//	currentLiveIn.addVariable(id.getName(), id);
				//}
				constructHops(current);
			}
		}
		
		
		// handle regular program blocks
		//currentLiveIn = new VariableSet();
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			constructHops(current);
			//currentLiveIn = current.liveOut();
		}
	}
	
	public void constructLops(DMLProgram dmlp) throws ParseException, LanguageException, HopsException {

		// for each namespace, handle function program blocks handle function 
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname: dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock current = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				constructLops(current);
			}
		}
		
		// handle regular program blocks
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			constructLops(current);
		}
	}
	
	public void constructSQLLops(DMLProgram dmlp) throws ParseException, HopsException {
		// handle regular program blocks
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			constructSQLLops(current);
		}	
	}
	
	public void constructLops(StatementBlock sb) throws HopsException {
		
		if (sb instanceof WhileStatementBlock){
			WhileStatement whileStmt = (WhileStatement)sb.getStatement(0);
			ArrayList<StatementBlock> body = whileStmt.getBody();
				
			if (sb.get_hops() != null && sb.get_hops().size() != 0) 
				throw new HopsException("WhileStatementBlock should not have hops");
			
			// step through stmt blocks in while stmt body
			for (StatementBlock stmtBlock : body){
				constructLops(stmtBlock);
			}
			
			// handle while stmt predicate
			Lops l = ((WhileStatementBlock) sb).getPredicateHops().constructLops();
			((WhileStatementBlock) sb).set_predicateLops(l);	
		}
		
		else if (sb instanceof IfStatementBlock){
			
			IfStatement ifStmt = (IfStatement)sb.getStatement(0);
			ArrayList<StatementBlock> ifBody = ifStmt.getIfBody();
			ArrayList<StatementBlock> elseBody = ifStmt.getElseBody();
				
			if (sb.get_hops() != null && sb.get_hops().size() != 0)
				throw new HopsException("IfStatementBlock should not have hops");
			
			// step through stmt blocks in if stmt ifBody
			for (StatementBlock stmtBlock : ifBody)
				constructLops(stmtBlock);
			
			// step through stmt blocks in if stmt elseBody
			for (StatementBlock stmtBlock : elseBody)
				constructLops(stmtBlock);
			
			// handle if stmt predicate
			Lops l = ((IfStatementBlock) sb).getPredicateHops().constructLops();
			((IfStatementBlock) sb).set_predicateLops(l);
		}
		
		else if (sb instanceof ForStatementBlock){
			ForStatement forStmt = (ForStatement)sb.getStatement(0);
			ArrayList<StatementBlock> body = forStmt.getBody();
						
			if (sb.get_hops() != null && sb.get_hops().size() != 0) 
				throw new HopsException("ForStatementBlock should not have hops");
			
			// step through stmt blocks in FOR stmt body
			for (StatementBlock stmtBlock : body)
				constructLops(stmtBlock);
			
			// handle for stmt predicate
			Hops predicateHops = ((ForStatementBlock) sb).getPredicateHops();
			if (predicateHops != null){
				Lops l = ((ForStatementBlock) sb).getPredicateHops().constructLops();
				((ForStatementBlock) sb).set_predicateLops(l);
			}
		}
		else if (sb instanceof FunctionStatementBlock){
			FunctionStatement functStmt = (FunctionStatement)sb.getStatement(0);
			ArrayList<StatementBlock> body = functStmt.getBody();
			
			if (sb.get_hops() != null && sb.get_hops().size() != 0) 
				throw new HopsException("FunctionStatementBlock should not have hops");
			
			// step through stmt blocks in while stmt body
			for (StatementBlock stmtBlock : body){
				constructLops(stmtBlock);
			}
		}
		
		// handle default case for regular StatementBlock
		else {
			
			if (sb.get_hops() == null)
				sb.set_hops(new ArrayList<Hops>());
			
			ArrayList<Lops> lops = new ArrayList<Lops>();
			for (Hops hop : sb.get_hops()) {
				lops.add(hop.constructLops());
			}
			sb.set_lops(lops);
		}
		
	} // end method
	
	public void constructSQLLops(StatementBlock sb) throws HopsException {
		if(sb.get_hops() != null)
		for (Hops hop : sb.get_hops()) {
			hop.constructSQLLOPs();
		}
		if(sb instanceof WhileStatementBlock)
		{
			WhileStatement wstmt = (WhileStatement)((WhileStatementBlock)sb).getStatement(0);
			((WhileStatementBlock)sb).getPredicateHops().constructSQLLOPs();
			for (StatementBlock bl : wstmt.getBody())
				constructSQLLops(bl);
		}
		if(sb instanceof IfStatementBlock)
		{
			IfStatement istmt = (IfStatement)((IfStatementBlock)sb).getStatement(0);
			for (StatementBlock bl : istmt.getIfBody())
				constructSQLLops(bl);
				
			((IfStatementBlock)sb).getPredicateHops().constructSQLLOPs();
		}
		
		//sb.set_lops(lops);
	}
	
	public void rewriteHopsDAG(DMLProgram dmlp, DMLConfig config) throws ParseException, LanguageException, HopsException {

		/**
		 * Rule 1: Eliminate for Transient Write DataHops to have no parents
		 * Solution: Move parent edges of Transient Write Hop to parent of
		 * its child 
		 * Reason: Transient Write not being a root messes up
		 * analysis for Lop's to Instruction translation (according to Amol)
		 */
		
		this.resetHopsDAGVisitStatus(dmlp); 
		eval_rule_RehangTransientWriteParents(dmlp);
		
		/**
		 * Rule: BlockSizeAndReblock. For all statement blocks, determine
		 * "optimal" block size, and place reblock Hops. For now, we just
		 * use BlockSize 1K x 1K and do reblock after Persistent Reads and
		 * before Persistent Writes.
		 */

		this.resetHopsDAGVisitStatus(dmlp);
		eval_rule_BlockSizeAndReblock(dmlp);
		
		/**
		 * Rule: Determine the optimal order of execution for a chain of
		 * matrix multiplications Solution: Classic Dynamic Programming
		 * Approach Currently, the approach based only on matrix dimensions
		 * Goal: To reduce the number of computations in the run-time
		 * (map-reduce) layer
		 */

		// Assumption: VISIT_STATUS of all Hops is set to NOTVISITED
		this.resetHopsDAGVisitStatus(dmlp);
		eval_rule_OptimizeMMChains(dmlp);		
	}
	
	
	// handle rule rule_OptimizeMMChains
	private void eval_rule_OptimizeMMChains(DMLProgram dmlp) throws HopsException, LanguageException {
		
		// for each namespace, handle function program blocks
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String functionName : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey,functionName);
				eval_rule_OptimizeMMChains(fsblock);	
			}
		}
		
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			eval_rule_OptimizeMMChains(current);
		}
	}
	
	private void eval_rule_OptimizeMMChains(StatementBlock current) throws HopsException {
		
		if (current instanceof FunctionStatementBlock){
			FunctionStatement fstmt = (FunctionStatement)((FunctionStatementBlock)current).getStatement(0);
			for (StatementBlock sb : fstmt.getBody())
				eval_rule_OptimizeMMChains(sb);
		}
		
		else if (current instanceof WhileStatementBlock){
			WhileStatement wstmt = (WhileStatement)((WhileStatementBlock)current).getStatement(0);
			for (StatementBlock sb : wstmt.getBody())
				eval_rule_OptimizeMMChains(sb);
		}
		
		else if (current instanceof IfStatementBlock){
			IfStatement istmt = (IfStatement)((IfStatementBlock)current).getStatement(0);
			for (StatementBlock sb : istmt.getIfBody())
				eval_rule_OptimizeMMChains(sb);
			
			for (StatementBlock sb : istmt.getElseBody())
				eval_rule_OptimizeMMChains(sb);
		}
			
		else if (current instanceof ForStatementBlock){
			ForStatement wstmt = (ForStatement)((ForStatementBlock)current).getStatement(0);
			for (StatementBlock sb : wstmt.getBody()){
				eval_rule_OptimizeMMChains(sb);
			}
		}

		else {
			// handle general case
			if (current.get_hops() != null){	
				for (Hops h : current.get_hops()) {
					if (h.get_visited() != Hops.VISIT_STATUS.DONE) {
						// Find the optimal order for the chain whose result is the current HOP
						h.rule_OptimizeMMChains();
					}
				}
			}
		} // end else	
	} // end method
	
	
	// handle rule rule_RehangTransientWriteParents
	private void eval_rule_RehangTransientWriteParents(DMLProgram dmlp) throws LanguageException, HopsException {
		
		// for each namespace, handle function program blocks
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey,fname);
				eval_rule_RehangTransientWriteParents(fsblock);
			}
		}
		
		// handle regular statement blocks in "main" function
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			eval_rule_RehangTransientWriteParents(current);
		}
	}
	
	private void eval_rule_RehangTransientWriteParents(StatementBlock current) throws HopsException {

		if (current instanceof FunctionStatementBlock){
			FunctionStatement fstmt = (FunctionStatement)current.getStatement(0);
			for (StatementBlock sb : fstmt.getBody()){
				eval_rule_RehangTransientWriteParents(sb);
			}
		}
		
		else if (current instanceof WhileStatementBlock){
			WhileStatement wstmt = (WhileStatement)((WhileStatementBlock)current).getStatement(0);
			for (StatementBlock sb : wstmt.getBody()){
				eval_rule_RehangTransientWriteParents(sb);
			}
		}
		
		else if (current instanceof IfStatementBlock){
			IfStatement istmt = (IfStatement)((IfStatementBlock)current).getStatement(0);
			for (StatementBlock sb : istmt.getIfBody()){
				eval_rule_RehangTransientWriteParents(sb);
			}
			for (StatementBlock sb : istmt.getElseBody()){
				eval_rule_RehangTransientWriteParents(sb);
			}	
		}
		
		else if (current instanceof ForStatementBlock){
			ForStatement wstmt = (ForStatement)((ForStatementBlock)current).getStatement(0);
			for (StatementBlock sb : wstmt.getBody()){
				eval_rule_RehangTransientWriteParents(sb);
			}
		}
		
		else {
			// handle general case
			if (current.get_hops() != null){
				for (Hops h : current.get_hops()) {
					h.rule_RehangTransientWriteParents(current);
				}
			}
		} // end else
		
	} // end method
	
	
	// handle rule rule_BlockSizeAndReblock
	private void eval_rule_BlockSizeAndReblock(DMLProgram dmlp) throws LanguageException, HopsException{
		
		// for each namespace, handle function statement blocks
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey,fname);
				eval_rule_BlockSizeAndReblock(fsblock);
			}
		}
		
		// handle regular statement blocks in "main" method
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			eval_rule_BlockSizeAndReblock(current);
		}
	}
	
	private void eval_rule_BlockSizeAndReblock(StatementBlock current) throws HopsException{
		
		if (current instanceof FunctionStatementBlock){
			FunctionStatement fstmt = (FunctionStatement)((FunctionStatementBlock)current).getStatement(0);
			for (StatementBlock sb : fstmt.getBody()){
				eval_rule_BlockSizeAndReblock(sb);
			}
		}
		else if (current instanceof WhileStatementBlock){
			WhileStatement wstmt = (WhileStatement)((WhileStatementBlock)current).getStatement(0);
			for (StatementBlock sb : wstmt.getBody()){
				eval_rule_BlockSizeAndReblock(sb);
			}
		}
		
		else if (current instanceof IfStatementBlock){
			IfStatement istmt = (IfStatement)((IfStatementBlock)current).getStatement(0);
			
			for (StatementBlock sb : istmt.getIfBody()){
				eval_rule_BlockSizeAndReblock(sb);
			}
			
			for (StatementBlock sb : istmt.getElseBody()){
				eval_rule_BlockSizeAndReblock(sb);
			}
			
		}
		
		else if (current instanceof ForStatementBlock){
			ForStatement wstmt = (ForStatement)((ForStatementBlock)current).getStatement(0);
			for (StatementBlock sb : wstmt.getBody()){
				eval_rule_BlockSizeAndReblock(sb);
			}
		}
		
		else {
			// handle general case
			if (current.get_hops() != null){
				for (Hops h : current.get_hops()) {
					h.rule_BlockSizeAndReblock(DMLTranslator.DMLBlockSize);
				}
			}
		} // end else
		
	} // end method
	

	//TODO Take nested blocks into account
	public void printSQLLops(DMLProgram dmlp) throws ParseException, HopsException, LanguageException
	{
		resetHopsDAGVisitStatus(dmlp);
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {		
			StatementBlock current = dmlp.getStatementBlock(i);
			if(current.get_hops() != null)
			{
				for(Hops h : current.get_hops())
				{
					if(h.get_sqllops() != null)
						h.get_sqllops().resetVisitStatus();
				}
				for(Hops h : current.get_hops())
				{
					printSQLLops(h.get_sqllops(), "");
				}
			}
		}
	}
	
	//TODO Take nested blocks into account
	public void printClusteredSQLLops(DMLProgram dmlp) throws ParseException, HopsException, LanguageException
	{
		resetSQLLopsDAGVisitStatus(dmlp);
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {		
			StatementBlock current = dmlp.getStatementBlock(i);
			if(current.get_hops() != null)
			{
				for(Hops h : current.get_hops())
				{
					printClusteredSQLLops(h.get_sqllops(), "");
				}
			}
		}
	}

	public void printSQLLops(SQLLops lops, String indendation)
	{
		if(lops.get_visited() != VISIT_STATUS.NOTVISITED)
			return;

		for(SQLLops l : lops.getInputs())
		{
			printSQLLops(l, indendation + "...");
		}
		
		lops.set_visited(VISIT_STATUS.DONE);
		System.out.println(indendation + lops.get_tableName() + ": " + lops.get_sql());
	}
	
	public void printClusteredSQLLops(SQLLops lops, String indendation)
	{
		if(lops.get_visited() != VISIT_STATUS.NOTVISITED)
			return;
		
		System.out.println(indendation + lops.get_tableName() + ": " + lops.get_sql());
		lops.set_visited(VISIT_STATUS.DONE);
		
		for(SQLLops l : lops.getInputs())
		{
			if(l.get_flag() == GENERATES.DML)
				indendation = "";
			else
				indendation = "   ";

			printClusteredSQLLops(l, indendation);
		}
	}
	
	public SQLProgram getSQLProgram(DMLProgram dmlp) throws HopsException, ParseException
	{
		SQLProgram prgr = new SQLProgram();
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {		
			StatementBlock current = dmlp.getStatementBlock(i);
			handleBlocks(prgr, current, prgr);
		}
		return prgr;
	}
	
	private void handleBlocks(SQLProgram prog, StatementBlock block, SQLBlockContainer parent) throws HopsException
	{
		if(block instanceof WhileStatementBlock)
		{
			SQLLops lop = ((WhileStatementBlock)block).getPredicateHops().get_sqllops();
			SQLWhileProgramBlock bl = new SQLWhileProgramBlock();
			bl.setPredicateTableName(lop.get_tableName());
			bl.set_predicate(getPredicate(lop));
			
			WhileStatement wstmt = (WhileStatement)block.getStatement(0);
			for(StatementBlock b : wstmt.getBody())
				handleBlocks(prog, b, bl);
			
			parent.get_blocks().add(bl);
		}
		else if(block instanceof IfStatementBlock)
		{
			SQLLops lop = ((IfStatementBlock)block).getPredicateHops().constructSQLLOPs();
			SQLIfElseProgramBlock bl = new SQLIfElseProgramBlock();
			bl.setPredicateTableName(lop.get_tableName());
			bl.set_predicate(getPredicate(lop));
			
			IfStatement ifstmt = (IfStatement)block.getStatement(0);
			
			SQLContainerProgramBlock ifblock = new SQLContainerProgramBlock();
			for(StatementBlock b : ifstmt.getIfBody())
			{
				handleBlocks(prog, b, ifblock);
			}
			bl.set_ifbody(ifblock);
			if(ifstmt.getElseBody().size() > 0)
			{
				SQLContainerProgramBlock elseblock = new SQLContainerProgramBlock();
				for(StatementBlock b : ifstmt.getElseBody())
				{
					handleBlocks(prog, b, elseblock);
				}
				bl.set_elsebody(elseblock);
			}
			parent.get_blocks().add(bl);
		}
		//Handle a normal block
		else
		{
			SQLProgramBlock b = getSQLBlock(block, prog, parent);
			parent.get_blocks().add(b);
		}
	}
	
	/*
	 * This creates a single SELECT with subqueries in WITH clauses to be used in a predicate
	 */
	private String getPredicate(SQLLops lop)
	{
		StringBuffer sb = new StringBuffer();
			
		//We need to count the subqueries in order to find positions of commas
		int sqls = 0;
		int countedSQLs = 0;
		
		for(SQLLops s : lop.getInputs())
			if(s.get_flag() != GENERATES.NONE && s.get_flag() != GENERATES.PROC)
				sqls += 1;
			
		if(sqls > 0)
		{
			sb.append("WITH ");
			
			for(SQLLops l : lop.getInputs())
			{
				if(l.get_flag() != GENERATES.NONE)
				{
					countedSQLs++;
					getSQLStatement(l, sb, countedSQLs != sqls);
				}
			}
		}
		sb.append(lop.get_sql());
		return sb.toString();
	}
	
	/**
	 * Append all sub-SQLLops and then the SQLLop to a StringBuffer
	 * @param lop
	 * @param sb
	 * @param addComma
	 */
	private void getSQLStatement(SQLLops lop, StringBuffer sb, boolean addComma)
	{
		for(int i = 0; i < lop.getInputs().size(); i++)
		{
			SQLLops l = lop.getInputs().get(i);
			if(l.get_flag() != GENERATES.NONE && l.get_flag() != GENERATES.PROC)
				getSQLStatement(l, sb, true);
		}
		
		sb.append(lop.get_tableName());
		sb.append(" AS ( ");
		sb.append(lop.get_sql());
		sb.append(" )");
		
		if(addComma)
			sb.append(", ");
	}
	
	private SQLProgramBlock getSQLBlock(StatementBlock block, SQLProgram prog, SQLBlockContainer parent) throws HopsException
	{
		SQLProgramBlock b = new SQLProgramBlock();

		if(block.get_hops() != null)
		{
			for(Hops h : block.get_hops())
			{
				/*if(h.get_sqllops() == null)
					System.out.print("ERROR");
				h.get_sqllops().resetVisitStatus();*/
				h.constructSQLLOPs().resetVisitStatus();
				//TODO make it work without constructSQL!
			}
			for(Hops h : block.get_hops())
				getCreateTables(h.get_sqllops(), b, null, prog);
		}
		return b;
	}
	
	private void getCreateTables(SQLLops lop, SQLProgramBlock block, SQLCreateTable parent, SQLProgram prog)
	{
		if(lop.get_visited() != VISIT_STATUS.NOTVISITED)
			return;

		if(lop.get_flag() == GENERATES.PROC)
		{
			block.get_creates().add(new SQLProcedureCall(lop.get_sql()));
			
			SQLCleanup clean = new SQLCleanup();
			clean.set_tableName(lop.get_tableName());
			prog.get_cleanups().add(clean);
		}
		//This block handles all SQLLops that create or assign something or do not have an output
		else if((lop.get_flag() != GENERATES.NONE  && lop.get_flag() != GENERATES.SQL) || lop.getOutputs().size() == 0)
		{
			if(lop.get_dataType() == DataType.SCALAR)
			{
				String name = lop.get_tableName();
				boolean exists = false;
				for(SQLDeclare d : prog.get_variableDeclarations())
					if(d.get_variable_name().equals(name))
					{
						exists = true;
						break;
					}
				
				//If the scalar needs a variable that has not been used yet,
				//this variable is added to the list of declares
				if(!exists && lop.get_flag() != GENERATES.DML_PERSISTENT)
				{
					String vt = "int8";
					if(lop.get_valueType() == ValueType.DOUBLE)
						vt = "double precision";
					else if(lop.get_valueType() == ValueType.STRING)
						vt = "varchar(999)";
					else if(lop.get_valueType() == ValueType.BOOLEAN)
						vt = "boolean";
					prog.get_variableDeclarations().add(new SQLDeclare(name, vt));
				}
				//A print is always the top of a HOPs DAG and creates a RAISE NOTICE statement
				if(lop.get_flag() == GENERATES.PRINT)
				{
					SQLPrint print = new SQLPrint();
					print.set_selectStatement(lop.get_sql());
					print.set_tableName(lop.get_tableName());
					
					for(SQLLops l : lop.getInputs())
						getCreateTables(l, block, print, prog);
					
					block.get_creates().add(print);
				}
				//Filter out Transient read with a transient write as value
				// Like: max_iterations := max_iterations;
				else if(!lop.get_sql().equals(lop.get_tableName()))
				{
					SQLCreateTable create;

					if(lop.get_flag() == GENERATES.DML_PERSISTENT)
						create = new SQLOverwriteScalar();
					else
					{
						// Here we have to differentiate between cases like a = b or a = 12 and cases like
						// a = SELECT b + 3 AS sval or a = SELECT sqrt(b) AS sval
						// lop.get_flag() == GENERATES.DML_TRANSIENT would be enough, because DML SQLLops always have a
						// select statement and whenever there is a transient write the child is DML
						// The rest is just to make this work in case this is changed later
						boolean hasSelect = lop.getInputs().size() > 1  ||
						!( lop.getInputs().get(0).get_dataType() == DataType.SCALAR
							&& ( lop.getInputs().get(0).get_flag() == GENERATES.NONE
								 || lop.getInputs().get(0).get_flag() == GENERATES.DML ||
								 lop.getInputs().get(0).get_flag() == GENERATES.DML_TRANSIENT)			 
								 && lop.get_flag() == GENERATES.DML_TRANSIENT);
						
						create = new SQLVariableAssignment(lop.get_valueType(), hasSelect);
					}
					create.set_selectStatement(lop.get_sql());
					create.set_tableName(lop.get_tableName());
					
					//Handle children
					for(SQLLops l : lop.getInputs())
						getCreateTables(l, block, create, prog);
					
					//SQL overwrites something and must be executed at the end of the block
					if(lop.get_flag() == GENERATES.DML_PERSISTENT || lop.get_flag() == GENERATES.DML_TRANSIENT)
						block.get_writes().add(create);
					//SQL does not overwrite anything and can therefore come before anything else
					else
						block.get_creates().add(create);
				}
			}
			//Handle matrices
			else
			{
				SQLCreateTable ct;
				//Any transient stuff can be cleaned up at the end of the program
				if(lop.get_flag() == GENERATES.DML_TRANSIENT)
				{
					//This can be cleaned up at the end of the program
					if(!prog.cleansUp(lop.get_tableName()))
					{
						SQLCleanup clean = new SQLCleanup();
						clean.set_tableName(lop.get_tableName());
						prog.get_cleanups().add(clean);
					}
				}
				// Special case: if input is DML, it would be dropped, except we rename it first,
				// that saves us some copy operations
				// Without this: create a SQLLops as DML, copy it to transient variable, drop it
				// With this: create SQLLops as DML, if it has only one direct "Write"-Parent,
				// rename it to this ones table name
				if(lop.get_flag() == GENERATES.DML_TRANSIENT ||
						lop.get_flag() == GENERATES.DML_PERSISTENT)
				{
					SQLLops input = lop.getInputs().get(0);
					if(input.get_flag() == GENERATES.DML)
					{
						int writes = 0;
						for(SQLLops o : input.getOutputs())
						{
							if(o.get_flag() == GENERATES.DML_PERSISTENT
									|| o.get_flag() == GENERATES.DML_TRANSIENT)
								writes++;
						}
						if(writes == 1)
						{
							SQLRenameTable ren = new SQLRenameTable(input.get_tableName(), lop.get_tableName());
							block.get_renames().add(ren);
							
							//Handle input and create objects for that
							for(SQLLops l : lop.getInputs())
								getCreateTables(l, block, null, prog);
							
							lop.set_visited(VISIT_STATUS.DONE);
							return;
						}
					}
				}
				
				if(lop.get_flag() == GENERATES.DML_TRANSIENT)
				{
					//TODO: Check if this works for all cases
					//Removes "copying" from one table to itself
					if(lop.getInputs().get(0).get_tableName().equals(lop.get_tableName()))
						return;

					//boolean isView = lop.getInputs().get(0).get_flag() == GENERATES.NONE;
					ct = new SQLOverwriteTable(true);
				}
				
				else if(lop.get_flag() == GENERATES.DML_PERSISTENT)
				{
					ct = new SQLOverwriteTable(false);
				}
				else
				{
					ct = new SQLCreateTable();
					
					//This can be cleaned up at the end of the block
					SQLCleanup clean = new SQLCleanup();
					clean.set_tableName(lop.get_tableName());
					block.get_cleanups().add(clean);
				}
				
				ct.set_selectStatement(lop.get_sql());
				ct.set_tableName(lop.get_tableName());
				
				//Handle input and create objects for that
				for(SQLLops l : lop.getInputs())
					getCreateTables(l, block, ct, prog);
				
				//SQL overwrites something and must be executed at the end of the block
				if(lop.get_flag() == GENERATES.DML_PERSISTENT || lop.get_flag() == GENERATES.DML_TRANSIENT)
					block.get_writes().add(ct);
				//SQL does not overwrite anything and can therefore come before anything else
				else
					block.get_creates().add(ct);
			}
		}
		//This block handles all SQLLops that create temporary SQL inside of a statement
		else if(lop.get_flag() == GENERATES.SQL && parent != null)
		{
			SQLWithTable with = new SQLWithTable();
			with.set_selectStatement(lop.get_sql());
			with.set_tableName(lop.get_tableName());
			
			for(SQLLops l : lop.getInputs())
				getCreateTables(l, block, parent, prog);
			
			parent.get_withs().add(with);
		}

		lop.set_visited(VISIT_STATUS.DONE);
	}
	
	public void printLops(DMLProgram dmlp) throws ParseException, LanguageException, HopsException {

		// for each namespace, handle function program blocks
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey,fname);
				printLops(fsblock);
			}
		}
		
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {		
			StatementBlock current = dmlp.getStatementBlock(i);
			printLops(current);
		}
	}
			
	public void printLops(StatementBlock current) throws ParseException, HopsException {
	
		ArrayList<Lops> lopsDAG = current.get_lops();
		
		System.out.println("********************** LOPS DAG FOR BLOCK *******************");
		
		if (current instanceof FunctionStatementBlock) {
			if (current.getNumStatements() > 1)
				System.out.println("error -- function stmt block has more than 1 stmt");
			FunctionStatement fstmt = (FunctionStatement)current.getStatement(0);
			for (StatementBlock child : fstmt.getBody()){
				printLops(child);
			}
		}
		
		if (current instanceof WhileStatementBlock) {
			
			// print predicate lops 
			WhileStatementBlock wstb = (WhileStatementBlock) current; 
			Hops predicateHops = ((WhileStatementBlock) current).getPredicateHops();
			System.out.println("********************** PREDICATE LOPS *******************");
			Lops predicateLops = predicateHops.get_lops();
			if (predicateLops == null)
				predicateLops = predicateHops.constructLops();
			predicateLops.printMe();
			
			if (wstb.getNumStatements() > 1)
				throw new HopsException("WhileStatementBlock has more than 1 statement");
			WhileStatement ws = (WhileStatement)wstb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				printLops(sb);
			}
		}

		if (current instanceof IfStatementBlock) {
			
			// print predicate lops 
			IfStatementBlock istb = (IfStatementBlock) current; 
			Hops predicateHops = ((IfStatementBlock) current).getPredicateHops();
			System.out.println("********************** PREDICATE LOPS *******************");
			Lops predicateLops = predicateHops.get_lops();
			if (predicateLops == null)
				predicateLops = predicateHops.constructLops();
			predicateLops.printMe();
			
			if (istb.getNumStatements() > 1)
				throw new HopsException("IfStatmentBlock has more than 1 statement");
			IfStatement is = (IfStatement)istb.getStatement(0);
			
			System.out.println("**** LOPS DAG FOR IF BODY ****");
			for (StatementBlock sb : is.getIfBody()){
				printLops(sb);
			}
			if (is.getElseBody().size() > 0){
				System.out.println("**** LOPS DAG FOR IF BODY ****");
				for (StatementBlock sb : is.getElseBody()){
					printLops(sb);
				}
			}
		}
			
		if (current instanceof ForStatementBlock) {
			
			// print predicate lops 
			ForStatementBlock fsb = (ForStatementBlock) current; 
			Hops predicateHops = ((ForStatementBlock) current).getPredicateHops();
			System.out.println("********************** PREDICATE LOPS *******************");
			if (predicateHops != null){
				Lops predicateLops = predicateHops.get_lops();
				if (predicateLops != null) {
					predicateLops = predicateHops.constructLops();
					predicateLops.printMe();
				}
			}
			
			if (fsb.getNumStatements() > 1)
				throw new HopsException("ForStatementBlock has more than 1 statement");
			ForStatement ws = (ForStatement)fsb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				printLops(sb);
			}
		}
		
		if (current instanceof CVStatementBlock) {
			System.out.println("********************** PARTITION LOPS *******************");
			((CVStatementBlock) current).getPartitionHop().get_lops().printMe();

		}

		if (current instanceof ELStatementBlock) {
			System.out.println("********************** PARTITION LOPS *******************");
			((ELStatementBlock) current).getPartitionHop().get_lops().printMe();

		}
		
		if (current instanceof ELUseStatementBlock) {
			System.out.println("********************** PARTITION LOPS *******************");
			((ELUseStatementBlock) current).getPartitionHop().get_lops().printMe();

		}
		
		
		if (lopsDAG != null && lopsDAG.size() > 0) {
			Iterator<Lops> iter = lopsDAG.iterator();
			while (iter.hasNext()) {
				System.out.println("********************** OUTPUT LOPS *******************");
				iter.next().printMe();
			}
		}
	}
	

	public void printHops(DMLProgram dmlp) throws ParseException, LanguageException, HopsException {

		// for each namespace, handle function program blocks
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey,fname);
				printHops(fsblock);
			}
		}
		
		// hand
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			printHops(current);
		}
	}
	
	public void printHops(StatementBlock current) throws ParseException, HopsException {
		
		ArrayList<Hops> hopsDAG = current.get_hops();
		System.out.println("********************** HOPS DAG FOR BLOCK *******************");
		
		if (current instanceof FunctionStatementBlock) {
			if (current.getNumStatements() > 1)
				System.out.println("error -- function stmt block has more than 1 stmt");
			FunctionStatement fstmt = (FunctionStatement)current.getStatement(0);
			for (StatementBlock child : fstmt.getBody()){
				printHops(child);
			}
		}
	
		if (current instanceof WhileStatementBlock) {
			
			// print predicate hops
			WhileStatementBlock wstb = (WhileStatementBlock) current; 
			Hops predicateHops = wstb.getPredicateHops();
			System.out.println("********************** PREDICATE HOPS *******************");
			predicateHops.printMe();
		
			if (wstb.getNumStatements() > 1)
				System.out.println("error -- while stmt block has more than 1 stmt");
			WhileStatement ws = (WhileStatement)wstb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				printHops(sb);
			}
		}
	
		if (current instanceof IfStatementBlock) {
			
			// print predicate hops
			IfStatementBlock istb = (IfStatementBlock) current; 
			Hops predicateHops = istb.getPredicateHops();
			System.out.println("********************** PREDICATE HOPS *******************");
			predicateHops.printMe();
		
			if (istb.getNumStatements() > 1)
				System.out.println("error -- while stmt block has more than 1 stmt");
			IfStatement is = (IfStatement)istb.getStatement(0);
			
			for (StatementBlock sb : is.getIfBody()){
				printHops(sb);
			}
			
			for (StatementBlock sb : is.getElseBody()){
				printHops(sb);
			}
		}
		
		
		if (current instanceof ForStatementBlock) {
			
			// print predicate hops
			ForStatementBlock wstb = (ForStatementBlock) current; 
			Hops predicateHops = wstb.getPredicateHops();
			System.out.println("********************** PREDICATE HOPS *******************");
			if (predicateHops != null) predicateHops.printMe();
		
			if (wstb.getNumStatements() > 1)
				System.out.println("error -- while stmt block has more than 1 stmt");
			ForStatement ws = (ForStatement)wstb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				printHops(sb);
			}
		}
		
		if (current instanceof CVStatementBlock) {
			System.out.println("********************** CROSSVAL HOPS *******************");
			((CVStatementBlock) current).getPartitionHop().printMe();

		}
		
		if (current instanceof ELStatementBlock) {
			System.out.println("********************** CROSSVAL HOPS *******************");
			((ELStatementBlock) current).getPartitionHop().printMe();

		}
		
		if (current instanceof ELUseStatementBlock) {
			System.out.println("********************** CROSSVAL HOPS *******************");
			((ELUseStatementBlock) current).getPartitionHop().printMe();

		}
		
		if (hopsDAG != null && hopsDAG.size() > 0) {
			// hopsDAG.iterator().next().printMe();
			Iterator<Hops> iter = hopsDAG.iterator();
			while (iter.hasNext()) {
				System.out.println("********************** OUTPUT HOPS *******************");
				iter.next().printMe();
			}
		}
	}

	public void resetHopsDAGVisitStatus(DMLProgram dmlp) throws ParseException, LanguageException, HopsException {

		// for each namespace, handle function program blocks -- forward direction
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				resetHopsDAGVisitStatus(fsblock);
			}
		}
		
		// handle statement blocks in "main" method
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			resetHopsDAGVisitStatus(current);
		}
	}
			
	public void resetHopsDAGVisitStatus(StatementBlock current) throws ParseException, HopsException {
	
		ArrayList<Hops> hopsDAG = current.get_hops();
		if (hopsDAG != null && hopsDAG.size() > 0) {
			Iterator<Hops> iter = hopsDAG.iterator();
			while (iter.hasNext()) {
				iter.next().resetVisitStatus();
			}
		}
		
		if (current instanceof FunctionStatementBlock) {
			
			FunctionStatement fstmt = (FunctionStatement)current.getStatement(0);
			for (StatementBlock sb : fstmt.getBody()){
				resetHopsDAGVisitStatus(sb);
			}
		}
		
		if (current instanceof WhileStatementBlock) {
			// handle predicate
			WhileStatementBlock wstb = (WhileStatementBlock) current;
			wstb.getPredicateHops().resetVisitStatus();
		
			if (wstb.getNumStatements() > 1)
				System.out.println("error -- while stmt block has more than 1 stmt");
			WhileStatement ws = (WhileStatement)wstb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				resetHopsDAGVisitStatus(sb);
			}
		}
		
		if (current instanceof IfStatementBlock) {
			// handle predicate
			IfStatementBlock istb = (IfStatementBlock) current;
			istb.getPredicateHops().resetVisitStatus();
		
			if (istb.getNumStatements() > 1)
				System.out.println("error -- if stmt block has more than 1 stmt");
			IfStatement is = (IfStatement)istb.getStatement(0);
			
			for (StatementBlock sb : is.getIfBody()){
				resetHopsDAGVisitStatus(sb);
			}
			for (StatementBlock sb : is.getElseBody()){
				resetHopsDAGVisitStatus(sb);
			}
		}
		
		if (current instanceof ForStatementBlock) {
			// handle predicate
			ForStatementBlock fstb = (ForStatementBlock) current;
			if (fstb.getPredicateHops() != null) 
				fstb.getPredicateHops().resetVisitStatus();
		
			if (fstb.getNumStatements() > 1)
				System.out.println("error -- while stmt block has more than 1 stmt");
			ForStatement ws = (ForStatement)fstb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				resetHopsDAGVisitStatus(sb);
			}
		}
	}
	
	public void resetSQLLopsDAGVisitStatus(DMLProgram dmlp) throws ParseException, HopsException, LanguageException {

		// for each namespace, handle function program blocks
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				resetSQLLopsDAGVisitStatus(fsblock);
			}
		}
		
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			resetSQLLopsDAGVisitStatus(current);
		}
	}
			
	public void resetSQLLopsDAGVisitStatus(StatementBlock current) throws ParseException, HopsException {
	
		ArrayList<Hops> hopsDAG = current.get_hops();
		if (hopsDAG != null && hopsDAG.size() > 0) {
			Iterator<Hops> iter = hopsDAG.iterator();
			while (iter.hasNext()) {
				iter.next().get_sqllops().resetVisitStatus();
			}
		}
		
		if (current instanceof FunctionStatementBlock) {
			
			FunctionStatement fstmt = (FunctionStatement)current.getStatement(0);
			for (StatementBlock sb : fstmt.getBody()){
				resetSQLLopsDAGVisitStatus(sb);
			}
		}
		
		if (current instanceof WhileStatementBlock) {
			// handle predicate
			WhileStatementBlock wstb = (WhileStatementBlock) current;
			wstb.getPredicateHops().get_sqllops().resetVisitStatus();
		
			if (wstb.getNumStatements() > 1)
				System.out.println("error -- while stmt block has more than 1 stmt");
			WhileStatement ws = (WhileStatement)wstb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				resetSQLLopsDAGVisitStatus(sb);
			}
		}
		
		if (current instanceof IfStatementBlock) {
			// handle predicate
			IfStatementBlock istb = (IfStatementBlock) current;
			istb.getPredicateHops().get_sqllops().resetVisitStatus();
		
			if (istb.getNumStatements() > 1)
				System.out.println("error -- if stmt block has more than 1 stmt");
			IfStatement is = (IfStatement)istb.getStatement(0);
			
			for (StatementBlock sb : is.getIfBody()){
				resetSQLLopsDAGVisitStatus(sb);
			}
			for (StatementBlock sb : is.getElseBody()){
				resetSQLLopsDAGVisitStatus(sb);
			}
		}
		
		if (current instanceof ForStatementBlock) {
			// handle predicate
			ForStatementBlock wstb = (ForStatementBlock) current;
			wstb.getPredicateHops().get_sqllops().resetVisitStatus();
		
			if (wstb.getNumStatements() > 1)
				System.out.println("error -- while stmt block has more than 1 stmt");
			ForStatement ws = (ForStatement)wstb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				resetSQLLopsDAGVisitStatus(sb);
			}
		}
	}

	public void resetLopsDAGVisitStatus(DMLProgram dmlp) throws HopsException, LanguageException {
		
		// for each namespace, handle function program blocks
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				resetLopsDAGVisitStatus(fsblock);
			}
		}
		
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			resetLopsDAGVisitStatus(current);
		}
	}
	
	public void resetLopsDAGVisitStatus(StatementBlock current) throws HopsException {
		
		ArrayList<Hops> hopsDAG = current.get_hops();

		if (hopsDAG != null && hopsDAG.size() > 0) {
			Iterator<Hops> iter = hopsDAG.iterator();
			while (iter.hasNext()){
				Hops currentHop = iter.next();
				currentHop.get_lops().resetVisitStatus();
			}
		}
		
		if (current instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock) current;
			FunctionStatement fs = (FunctionStatement)fsb.getStatement(0);
			
			for (StatementBlock sb : fs.getBody()){
				resetLopsDAGVisitStatus(sb);
			}
		}
		
		
		if (current instanceof WhileStatementBlock) {
			WhileStatementBlock wstb = (WhileStatementBlock) current;
			wstb.get_predicateLops().resetVisitStatus();
			if (wstb.getNumStatements() > 1)
				System.out.println("error -- while stmt block has more than 1 stmt");
			WhileStatement ws = (WhileStatement)wstb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				resetLopsDAGVisitStatus(sb);
			}
		}
		
		if (current instanceof IfStatementBlock) {
			IfStatementBlock istb = (IfStatementBlock) current;
			istb.get_predicateLops().resetVisitStatus();
			if (istb.getNumStatements() > 1)
				System.out.println("error -- if stmt block has more than 1 stmt");
			IfStatement is = (IfStatement)istb.getStatement(0);
			
			for (StatementBlock sb : is.getIfBody()){
				resetLopsDAGVisitStatus(sb);
			}
			
			for (StatementBlock sb : is.getElseBody()){
				resetLopsDAGVisitStatus(sb);
			}
		}
		
		if (current instanceof ForStatementBlock) {
			ForStatementBlock fsb = (ForStatementBlock) current;
			if (fsb.get_predicateLops() != null)
				fsb.get_predicateLops().resetVisitStatus();
			if (fsb.getNumStatements() > 1)
				System.out.println("error -- for stmt block has more than 1 stmt");
			ForStatement ws = (ForStatement)fsb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				resetLopsDAGVisitStatus(sb);
			}
		}
	}


	public void constructHops(StatementBlock sb) throws ParseException, LanguageException {

		if (sb instanceof WhileStatementBlock) {
			constructHopsForWhileControlBlock((WhileStatementBlock) sb);
			return;
		}

		if (sb instanceof IfStatementBlock) {
			constructHopsForIfControlBlock((IfStatementBlock) sb);
			return;
		}
		
		if (sb instanceof ForStatementBlock) {
			constructHopsForForControlBlock((ForStatementBlock) sb);
			return;
		}
		
		if (sb instanceof CVStatementBlock) {
			((CVStatementBlock) sb).get_hops();
			return;
		}
		
		if (sb instanceof ELStatementBlock) {
			((ELStatementBlock) sb).get_hops();
			return;
		}
		
		if (sb instanceof ELUseStatementBlock) {
			((ELUseStatementBlock) sb).get_hops();
			return;
		}
		
		if (sb instanceof FunctionStatementBlock) {
			constructHopsForFunctionControlBlock((FunctionStatementBlock) sb);
			return;
		}
		
		
		HashMap<String, Hops> _ids = new HashMap<String, Hops>();
		ArrayList<Hops> output = new ArrayList<Hops>();

		VariableSet liveIn = sb.liveIn();
		VariableSet liveOut = sb.liveOut();
		VariableSet updatedLiveOut = new VariableSet();

		HashMap<String, Integer> liveOutToTemp = new HashMap<String, Integer>();
		for (int i = 0; i < sb.getNumStatements(); i++) {
			Statement current = sb.getStatement(i);
			if (current instanceof AssignmentStatement) {
				AssignmentStatement as = (AssignmentStatement) current;
				DataIdentifier target = as.getTarget();
				if (liveOut.containsVariable(target.getName())) {
					liveOutToTemp.put(target.getName(), new Integer(i));
				}
			}
			if (current instanceof MultiAssignmentStatement) {
				MultiAssignmentStatement mas = (MultiAssignmentStatement) current;
				
				for (DataIdentifier target : mas.getTargetList()){
					if (liveOut.containsVariable(target.getName())) {
						liveOutToTemp.put(target.getName(), new Integer(i));
					}
				}
			}
			
			if (current instanceof RandStatement) {
				RandStatement rs = (RandStatement) current;
				DataIdentifier target = rs.getIdentifier();
				if (liveOut.containsVariable(target.getName())) {
					liveOutToTemp.put(target.getName(), new Integer(i));
				}
			}
			if (current instanceof InputStatement) {
				InputStatement is = (InputStatement) current;
				DataIdentifier target = is.getIdentifier();
				if (liveOut.containsVariable(target.getName())) {
					liveOutToTemp.put(target.getName(), new Integer(i));
				}
			}
		}

		if (liveIn.getVariables().values().size() > 0) {
			
			for (String varName : liveIn.getVariables().keySet()) {
				DataIdentifier var = liveIn.getVariables().get(varName);
				DataOp read = new DataOp(var.getName(), var.getDataType(), var.getValueType(), DataOpTypes.TRANSIENTREAD, null, var.getDim1(), var.getDim2(),(int) var.getRowsInBlock(), (int) var.getColumnsInBlock());
				_ids.put(varName, read);
			}
		}

		// Handle liveout variables that are not updated in this block

		if (liveOut.getVariables().values().size() > 0) {
			for (String varName : liveOut.getVariables().keySet()) {
				Integer statementId = liveOutToTemp.get(varName);
				if (statementId == null) {
					Hops varHop = _ids.get(varName);

					DataOp transientwrite = new DataOp(varName, varHop.get_dataType(), varHop.get_valueType(), varHop, DataOpTypes.TRANSIENTWRITE, null);
					transientwrite.setOutputParams(varHop.get_dim1(), varHop.get_dim2(), varHop.get_rows_per_block(), varHop.get_cols_per_block());
					output.add(transientwrite);
				}
			}
		}

		for (int i = 0; i < sb.getNumStatements(); i++) {
			Statement current = sb.getStatement(i);

			if (current instanceof InputStatement) {
				InputStatement is = (InputStatement) current;
				DataIdentifier ds = is.getIdentifier();
				DataOp read = new DataOp(is.getIdentifier().getName(), is.getIdentifier().getDataType(), is.getIdentifier().getValueType(), DataOpTypes.PERSISTENTREAD, is.getFilename(), ds.getDim1(), ds.getDim2(), (int) ds.getRowsInBlock(), (int) ds.getColumnsInBlock());
				String formatName = is.getFormatName();
				read.setFormatType(Expression.convertFormatType(formatName));
				_ids.put(is.getIdentifier().getName(), read);
	
				DataIdentifier target = is.getIdentifier();
				Integer statementId = liveOutToTemp.get(target.getName());
				if ((statementId != null) && (statementId.intValue() == i)) {
					DataOp transientwrite = new DataOp(target.getName(), target.getDataType(), target.getValueType(), read, DataOpTypes.TRANSIENTWRITE, null);
					transientwrite.setOutputParams(read.get_dim1(), read.get_dim2(), read.get_rows_per_block(), read.get_cols_per_block());
					updatedLiveOut.addVariable(target.getName(), target);
					output.add(transientwrite);
				}
			}

			if (current instanceof OutputStatement) {
				OutputStatement os = (OutputStatement) current;
				String name = os.getIdentifier().getName();
				DataOp write = new DataOp(name, os.getIdentifier().getDataType(), os.getIdentifier().getValueType(), _ids.get(name), DataOpTypes.PERSISTENTWRITE, os.getFilename());
				write.setOutputParams(_ids.get(name).get_dim1(), _ids.get(name).get_dim2(), _ids.get(name).get_rows_per_block(), _ids.get(name).get_cols_per_block());
				String formatName = os.getFormatName();
				write.setFormatType(Expression.convertFormatType(formatName));
				setIdentifierParams(write, os.getIdentifier());
				output.add(write);
			}

			/**
			if (current instanceof PrintStatement) {
				PrintStatement ps = (PrintStatement) current;
				DataIdentifier id = ps.getIdentifier();
				DataIdentifier target = createTarget();
				target.setDataType(DataType.SCALAR);
				LiteralOp lop = new LiteralOp(target.getName(), ps.getMessage());

				if (id != null) {
					String name = ps.getIdentifier().getName();
					Hops idHop = _ids.get(name);
					Hops printHop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(),OpOp2.PRINT, idHop, lop);
					output.add(printHop);
				} else {
					try {
						Hops printHop = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), Hops.OpOp1.PRINT, lop);
						output.add(printHop);
					} catch ( HopsException e ) {
						e.printStackTrace();
					}
				}
			}
			**/
			if (current instanceof PrintStatement) {
				PrintStatement ps = (PrintStatement) current;
				Expression source = ps.getExpression();
				
				DataIdentifier target = createTarget();
				target.setDataType(DataType.SCALAR);
				target.setValueType(ValueType.STRING);
				
				Hops ae = processExpression(source, target, _ids);
				
				try {
					Hops printHop = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), Hops.OpOp1.PRINT2, ae);
					output.add(printHop);
				} catch ( HopsException e ) {
					e.printStackTrace();
				}
			}

			if (current instanceof AssignmentStatement) {
							
				AssignmentStatement as = (AssignmentStatement) current;
				DataIdentifier target = as.getTarget();
				Expression source = as.getSource();
				
				if (!(source instanceof FunctionCallIdentifier)){
				
					Hops ae = processExpression(source, target, _ids);
					_ids.put(target.getName(), ae);
					target.setProperties(source.getOutput());
					Integer statementId = liveOutToTemp.get(target.getName());
					if ((statementId != null) && (statementId.intValue() == i)) {
						DataOp transientwrite = new DataOp(target.getName(), target.getDataType(), target.getValueType(), ae, DataOpTypes.TRANSIENTWRITE, null);
						transientwrite.setOutputParams(ae.get_dim1(), ae.get_dim2(), ae.get_rows_per_block(), ae.get_cols_per_block());
						updatedLiveOut.addVariable(target.getName(), target);
						output.add(transientwrite);
					}
				}
				else {
					
					/**
					 * Instruction format extFunct:::[FUNCTION NAME]:::[num input params]:::[num output params]:::[list of delimited input params ]:::[list of delimited ouput params]
					 * These are the "bound names" for the inputs / outputs.  For example, out1 = foo(in1, in2) yields
					 * extFunct:::foo:::2:::1:::in1:::in2:::out1
					 * 
					 */

					FunctionCallIdentifier fci = (FunctionCallIdentifier) source;
					FunctionStatement fstmt = (FunctionStatement)this._dmlProg.getFunctionStatementBlock(fci.getNamespace(), fci.getName()).getStatement(0);
					StringBuilder inst = new StringBuilder();

					inst.append("extfunct");
					inst.append(Lops.OPERAND_DELIMITOR);
					
					inst.append(fstmt.getName());
					inst.append(Lops.OPERAND_DELIMITOR);
			
					inst.append(fstmt._inputParams.size());
					inst.append(Lops.OPERAND_DELIMITOR);

					inst.append(fstmt._outputParams.size());
					inst.append(Lops.OPERAND_DELIMITOR);
					
					// TODO: DRB: make assumption that function call DOES NOT contain complex expressions
					ArrayList<String> inParamNames = new ArrayList<String>();
					for (Expression paramName : fci.getParamExpressions()){
						inst.append(paramName.toString());
						inst.append(Lops.OPERAND_DELIMITOR);
						inParamNames.add(paramName.toString());
					}
					
					ArrayList<String> outParamNames = new ArrayList<String>();
					outParamNames.add(target.getName());
					inst.append(target.getName());
					
					// create the instruction for the function call
					sb.setFunctionCallInst(new FunctionCallCPInstruction(fci.getName(), inParamNames,outParamNames, inst.toString()));
					
				}
			}

			else if (current instanceof MultiAssignmentStatement) {
				MultiAssignmentStatement mas = (MultiAssignmentStatement) current;
				ArrayList<DataIdentifier> target = mas.getTargetList();
				Expression source = mas.getSource();
				
				FunctionCallIdentifier fci = (FunctionCallIdentifier) source;
				FunctionStatement fstmt = (FunctionStatement)this._dmlProg.getFunctionStatementBlock(fci.getNamespace(),fci.getName()).getStatement(0);
				StringBuilder inst = new StringBuilder();

				inst.append("extfunct");
				inst.append(Lops.OPERAND_DELIMITOR);

				inst.append(fstmt.getName());
				inst.append(Lops.OPERAND_DELIMITOR);
				
				inst.append(fstmt.getInputParams().size());
				inst.append(Lops.OPERAND_DELIMITOR);

				inst.append(fstmt.getOutputParams().size());
				inst.append(Lops.OPERAND_DELIMITOR);
				
				// TODO: DRB: make assumption that function call DOES NOT contain complex expressions
				ArrayList<String> inParamNames = new ArrayList<String>();
				for (Expression paramName : fci.getParamExpressions()){
					inst.append(paramName.toString());
					inst.append(Lops.OPERAND_DELIMITOR);
					inParamNames.add(paramName.toString());
				}
				
				ArrayList<String> outParamNames = new ArrayList<String>();
				for (int j=0; j<mas.getTargetList().size();j++){
					DataIdentifier curr = mas.getTargetList().get(j);
					outParamNames.add(curr.getName());
					inst.append(curr.getName());
					if (j < mas.getTargetList().size() - 1)
						inst.append(Lops.OPERAND_DELIMITOR);
				}
				// create the instruction for the function call
				sb.setFunctionCallInst(new FunctionCallCPInstruction(fci.getName(), inParamNames,outParamNames, inst.toString()));
			}
			
			if (current instanceof RandStatement) {
				RandStatement rs = (RandStatement) current;
				DataIdentifier target = rs.getIdentifier();
				Hops rand = new RandOp(target, rs.getMinValue(), rs.getMaxValue(), rs.getSparsity(), rs.getProbabilityDensityFunction());
			
				setIdentifierParams(rand, target);
				_ids.put(target.getName(), rand);
				Integer statementId = liveOutToTemp.get(target.getName());
				if ((statementId != null) && (statementId.intValue() == i)) {
					DataOp transientwrite = new DataOp(target.getName(), target.getDataType(), target.getValueType(), rand, DataOpTypes.TRANSIENTWRITE, null);
					transientwrite.setOutputParams(rand.get_dim1(), rand.get_dim2(), rand.get_rows_per_block(), rand.get_cols_per_block());
					updatedLiveOut.addVariable(target.getName(), target);
					output.add(transientwrite);
				}
			}
		}
		sb.updateLiveVariablesOut(updatedLiveOut);
		sb.set_hops(output);

	}
	
	public void constructHopsForIfControlBlock(IfStatementBlock sb) throws ParseException, LanguageException {
		
		IfStatement ifsb = (IfStatement) sb.getStatement(0);
		ArrayList<StatementBlock> ifBody = ifsb.getIfBody();
		ArrayList<StatementBlock> elseBody = ifsb.getElseBody();
	
		// construct hops for predicate in if statement
		constructHopsForConditionalPredicate(sb);
		
		// handle if statement body
		for (int i = 0; i < ifBody.size(); i++) {
			StatementBlock current = ifBody.get(i);
			constructHops(current);
		}
		
		// handle else stmt body
		for (int i = 0; i < elseBody.size(); i++) {
			StatementBlock current = elseBody.get(i);
			constructHops(current);
		}
	}
	
	
	public void constructHopsForForControlBlock(ForStatementBlock sb) throws ParseException, LanguageException {
		
		ForStatement whilesb = (ForStatement) sb.getStatement(0);
		ArrayList<StatementBlock> body = whilesb.getBody();
			
		// construct hops for iterable predicate
		//constructHopsForIterablePredicate(sb);
			
		for (int i = 0; i < body.size(); i++) {
			StatementBlock current = body.get(i);
			constructHops(current);
		}
	}
	
	public void constructHopsForFunctionControlBlock(FunctionStatementBlock fsb) throws ParseException, LanguageException {

		ArrayList<StatementBlock> body = ((FunctionStatement)fsb.getStatement(0)).getBody();

		for (int i = 0; i < body.size(); i++) {
			StatementBlock current = body.get(i);
			constructHops(current);
		}
	}
	
	public void constructHopsForWhileControlBlock(WhileStatementBlock sb) 
			throws ParseException, LanguageException {
		
		ArrayList<StatementBlock> body = ((WhileStatement)sb.getStatement(0)).getBody();
		
		// construct hops for while predicate
		constructHopsForConditionalPredicate(sb);
			
		for (int i = 0; i < body.size(); i++) {
			StatementBlock current = body.get(i);
			constructHops(current);
		}
	}
	
	
	public void constructHopsForConditionalPredicate(StatementBlock passedSB) throws ParseException {

		HashMap<String, Hops> _ids = new HashMap<String, Hops>();
		
		// set conditional predicate
		ConditionalPredicate cp = null;
		
		if (passedSB instanceof WhileStatementBlock){
			WhileStatement ws = (WhileStatement) ((WhileStatementBlock)passedSB).getStatement(0);
			cp = ws.getConditionalPredicate();
		} else if (passedSB instanceof IfStatementBlock){
			IfStatement ws = (IfStatement) ((IfStatementBlock)passedSB).getStatement(0);
			cp = ws.getConditionalPredicate();
		}
		
		VariableSet varsRead = cp.variablesRead();

		for (String varName : varsRead.getVariables().keySet()) {
			
			// creating transient read for live in variables
			DataIdentifier var = passedSB.liveIn().getVariables().get(varName);
			DataOp read = null;
			
			if (var == null) {
				throw new ParseException("variable " + varName + " not live variable for conditional predicate");
			} else {
				read = new DataOp(var.getName(), var.getDataType(), var.getValueType(), DataOpTypes.TRANSIENTREAD,
						null, var.getDim1(), var.getDim2(), (int) var.getRowsInBlock(), (int) var.getColumnsInBlock());
			}
			_ids.put(varName, read);
		}
		
		DataIdentifier target = new DataIdentifier(Expression.getTempName());
		target.setDataType(DataType.SCALAR);
		target.setValueType(ValueType.BOOLEAN);
		Hops predicateHops = null;
		Expression predicate = cp.getPredicate();
		
		if (predicate instanceof RelationalExpression) {
			predicateHops = processRelationalExpression((RelationalExpression) cp.getPredicate(), target, _ids);
		} else if (predicate instanceof BooleanExpression) {
			predicateHops = processBooleanExpression((BooleanExpression) cp.getPredicate(), target, _ids);
		} else if (predicate instanceof DataIdentifier) {
			// handle data identifier predicate
			predicateHops = processExpression(cp.getPredicate(), null, _ids);
		}
		if (passedSB instanceof WhileStatementBlock)
			((WhileStatementBlock)passedSB).set_predicate_hops(predicateHops);
		else if (passedSB instanceof IfStatementBlock)
			((IfStatementBlock)passedSB).set_predicate_hops(predicateHops);
	}

	/*
	public void constructHopsForIterablePredicate(StatementBlock passedSB) throws ParseException {

		
		HashMap<String, Hops> _ids = new HashMap<String, Hops>();
		
		
		// set iterable predicate
		ForStatement fs = (ForStatement) ((ForStatementBlock)passedSB).getStatement(0);
		IterablePredicate ip = fs.getIterablePredicate();
	
		VariableSet varsRead = ip.variablesRead();
		
		for (String varName : varsRead.getVariables().keySet()) {
		
			DataIdentifier var = passedSB.liveIn().getVariable(varName);
			DataOp read = null;
			if (var == null) {
				throw new ParseException("variable " + varName + " is not available for iterable predicate");
			} else {
				read = new DataOp(var.getName(), var.getDataType(), var.getValueType(), DataOpTypes.TRANSIENTREAD,
						null, var.getDim1(), var.getDim2(), (int) var.getRowsInBlock(), (int) var.getColumnsInBlock());
			}
			_ids.put(varName, read);
		}
		
		DataIdentifier target = new DataIdentifier(Expression.getTempName());
		target.setDataType(DataType.SCALAR);
		target.setValueType(ValueType.BOOLEAN);
		Hops predicateHops = null;
		Expression predicate = ip.getPredicate();
		
		if (predicate instanceof RelationalExpression) {
			predicateHops = processRelationalExpression((RelationalExpression) ip.getPredicate(), target, _ids);
		} else if (predicate instanceof BooleanExpression) {
			predicateHops = processBooleanExpression((BooleanExpression) ip.getPredicate(), target, _ids);
		} else if (predicate instanceof DataIdentifier) {
			predicateHops = processExpression(ip.getPredicate(), null, _ids);
		}
		((ForStatementBlock)passedSB).set_predicate_hops(predicateHops);
		
		
	}
	 */
	
	/**
	 * Construct Hops from parse tree : Process Expression in an assignment
	 * statement
	 * 
	 * @throws ParseException
	 */
	private Hops processExpression(Expression source, DataIdentifier target, HashMap<String, Hops> hops) throws ParseException {

		if (source.getKind() == Expression.Kind.BinaryOp) {
			return processBinaryExpression((BinaryExpression) source, target, hops);
		} else if (source.getKind() == Expression.Kind.RelationalOp) {
			return processRelationalExpression((RelationalExpression) source, target, hops);
		} else if (source.getKind() == Expression.Kind.BooleanOp) {
			return processBooleanExpression((BooleanExpression) source, target, hops);
		} else if (source.getKind() == Expression.Kind.Data) {
			if (source instanceof IntIdentifier) {
				IntIdentifier sourceInt = (IntIdentifier) source;
				LiteralOp litop = new LiteralOp(Long.toString(sourceInt.getValue()), sourceInt.getValue());
				setIdentifierParams(litop, sourceInt);
				return litop;
			} else if (source instanceof DoubleIdentifier) {
				DoubleIdentifier sourceDouble = (DoubleIdentifier) source;
				LiteralOp litop = new LiteralOp(Double.toString(sourceDouble.getValue()), sourceDouble.getValue());
				setIdentifierParams(litop, sourceDouble);
				return litop;
			} else if (source instanceof DataIdentifier) {
				DataIdentifier sourceId = (DataIdentifier) source;
				return hops.get(sourceId.getName());
			} else if (source instanceof BooleanIdentifier) {
				BooleanIdentifier sourceBoolean = (BooleanIdentifier) source;
				LiteralOp litop = new LiteralOp(Boolean.toString(sourceBoolean.getValue()), sourceBoolean.getValue());
				setIdentifierParams(litop, sourceBoolean);
				return litop;
			} else if (source instanceof StringIdentifier) {
				StringIdentifier sourceString = (StringIdentifier) source;
				LiteralOp litop = new LiteralOp(sourceString.getValue(), sourceString.getValue());
				setIdentifierParams(litop, sourceString);
				return litop;
			}
		} else if (source.getKind() == Expression.Kind.BuiltinFunctionOp) {
			try {
				return processBuiltinFunctionExpression((BuiltinFunctionExpression) source, target, hops);
			} catch (HopsException e) {
				e.printStackTrace();
			}
		} else if (source.getKind() == Expression.Kind.ParameterizedBuiltinFunctionOp ) {
			try {
				return processParameterizedBuiltinFunctionExpression((ParameterizedBuiltinFunctionExpression)source, target, hops);
			} catch ( HopsException e ) {
				e.printStackTrace();
			}
		}
		return null;
	} // end method processExpression

	private DataIdentifier createTarget(Expression source) {
		Identifier id = source.getOutput();
		if (id instanceof DataIdentifier)
			return (DataIdentifier) id;
		DataIdentifier target = new DataIdentifier(Expression.getTempName());
		target.setProperties(id);
		return target;
	}

	private DataIdentifier createTarget() {
		DataIdentifier target = new DataIdentifier(Expression.getTempName());
		return target;
	}

	/**
	 * Construct Hops from parse tree : Process Binary Expression in an
	 * assignment statement
	 * 
	 * @throws ParseException
	 */
	private Hops processBinaryExpression(BinaryExpression source, DataIdentifier target, HashMap<String, Hops> hops)
			throws ParseException {

		Hops left = processExpression(source.getLeft(), null, hops);
		Hops right = processExpression(source.getRight(), null, hops);

		if (left == null || right == null){
			System.out.println("broken");
			left = processExpression(source.getLeft(), null, hops);
			right = processExpression(source.getRight(), null, hops);
		}
	
		Hops currBop = null;

		if (target == null) {
			target = createTarget(source);
		}

		if (source.getOpCode() == Expression.BinaryOp.PLUS) {
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.PLUS, left, right);
		} else if (source.getOpCode() == Expression.BinaryOp.MINUS) {
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.MINUS, left, right);
		} else if (source.getOpCode() == Expression.BinaryOp.MULT) {
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.MULT, left, right);
		} else if (source.getOpCode() == Expression.BinaryOp.DIV) {
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.DIV, left, right);
		} else if (source.getOpCode() == Expression.BinaryOp.MATMULT) {
			currBop = new AggBinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.MULT, AggOp.SUM, left, right);
		} else if (source.getOpCode() == Expression.BinaryOp.POW) {
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.POW, left, right);
		}
		setIdentifierParams(currBop, source.getOutput());
		return currBop;
	}

	private Hops processRelationalExpression(RelationalExpression source, DataIdentifier target,
			HashMap<String, Hops> hops) throws ParseException {

		Hops left = processExpression(source.getLeft(), null, hops);
		Hops right = processExpression(source.getRight(), null, hops);

		Hops currBop = null;

		if (target == null) {
			target = createTarget(source);
			target.setValueType(ValueType.BOOLEAN);
		}
		OpOp2 op = null;

		if (source.getOpCode() == Expression.RelationalOp.LESS) {
			op = OpOp2.LESS;
		} else if (source.getOpCode() == Expression.RelationalOp.LESSEQUAL) {
			op = OpOp2.LESSEQUAL;
		} else if (source.getOpCode() == Expression.RelationalOp.GREATER) {
			op = OpOp2.GREATER;
		} else if (source.getOpCode() == Expression.RelationalOp.GREATEREQUAL) {
			op = OpOp2.GREATEREQUAL;
		} else if (source.getOpCode() == Expression.RelationalOp.EQUAL) {
			op = OpOp2.EQUAL;
		} else if (source.getOpCode() == Expression.RelationalOp.NOTEQUAL) {
			op = OpOp2.NOTEQUAL;
		}
		currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), op, left, right);
		return currBop;
	}

	private Hops processBooleanExpression(BooleanExpression source, DataIdentifier target, HashMap<String, Hops> hops)
			throws ParseException {

		// Boolean Not has a single parameter
		boolean constLeft = (source.getLeft().getOutput() instanceof ConstIdentifier);
		boolean constRight = false;
		if (source.getRight() != null) {
			constRight = (source.getRight().getOutput() instanceof ConstIdentifier);
		}

		if (constLeft || constRight) {
			throw new RuntimeException("Boolean expression with constant unsupported");
		}

		Hops left = processExpression(source.getLeft(), null, hops);
		Hops right = null;
		if (source.getRight() != null) {
			right = processExpression(source.getRight(), null, hops);
		}

		if (target == null) {
			target = createTarget(source);
			// ToDo : Remove this statement
			target.setValueType(ValueType.BOOLEAN);
		}

		if (source.getRight() == null) {
			Hops currUop = null;
			try {
				currUop = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), Hops.OpOp1.NOT, left);
			} catch (HopsException e) {
				e.printStackTrace();
			}
			return currUop;
		} else {
			Hops currBop = null;
			OpOp2 op = null;

			if (source.getOpCode() == Expression.BooleanOp.LOGICALAND) {
				op = OpOp2.AND;
			} else if (source.getOpCode() == Expression.BooleanOp.LOGICALOR) {
				op = OpOp2.OR;
			} else
				throw new RuntimeException("Unknown boolean operation " + source.getOpCode());
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), op, left, right);
			// setIdentifierParams(currBop,source.getOutput());
			return currBop;
		}
	}

	/**
	 * Construct Hops from parse tree : Process ParameterizedBuiltinFunction Expression in an
	 * assignment statement
	 * 
	 * @throws ParseException
	 * @throws HopsException 
	 */
	private Hops processParameterizedBuiltinFunctionExpression(ParameterizedBuiltinFunctionExpression source, DataIdentifier target,
			HashMap<String, Hops> hops) throws ParseException, HopsException {
		
		// this expression has multiple "named" parameters
		HashMap<String, Hops> paramHops = new HashMap<String,Hops>();
		
		// -- construct hops for all input parameters
		// -- store them in hashmap so that their "name"s are maintained
		Hops pHop = null;
		for ( String paramName : source.getVarParams().keySet() ) {
			pHop = processExpression(source.getVarParam(paramName), null, hops);
			paramHops.put(paramName, pHop);
		}
		
		Hops currBuiltinOp = null;

		if (target == null) {
			target = createTarget(source);
		}
		
		// construct hop based on opcode
		switch(source.getOpCode()) {
		case CDF:
			currBuiltinOp = new ParameterizedBuiltinOp(
					target.getName(), target.getDataType(), target.getValueType(), ParamBuiltinOp.CDF, paramHops);
			break;
			
		case GROUPEDAGG:
			currBuiltinOp = new ParameterizedBuiltinOp(
					target.getName(), target.getDataType(), target.getValueType(), ParamBuiltinOp.GROUPEDAGG, paramHops);
			break;
			
		default:
			throw new ParseException(
					"processParameterizedBuiltinFunctionExpression():: Unknown operation:  "
							+ source.getOpCode());
		}
		
		setIdentifierParams(currBuiltinOp, source.getOutput());
		
		return currBuiltinOp;
	}
	

	/**
	 * Construct Hops from parse tree : Process BuiltinFunction Expression in an
	 * assignment statement
	 * 
	 * @throws ParseException
	 * @throws HopsException 
	 */
	private Hops processBuiltinFunctionExpression(BuiltinFunctionExpression source, DataIdentifier target,
			HashMap<String, Hops> hops) throws ParseException, HopsException {
		Hops expr = processExpression(source.getFirstExpr(), null, hops);
		Hops expr2 = null;
		if (source.getSecondExpr() != null) {
			expr2 = processExpression(source.getSecondExpr(), null, hops);
		}
		Hops expr3 = null;
		if (source.getThirdExpr() != null) {
			expr3 = processExpression(source.getThirdExpr(), null, hops);
		}
		
		Hops currBuiltinOp = null;

		if (target == null) {
			target = createTarget(source);
		}

		// Construct the hop based on the type of Builtin function
		switch (source.getOpCode()) {

		case COLSUM:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.SUM,
					Direction.Col, expr);
			break;

		case COLMAX:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MAX,
					Direction.Col, expr);
			break;

		case COLMIN:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MIN,
					Direction.Col, expr);
			break;

		case COLMEAN:
			// hop to compute colSums
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MEAN,
					Direction.Col, expr);
			break;

		case ROWSUM:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.SUM,
					Direction.Row, expr);
			break;

		case ROWMAX:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MAX,
					Direction.Row, expr);
			break;

		case ROWMIN:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MIN,
					Direction.Row, expr);
			break;

		case ROWMEAN:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MEAN,
					Direction.Row, expr);
			break;

		case NROW:
			/* 
			 * If the dimensions are available at compile time, then create a LiteralOp (constant propagation)
			 * Else create a UnaryOp so that a control program instruction is generated
			 */
			long nRows = expr.get_dim1();
			if (nRows == -1) {
				currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), Hops.OpOp1.NROW, expr);
			}
			else {
				int numRowsIntValue = (int) nRows;
				currBuiltinOp = new LiteralOp(Integer.toString(numRowsIntValue), numRowsIntValue);
			}
			break;

		case NCOL:
			/* 
			 * If the dimensions are available at compile time, then create a LiteralOp (constant propagation)
			 * Else create a UnaryOp so that a control program instruction is generated
			 */
			long nCols = expr.get_dim2();
			if (nCols == -1) {
				currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), Hops.OpOp1.NCOL, expr);
			}
			else {
				int numColsIntValue = (int) nCols;
				currBuiltinOp = new LiteralOp(Integer.toString(numColsIntValue), numColsIntValue);
			}
			break;
		case LENGTH:
			long nRows2 = expr.get_dim1();
			long nCols2 = expr.get_dim2();
			/* 
			 * If the dimensions are available at compile time, then create a LiteralOp (constant propagation)
			 * Else create a UnaryOp so that a control program instruction is generated
			 */
			if ((nCols2 == -1) || (nRows2 == -1)) {
				currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), Hops.OpOp1.LENGTH, expr);
			}
			else {
				int len = (int) (nCols2 * nRows2);
				currBuiltinOp = new LiteralOp(Integer.toString(len), len);
			}
			break;

		case SUM:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.SUM,
					Direction.RowCol, expr);
			break;
			
		case MEAN:
			if ( expr2 == null ) {
				// example: x = mean(Y);
				currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MEAN,
					Direction.RowCol, expr);
			}
			else if ( expr2 != null ) {
				// example: x = mean(Y,W);
				
				// stable weighted mean is implemented by using centralMoment with order = 0
				Hops orderHop = new LiteralOp(Integer.toString(0), 0);
				currBuiltinOp=new TertiaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hops.OpOp3.CENTRALMOMENT, expr, expr2, orderHop);
			}
			break;
			
		case MIN:
			if (expr.get_dataType() == DataType.MATRIX) {
				currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(),
						AggOp.MIN, Direction.RowCol, expr);
			} else {
				currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.MIN,
						expr, expr2);
			}
			break;
		case MAX:
			if (expr.get_dataType() == DataType.MATRIX) {
				currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(),
						AggOp.MAX, Direction.RowCol, expr);
			} else {
				currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.MAX,
						expr, expr2);
			}
			break;
		case PMIN:
			currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.MIN,
					expr, expr2);
			break;
		case PMAX:
			currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.MAX,
					expr, expr2);
			break;
			
		case PPRED:
			String sop = ((StringIdentifier)source.getThirdExpr()).getValue();
			sop = sop.replace("\"", "");
			OpOp2 operation;
			if ( sop.equalsIgnoreCase(">=") ) 
				operation = OpOp2.GREATEREQUAL;
			else if ( sop.equalsIgnoreCase(">") )
				operation = OpOp2.GREATER;
			else if ( sop.equalsIgnoreCase("<=") )
				operation = OpOp2.LESSEQUAL;
			else if ( sop.equalsIgnoreCase("<") )
				operation = OpOp2.LESS;
			else if ( sop.equalsIgnoreCase("==") )
				operation = OpOp2.EQUAL;
			else if ( sop.equalsIgnoreCase("!=") )
				operation = OpOp2.NOTEQUAL;
			else
				throw new ParseException("Unknown argument (" + sop + ") for PPRED.");
			
			currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), operation, expr, expr2);
			break;
			
		case PROD:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.PROD,
					Direction.RowCol, expr);
			break;
		case TRACE:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.TRACE,
					Direction.RowCol, expr);
			break;

		case TRANS:
			currBuiltinOp = new ReorgOp(target.getName(), target.getDataType(), target.getValueType(),
					Hops.ReorgOp.TRANSPOSE, expr);
			// currBop = new AggUnaryOp(targetName,AggOp.SUM,TransfOp.ColKey,
			// expr);
			break;

		case DIAG:
			// If either of the input is a vector, then output is a matrix
			if (expr.get_dim1() == 1  || expr.get_dim2() == 1) {
				currBuiltinOp = new ReorgOp(target.getName(), target.getDataType(), target.getValueType(),
						Hops.ReorgOp.DIAG_V2M, expr);
			} else {
				currBuiltinOp = new ReorgOp(target.getName(), target.getDataType(), target.getValueType(),
						Hops.ReorgOp.DIAG_M2V, expr);
			}
			break;
			
		case CTABLE:
			if ( expr3 != null ) {
				// example DML statement: F = ctable(A,B,W) 
				currBuiltinOp = new TertiaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp3.CTABLE, expr, expr2, expr3);
			}
			else {				
				// example DML statement: F = ctable(A,B)
				// here, weight is interpreted as 1.0
				Hops weightHop = new LiteralOp(Double.toString(1.0), 1.0);
				currBuiltinOp = new TertiaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp3.CTABLE, expr, expr2, weightHop);
				/*
				RandOp rand = new RandOp(target, 1.0, 1.0, 1.0, "uniform");
				setIdentifierParams(rand, expr); // Rand lop should have same dimensions as the input hop
				currBuiltinOp = new TertiaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp3.CTABLE, expr, expr2, rand);
				*/
				
			} 
			break;

		case SPEARMAN:
			if ( expr3 != null ) {
				currBuiltinOp = new TertiaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp3.SPEARMAN, expr, expr2, expr3);
			}
			else {				
				// example DML statement: s = spearman(A,B)
				// here, weight is interpreted as 1.0
				Hops weightHop = new LiteralOp(Double.toString(1.0), 1.0);
				currBuiltinOp = new TertiaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp3.SPEARMAN, expr, expr2, weightHop);
				/*
				RandOp rand = new RandOp(target, 1.0, 1.0, 1.0, "uniform");
				setIdentifierParams(rand, expr); // Rand lop should have same dimensions as the input hop
				currBuiltinOp = new TertiaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp3.SPEARMAN, expr, expr2, rand);
				*/
			}
			break;

		case ROUND:
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp1.ROUND, expr);
			break;
			
		case CAST_AS_SCALAR:
			// TODO: fix the hops/lops first.
			try {
				currBuiltinOp = new UnaryOp(target.getName(), DataType.SCALAR, target.getValueType(), Hops.OpOp1.CAST_AS_SCALAR, expr);
			} catch (HopsException e) {
				e.printStackTrace();
			}
			break;

		case ABS:
		case SIN:
		case COS:
		case TAN:
		case SQRT:
		case EXP:
			Hops.OpOp1 mathOp1;
			switch (source.getOpCode()) {
			case ABS:
				mathOp1 = Hops.OpOp1.ABS;
				break;
			case SIN:
				mathOp1 = Hops.OpOp1.SIN;
				break;
			case COS:
				mathOp1 = Hops.OpOp1.COS;
				break;
			case TAN:
				mathOp1 = Hops.OpOp1.TAN;
				break;
			case SQRT:
				mathOp1 = Hops.OpOp1.SQRT;
				break;
			case EXP:
				mathOp1 = Hops.OpOp1.EXP;
				break;
			default:
				throw new ParseException(
						"processBuiltinFunctionExpression():: Could not find Operation type for builtin function: "
								+ source.getOpCode());
			}
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), mathOp1, expr);
			break;
		case LOG:
				if (expr2 == null) {
					Hops.OpOp1 mathOp2;
					switch (source.getOpCode()) {
					case LOG:
						mathOp2 = Hops.OpOp1.LOG;
						break;
					default:
						throw new ParseException(
								"processBuiltinFunctionExpression():: Could not find Operation type for builtin function: "
										+ source.getOpCode());
					}
					currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), mathOp2,
							expr);
				} else {
					Hops.OpOp2 mathOp3;
					switch (source.getOpCode()) {
					case LOG:
						mathOp3 = Hops.OpOp2.LOG;
						break;
					default:
						throw new ParseException(
								"processBuiltinFunctionExpression():: Could not find Operation type for builtin function: "
										+ source.getOpCode());
					}
					currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), mathOp3,
							expr, expr2);
				}
			break;
		case CENTRALMOMENT:
			if (expr3 == null){
				currBuiltinOp=new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hops.OpOp2.CENTRALMOMENT, expr, expr2);
			}
			else {
				currBuiltinOp=new TertiaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hops.OpOp3.CENTRALMOMENT, expr, expr2,expr3);
			}
			break;
			
		case COVARIANCE:
			if (expr3 == null){
				currBuiltinOp=new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hops.OpOp2.COVARIANCE, expr, expr2);
			}
			else {
				currBuiltinOp=new TertiaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hops.OpOp3.COVARIANCE, expr, expr2,expr3);
			}
			break;
			
		case QUANTILE:
			if (expr3 == null){
				currBuiltinOp=new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hops.OpOp2.QUANTILE, expr, expr2);
			}
			else {
				currBuiltinOp=new TertiaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hops.OpOp3.QUANTILE, expr, expr2,expr3);
			}
			break;
			
		case INTERQUANTILE:
			if ( expr3 == null ) {
				currBuiltinOp=new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hops.OpOp2.INTERQUANTILE, expr, expr2);
			}
			else {
				currBuiltinOp=new TertiaryOp(target.getName(), target.getDataType(), target.getValueType(), 
					Hops.OpOp3.INTERQUANTILE, expr, expr2,expr3);
			}
			break;	
			
		case IQM:
			if ( expr2 == null ) {
				currBuiltinOp=new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hops.OpOp1.IQM, expr);
			}
			else {
				currBuiltinOp=new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), 
					Hops.OpOp2.IQM, expr, expr2);
			}
			break;	
			
		default:
			break;
		}
		setIdentifierParams(currBuiltinOp, source.getOutput());
		return currBuiltinOp;
	}
		
	public void setIdentifierParams(Hops h, Identifier id) {
		// ToDo: Fix this type cast from long to int
		h.set_dim1(id.getDim1());
		h.set_dim2(id.getDim2());
		h.set_rows_per_block((int) id.getRowsInBlock());
		h.set_cols_per_block((int) id.getColumnsInBlock());
	}

	public void setIdentifierParams(Hops h, Hops source) {
		// ToDo: Fix this type cast from long to int
		h.set_dim1(source.get_dim1());
		h.set_dim2(source.get_dim2());
		h.set_rows_per_block(source.get_rows_per_block());
		h.set_cols_per_block(source.get_cols_per_block());
	}

}
