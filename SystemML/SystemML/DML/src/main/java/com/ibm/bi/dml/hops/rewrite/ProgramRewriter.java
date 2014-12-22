/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.ForStatement;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.FunctionStatement;
import com.ibm.bi.dml.parser.FunctionStatementBlock;
import com.ibm.bi.dml.parser.IfStatement;
import com.ibm.bi.dml.parser.IfStatementBlock;
import com.ibm.bi.dml.parser.LanguageException;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.WhileStatement;
import com.ibm.bi.dml.parser.WhileStatementBlock;

/**
 * This program rewriter applies a variety of rule-based rewrites
 * on all hop dags of the given program in one pass over the entire
 * program. 
 * 
 */
public class ProgramRewriter 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final boolean LDEBUG = false; //internal local debug level
	
	private ArrayList<HopRewriteRule> _dagRuleSet = null;
	private ArrayList<StatementBlockRewriteRule> _sbRuleSet = null;
	
	static{
		// for internal debugging only
		if( LDEBUG ) {
			Logger.getLogger("com.ibm.bi.dml.hops.rewrite")
				  .setLevel((Level) Level.DEBUG);
		}
		
	}
	
	public ProgramRewriter()
	{
		// by default which is used during initial compile 
		// apply all (static and dynamic) rewrites
		this( true, true );
	}
	
	public ProgramRewriter( boolean staticRewrites, boolean dynamicRewrites )
	{
		//initialize HOP DAG rewrite ruleSet (with fixed rewrite order)
		_dagRuleSet = new ArrayList<HopRewriteRule>();
		
		//initialize StatementBlock rewrite ruleSet (with fixed rewrite order)
		_sbRuleSet = new ArrayList<StatementBlockRewriteRule>();
		
		
		//STATIC REWRITES (which do not rely on size information)
		if( staticRewrites )
		{
			//add static HOP DAG rewrite rules
			_dagRuleSet.add(     new RewriteTransientWriteParentHandling()       );
			_dagRuleSet.add(     new RewriteRemoveReadAfterWrite()               ); //dependency: before blocksize
			_dagRuleSet.add(     new RewriteBlockSizeAndReblock()                );
			_dagRuleSet.add(     new RewriteRemoveUnnecessaryCasts()             );		
			if( OptimizerUtils.ALLOW_COMMON_SUBEXPRESSION_ELIMINATION )
				_dagRuleSet.add( new RewriteCommonSubexpressionElimination()     );
			if( OptimizerUtils.ALLOW_CONSTANT_FOLDING )
				_dagRuleSet.add( new RewriteConstantFolding()                    ); //dependency: cse
			if( OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION )
				_dagRuleSet.add( new RewriteAlgebraicSimplificationStatic()      ); //dependencies: cse
			if( OptimizerUtils.ALLOW_COMMON_SUBEXPRESSION_ELIMINATION )             //dependency: simplifications (no need to merge leafs again)
				_dagRuleSet.add( new RewriteCommonSubexpressionElimination()     ); 
			if( OptimizerUtils.ALLOW_AUTO_VECTORIZATION )
				_dagRuleSet.add( new RewriteIndexingVectorization()              ); //dependency: cse, simplifications
			
			//add statment block rewrite rules
 			if( OptimizerUtils.ALLOW_BRANCH_REMOVAL )			
				_sbRuleSet.add(  new RewriteRemoveUnnecessaryBranches()          ); //dependency: constant folding		
 			if( OptimizerUtils.ALLOW_SPLIT_HOP_DAGS )
 				_sbRuleSet.add(  new RewriteSplitDagUnknownCSVRead()             ); //dependency: reblock	
 			if( OptimizerUtils.ALLOW_AUTO_VECTORIZATION )
				_sbRuleSet.add(  new RewriteForLoopVectorization()               );
		}
		
		// DYNAMIC REWRITES (which do require size information)
		if( dynamicRewrites )
		{
			_dagRuleSet.add(     new RewriteMatrixMultChainOptimization()         ); //dependency: cse 
			
			if( OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION )
			{
				_dagRuleSet.add( new RewriteAlgebraicSimplificationDynamic()      ); //dependencies: cse
				_dagRuleSet.add( new RewriteAlgebraicSimplificationStatic()       ); //dependencies: cse
			}
			
			if( OptimizerUtils.ALLOW_COMMON_SUBEXPRESSION_ELIMINATION )             
				_dagRuleSet.add( new RewriteCommonSubexpressionElimination(false) ); //dependency: simplifications (no need to merge leafs again) 
			
		}
	}
	
	/**
	 * Construct a program rewriter for a given rewrite which is passed from outside.
	 * 
	 * @param rewrite
	 */
	public ProgramRewriter( HopRewriteRule rewrite )
	{
		//initialize HOP DAG rewrite ruleSet (with fixed rewrite order)
		_dagRuleSet = new ArrayList<HopRewriteRule>();
		_dagRuleSet.add( rewrite );		
		
		_sbRuleSet = new ArrayList<StatementBlockRewriteRule>();
	}
	
	/**
	 * Construct a program rewriter for a given rewrite which is passed from outside.
	 * 
	 * @param rewrite
	 */
	public ProgramRewriter( StatementBlockRewriteRule rewrite )
	{
		//initialize HOP DAG rewrite ruleSet (with fixed rewrite order)
		_dagRuleSet = new ArrayList<HopRewriteRule>();
		
		_sbRuleSet = new ArrayList<StatementBlockRewriteRule>();
		_sbRuleSet.add( rewrite );
	}
	
	/**
	 * Construct a program rewriter for the given rewrite sets which are passed from outside.
	 * 
	 * @param rewrite
	 */
	public ProgramRewriter( ArrayList<HopRewriteRule> hRewrites, ArrayList<StatementBlockRewriteRule> sbRewrites )
	{
		//initialize HOP DAG rewrite ruleSet (with fixed rewrite order)
		_dagRuleSet = new ArrayList<HopRewriteRule>();
		_dagRuleSet.addAll( hRewrites );
		
		_sbRuleSet = new ArrayList<StatementBlockRewriteRule>();
		_sbRuleSet.addAll( sbRewrites );
	}
	
	/**
	 * 
	 * @param dmlp
	 * @return
	 * @throws LanguageException
	 * @throws HopsException
	 */
	public ProgramRewriteStatus rewriteProgramHopDAGs(DMLProgram dmlp) 
		throws LanguageException, HopsException
	{	
		ProgramRewriteStatus state = new ProgramRewriteStatus();
		
		// for each namespace, handle function statement blocks
		for (String namespaceKey : dmlp.getNamespaces().keySet())
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet())
			{
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey,fname);
				rewriteStatementBlockHopDAGs(fsblock, state);
				rewriteStatementBlock(fsblock, state);
			}
		
		// handle regular statement blocks in "main" method
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) 
		{
			StatementBlock current = dmlp.getStatementBlock(i);
			rewriteStatementBlockHopDAGs(current, state);
		}
		dmlp.setStatementBlocks( rewriteStatementBlocks(dmlp.getStatementBlocks(), state) );
		
		return state;
	}
	
	/**
	 * 
	 * @param current
	 * @throws LanguageException
	 * @throws HopsException
	 */
	public void rewriteStatementBlockHopDAGs(StatementBlock current, ProgramRewriteStatus state) 
		throws LanguageException, HopsException
	{	
		//ensure robustness for calls from outside
		if( state == null )
			state = new ProgramRewriteStatus();
		
		if (current instanceof FunctionStatementBlock)
		{
			FunctionStatementBlock fsb = (FunctionStatementBlock)current;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for (StatementBlock sb : fstmt.getBody())
				rewriteStatementBlockHopDAGs(sb, state);
		}
		else if (current instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock) current;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			wsb.setPredicateHops(rewriteHopDAG(wsb.getPredicateHops(), state));
			for (StatementBlock sb : wstmt.getBody())
				rewriteStatementBlockHopDAGs(sb, state);
		}	
		else if (current instanceof IfStatementBlock)
		{
			IfStatementBlock isb = (IfStatementBlock) current;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			isb.setPredicateHops(rewriteHopDAG(isb.getPredicateHops(), state));
			for (StatementBlock sb : istmt.getIfBody())
				rewriteStatementBlockHopDAGs(sb, state);
			for (StatementBlock sb : istmt.getElseBody())
				rewriteStatementBlockHopDAGs(sb, state);
		}
		else if (current instanceof ForStatementBlock) //incl parfor
		{
			ForStatementBlock fsb = (ForStatementBlock) current;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			fsb.setFromHops(rewriteHopDAG(fsb.getFromHops(), state));
			fsb.setToHops(rewriteHopDAG(fsb.getToHops(), state));
			fsb.setIncrementHops(rewriteHopDAG(fsb.getIncrementHops(), state));
			for (StatementBlock sb : fstmt.getBody())
				rewriteStatementBlockHopDAGs(sb, state);
		}
		else //generic (last-level)
		{
			current.set_hops( rewriteHopDAGs(current.get_hops(), state) );
		}
	}
	
	/**
	 * 
	 * @param roots
	 * @throws LanguageException
	 * @throws HopsException
	 */
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) 
		throws HopsException
	{	
		for( HopRewriteRule r : _dagRuleSet )
		{
			Hop.resetVisitStatus( roots ); //reset for each rule
			roots = r.rewriteHopDAGs(roots, state);
		}
		
		return roots;
	}
	
	/**
	 * 
	 * @param root
	 * @throws LanguageException
	 * @throws HopsException
	 */
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) 
		throws HopsException
	{	
		for( HopRewriteRule r : _dagRuleSet )
		{
			root.resetVisitStatus(); //reset for each rule
			root = r.rewriteHopDAG(root, state);
		}
		
		return root;
	}
	
	/**
	 * 
	 * @param sbs
	 * @return
	 * @throws HopsException 
	 */
	public ArrayList<StatementBlock> rewriteStatementBlocks( ArrayList<StatementBlock> sbs, ProgramRewriteStatus state ) 
		throws HopsException
	{
		//ensure robustness for calls from outside
		if( state == null )
			state = new ProgramRewriteStatus();
				
		
		ArrayList<StatementBlock> tmp = new ArrayList<StatementBlock>();
		
		//rewrite statement blocks (with potential expansion)
		for( StatementBlock sb : sbs )
			tmp.addAll( rewriteStatementBlock(sb, state) );
		
		//copy results into original collection
		sbs.clear();
		sbs.addAll( tmp );
		
		return sbs;
	}
	
	/**
	 * 
	 * @param sb
	 * @return
	 * @throws HopsException
	 */
	private ArrayList<StatementBlock> rewriteStatementBlock( StatementBlock sb, ProgramRewriteStatus state ) 
		throws HopsException
	{
		ArrayList<StatementBlock> ret = new ArrayList<StatementBlock>();
		ret.add(sb);
		
		//recursive invocation
		if (sb instanceof FunctionStatementBlock)
		{
			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			fstmt.setBody( rewriteStatementBlocks(fstmt.getBody(), state) );			
		}
		else if (sb instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			wstmt.setBody( rewriteStatementBlocks( wstmt.getBody(), state ) );
		}	
		else if (sb instanceof IfStatementBlock)
		{
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			istmt.setIfBody( rewriteStatementBlocks( istmt.getIfBody(), state ) );
			istmt.setElseBody( rewriteStatementBlocks( istmt.getElseBody(), state ) );
		}
		else if (sb instanceof ForStatementBlock) //incl parfor
		{
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			fstmt.setBody( rewriteStatementBlocks(fstmt.getBody(), state) );
		}
		
		//apply rewrite rules
		for( StatementBlockRewriteRule r : _sbRuleSet )
		{
			ArrayList<StatementBlock> tmp = new ArrayList<StatementBlock>();			
			for( StatementBlock sbc : ret )
				tmp.addAll( r.rewriteStatementBlock(sbc, state) );
			
			//take over set of rewritten sbs		
			ret.clear();
			ret.addAll(tmp);
		}
		
		return ret;
	}
}
