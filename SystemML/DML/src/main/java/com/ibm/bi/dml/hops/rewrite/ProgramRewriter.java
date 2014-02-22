/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

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
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private ArrayList<HopRewriteRule> _ruleSet = null;
	
	public ProgramRewriter()
	{
		//initialize ruleSet (with fixed rewrite order)
		_ruleSet = new ArrayList<HopRewriteRule>();
			
		_ruleSet.add(     new RewriteTransientWriteParentHandling()   );
		_ruleSet.add(     new RewriteBlockSizeAndReblock()            );
		if( OptimizerUtils.ALLOW_COMMON_SUBEXPRESSION_ELIMINATION )
			_ruleSet.add( new RewriteCommonSubexpressionElimination() );
		if( OptimizerUtils.ALLOW_CONSTANT_FOLDING )
			_ruleSet.add( new RewriteConstantFolding()                ); //dependencies: common subexpression elimination
		_ruleSet.add(     new RewriteMatrixMultChainOptimization()    );
		_ruleSet.add(     new RewriteAlgebraicSimplification()        ); //dependencies: common subexpression elimination
	}
	
	/**
	 * 
	 * @param dmlp
	 * @throws LanguageException
	 * @throws HopsException
	 */
	public void rewriteProgramHopDAGs(DMLProgram dmlp) 
		throws LanguageException, HopsException
	{	
		// for each namespace, handle function statement blocks
		for (String namespaceKey : dmlp.getNamespaces().keySet())
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet())
			{
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey,fname);
				rewriteStatementBlockHopDAGs(fsblock);
			}
		
		// handle regular statement blocks in "main" method
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) 
		{
			StatementBlock current = dmlp.getStatementBlock(i);
			rewriteStatementBlockHopDAGs(current);
		}
	}
	
	/**
	 * 
	 * @param current
	 * @throws LanguageException
	 * @throws HopsException
	 */
	private void rewriteStatementBlockHopDAGs(StatementBlock current) 
		throws LanguageException, HopsException
	{	
		if (current instanceof FunctionStatementBlock)
		{
			FunctionStatementBlock fsb = (FunctionStatementBlock)current;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for (StatementBlock sb : fstmt.getBody())
				rewriteStatementBlockHopDAGs(sb);
		}
		else if (current instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock) current;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			wsb.setPredicateHops(rewriteHopDAG(wsb.getPredicateHops()));
			for (StatementBlock sb : wstmt.getBody())
				rewriteStatementBlockHopDAGs(sb);
		}	
		else if (current instanceof IfStatementBlock)
		{
			IfStatementBlock isb = (IfStatementBlock) current;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			isb.setPredicateHops(rewriteHopDAG(isb.getPredicateHops()));
			for (StatementBlock sb : istmt.getIfBody())
				rewriteStatementBlockHopDAGs(sb);
			for (StatementBlock sb : istmt.getElseBody())
				rewriteStatementBlockHopDAGs(sb);
		}
		else if (current instanceof ForStatementBlock) //incl parfor
		{
			ForStatementBlock fsb = (ForStatementBlock) current;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			fsb.setFromHops(rewriteHopDAG(fsb.getFromHops()));
			fsb.setToHops(rewriteHopDAG(fsb.getToHops()));
			fsb.setIncrementHops(rewriteHopDAG(fsb.getIncrementHops()));
			for (StatementBlock sb : fstmt.getBody())
				rewriteStatementBlockHopDAGs(sb);
		}
		else //generic (last-level)
		{
			current.set_hops(rewriteHopDAGs(current.get_hops()));
		}
	}
	
	/**
	 * 
	 * @param roots
	 * @throws LanguageException
	 * @throws HopsException
	 */
	private ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots) 
		throws LanguageException, HopsException
	{	
		for( HopRewriteRule r : _ruleSet )
		{
			Hop.resetVisitStatus( roots ); //reset for each rule
			roots = r.rewriteHopDAGs(roots);
		}
		
		return roots;
	}
	
	/**
	 * 
	 * @param root
	 * @throws LanguageException
	 * @throws HopsException
	 */
	private Hop rewriteHopDAG(Hop root) 
		throws LanguageException, HopsException
	{	
		for( HopRewriteRule r : _ruleSet )
		{
			root.resetVisitStatus(); //reset for each rule
			root = r.rewriteHopDAG(root);
		}
		
		return root;
	}
}
