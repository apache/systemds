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

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysml.conf.CompilerConfig.ConfigType;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.ForStatement;
import org.apache.sysml.parser.ForStatementBlock;
import org.apache.sysml.parser.FunctionStatement;
import org.apache.sysml.parser.FunctionStatementBlock;
import org.apache.sysml.parser.IfStatement;
import org.apache.sysml.parser.IfStatementBlock;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.ParForStatementBlock;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.WhileStatement;
import org.apache.sysml.parser.WhileStatementBlock;

/**
 * This program rewriter applies a variety of rule-based rewrites
 * on all hop dags of the given program in one pass over the entire
 * program. 
 * 
 */
public class ProgramRewriter 
{
	private static final Log LOG = LogFactory.getLog(ProgramRewriter.class.getName());
	
	//internal local debug level
	private static final boolean LDEBUG = false; 
	private static final boolean CHECK = false;
	
	private ArrayList<HopRewriteRule> _dagRuleSet = null;
	private ArrayList<StatementBlockRewriteRule> _sbRuleSet = null;
	
	static{
		// for internal debugging only
		if( LDEBUG ) {
			Logger.getLogger("org.apache.sysml.hops.rewrite")
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
			_dagRuleSet.add(     new RewriteCompressedReblock()                  );
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
			_dagRuleSet.add( new RewriteInjectSparkPReadCheckpointing()          ); //dependency: reblock
			
			//add statment block rewrite rules
 			if( OptimizerUtils.ALLOW_BRANCH_REMOVAL )			
				_sbRuleSet.add(  new RewriteRemoveUnnecessaryBranches()          ); //dependency: constant folding		
 			if( OptimizerUtils.ALLOW_SPLIT_HOP_DAGS )
 				_sbRuleSet.add(  new RewriteSplitDagUnknownCSVRead()             ); //dependency: reblock	
 			if( ConfigurationManager.getCompilerConfigFlag(ConfigType.ALLOW_INDIVIDUAL_SB_SPECIFIC_OPS) )
 				_sbRuleSet.add(  new RewriteSplitDagDataDependentOperators()     );
 			if( OptimizerUtils.ALLOW_AUTO_VECTORIZATION )
				_sbRuleSet.add(  new RewriteForLoopVectorization()               ); //dependency: reblock (reblockop)
 			_sbRuleSet.add( new RewriteInjectSparkLoopCheckpointing(true)        ); //dependency: reblock (blocksizes)
 			if( OptimizerUtils.ALLOW_LOOP_UPDATE_IN_PLACE )
 				_sbRuleSet.add(  new RewriteMarkLoopVariablesUpdateInPlace()      );
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
		}
		
		// cleanup after all rewrites applied 
		// (newly introduced operators, introduced redundancy after rewrites w/ multiple parents) 
		_dagRuleSet.add(     new RewriteRemoveUnnecessaryCasts()             );		
		if( OptimizerUtils.ALLOW_COMMON_SUBEXPRESSION_ELIMINATION )             
			_dagRuleSet.add( new RewriteCommonSubexpressionElimination(true) ); 			
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
		
			if( CHECK ) {		
				LOG.info("Validation after: "+r.getClass().getName());
				HopDagValidator.validateHopDag(roots);
			}
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
		if( root == null )
			return root;
		
		for( HopRewriteRule r : _dagRuleSet )
		{
			root.resetVisitStatus(); //reset for each rule
			root = r.rewriteHopDAG(root, state);
		
			if( CHECK ) {
				LOG.info("Validation after: "+r.getClass().getName());
				HopDagValidator.validateHopDag(root);
			}
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
	private ArrayList<StatementBlock> rewriteStatementBlock( StatementBlock sb, ProgramRewriteStatus status ) 
		throws HopsException
	{
		ArrayList<StatementBlock> ret = new ArrayList<StatementBlock>();
		ret.add(sb);
		
		//recursive invocation
		if (sb instanceof FunctionStatementBlock)
		{
			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			fstmt.setBody( rewriteStatementBlocks(fstmt.getBody(), status) );			
		}
		else if (sb instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			wstmt.setBody( rewriteStatementBlocks( wstmt.getBody(), status ) );
		}	
		else if (sb instanceof IfStatementBlock)
		{
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			istmt.setIfBody( rewriteStatementBlocks( istmt.getIfBody(), status ) );
			istmt.setElseBody( rewriteStatementBlocks( istmt.getElseBody(), status ) );
		}
		else if (sb instanceof ForStatementBlock) //incl parfor
		{
			//maintain parfor context information (e.g., for checkpointing)
			boolean prestatus = status.isInParforContext();
			if( sb instanceof ParForStatementBlock )
				status.setInParforContext(true);
			
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			fstmt.setBody( rewriteStatementBlocks(fstmt.getBody(), status) );
			
			status.setInParforContext(prestatus);
		}
		
		//apply rewrite rules
		for( StatementBlockRewriteRule r : _sbRuleSet )
		{
			ArrayList<StatementBlock> tmp = new ArrayList<StatementBlock>();			
			for( StatementBlock sbc : ret )
				tmp.addAll( r.rewriteStatementBlock(sbc, status) );
			
			//take over set of rewritten sbs		
			ret.clear();
			ret.addAll(tmp);
		}
		
		return ret;
	}
}
