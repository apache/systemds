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

package org.apache.sysds.hops.rewrite;

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.rewriter.generated.GeneratedRewriteClass;
import org.apache.sysds.hops.rewriter.generated.RewriteAutomaticallyGenerated;
import org.apache.sysds.hops.rewriter.dml.DMLExecutor;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.ParForStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;

/**
 * This program rewriter applies a variety of rule-based rewrites
 * on all hop dags of the given program in one pass over the entire
 * program. 
 * 
 */
public class ProgramRewriter{
	private static final boolean CHECK = false;
	
	static {
		//Logger.getLogger("org.apache.sysds.hops.rewrite").setLevel(Level.DEBUG);
	}
	
	private ArrayList<HopRewriteRule> _dagRuleSet = null;
	private ArrayList<StatementBlockRewriteRule> _sbRuleSet = null;

	public ProgramRewriter() {
		// by default which is used during initial compile 
		// apply all (static and dynamic) rewrites
		this( true, true );
	}
	
	public ProgramRewriter(boolean staticRewrites, boolean dynamicRewrites)
	{
		//initialize HOP DAG rewrite ruleSet (with fixed rewrite order)
		_dagRuleSet = new ArrayList<>();
		
		//initialize StatementBlock rewrite ruleSet (with fixed rewrite order)
		_sbRuleSet = new ArrayList<>();
		
		
		//STATIC REWRITES (which do not rely on size information)
		if( staticRewrites )
		{
			//add static HOP DAG rewrite rules
			_dagRuleSet.add(     new RewriteRemoveReadAfterWrite()               ); //dependency: before blocksize
			_dagRuleSet.add(     new RewriteBlockSizeAndReblock()                );
			if( OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION )
				_dagRuleSet.add( new RewriteRemoveUnnecessaryCasts()             );
			if( OptimizerUtils.ALLOW_COMMON_SUBEXPRESSION_ELIMINATION )
				_dagRuleSet.add( new RewriteCommonSubexpressionElimination()     );
			if( OptimizerUtils.ALLOW_CONSTANT_FOLDING )
				_dagRuleSet.add( new RewriteConstantFolding()                    ); //dependency: cse
			if ( DMLScript.APPLY_GENERATED_REWRITES ) {
				_dagRuleSet.add(new RewriteAutomaticallyGenerated(new GeneratedRewriteClass()));
			}
			if( OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION )
				_dagRuleSet.add( new RewriteAlgebraicSimplificationStatic()      ); //dependencies: cse
			if( OptimizerUtils.ALLOW_COMMON_SUBEXPRESSION_ELIMINATION )             //dependency: simplifications (no need to merge leafs again)
				_dagRuleSet.add( new RewriteCommonSubexpressionElimination()     );
			if( OptimizerUtils.ALLOW_AUTO_VECTORIZATION )
				_dagRuleSet.add( new RewriteIndexingVectorization()              ); //dependency: cse, simplifications
			_dagRuleSet.add( new RewriteInjectSparkPReadCheckpointing()          ); //dependency: reblock
			
			//add statement block rewrite rules
			if( OptimizerUtils.ALLOW_BRANCH_REMOVAL )
				_sbRuleSet.add(  new RewriteRemoveUnnecessaryBranches()          ); //dependency: constant folding
			if( OptimizerUtils.ALLOW_FOR_LOOP_REMOVAL )
				_sbRuleSet.add(  new RewriteRemoveForLoopEmptySequence()         ); //dependency: constant folding
			if( OptimizerUtils.ALLOW_BRANCH_REMOVAL || OptimizerUtils.ALLOW_FOR_LOOP_REMOVAL )
				_sbRuleSet.add(  new RewriteMergeBlockSequence()                 ); //dependency: remove branches, remove for-loops
			if(OptimizerUtils.ALLOW_COMPRESSION_REWRITE)
				_sbRuleSet.add(      new RewriteCompressedReblock()              ); // Compression Rewrite
 			if( OptimizerUtils.ALLOW_SPLIT_HOP_DAGS )
 				_sbRuleSet.add(  new RewriteSplitDagUnknownCSVRead()             ); //dependency: reblock, merge blocks
 			if( OptimizerUtils.ALLOW_SPLIT_HOP_DAGS && 
 				ConfigurationManager.getCompilerConfigFlag(ConfigType.ALLOW_INDIVIDUAL_SB_SPECIFIC_OPS) )
 				_sbRuleSet.add(  new RewriteSplitDagDataDependentOperators()     ); //dependency: merge blocks
 			if( OptimizerUtils.ALLOW_AUTO_VECTORIZATION )
				_sbRuleSet.add(  new RewriteForLoopVectorization()               ); //dependency: reblock (reblockop)
 			_sbRuleSet.add( new RewriteInjectSparkLoopCheckpointing(true)        ); //dependency: reblock (blocksizes)
 			if( OptimizerUtils.ALLOW_CODE_MOTION )
 				_sbRuleSet.add(  new RewriteHoistLoopInvariantOperations()       ); //dependency: vectorize, but before inplace
 			if( OptimizerUtils.ALLOW_LOOP_UPDATE_IN_PLACE )
 				_sbRuleSet.add(  new RewriteMarkLoopVariablesUpdateInPlace()     );
 			if( LineageCacheConfig.getCompAssRW() )
 				_sbRuleSet.add(  new MarkForLineageReuse()                       );
 			_sbRuleSet.add(      new RewriteRemoveTransformEncodeMeta()          );
 			_dagRuleSet.add( new RewriteNonScalarPrint()                         );
 		}
		
		// DYNAMIC REWRITES (which do require size information)
		if( dynamicRewrites )
		{
			if ( DMLScript.USE_ACCELERATOR ){
				_dagRuleSet.add( new RewriteGPUSpecificOps() );	// gpu-specific rewrites
			}
			if ( DMLScript.APPLY_GENERATED_REWRITES ) {
				_dagRuleSet.add(new RewriteAutomaticallyGenerated(new GeneratedRewriteClass()));
			}
			if ( OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES) {
				_dagRuleSet.add( new RewriteMatrixMultChainOptimization()         ); //dependency: cse
				if( OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES2 )
					_dagRuleSet.add( new RewriteElementwiseMultChainOptimization()); //dependency: cse
			}
			if(OptimizerUtils.ALLOW_ADVANCED_MMCHAIN_REWRITES){
				_dagRuleSet.add( new RewriteMatrixMultChainOptimizationTranspose()      ); //dependency: cse
				_dagRuleSet.add( new RewriteMatrixMultChainOptimizationSparse()         ); //dependency: cse
			}
			if( OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION ) {
				_dagRuleSet.add( new RewriteAlgebraicSimplificationDynamic()      ); //dependencies: cse
				_dagRuleSet.add( new RewriteAlgebraicSimplificationStatic()       ); //dependencies: cse
			}
		}
		
		// cleanup after all rewrites applied 
		// (newly introduced operators, introduced redundancy after rewrites w/ multiple parents)
		if( OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION )
			_dagRuleSet.add( new RewriteRemoveUnnecessaryCasts()             );
		if( OptimizerUtils.ALLOW_COMMON_SUBEXPRESSION_ELIMINATION )
			_dagRuleSet.add( new RewriteCommonSubexpressionElimination(true) );
		if( OptimizerUtils.ALLOW_CONSTANT_FOLDING )
			_dagRuleSet.add( new RewriteConstantFolding()                    ); //dependency: cse
		_sbRuleSet.add(  new RewriteRemoveEmptyBasicBlocks()                 );
		_sbRuleSet.add(  new RewriteRemoveEmptyForLoops()                    );
	}
	
	/**
	 * Construct a program rewriter for a given rewrite which is passed from outside.
	 * 
	 * @param rewrites the HOP rewrite rules
	 */
	public ProgramRewriter(HopRewriteRule... rewrites) {
		//initialize HOP DAG rewrite ruleSet (with fixed rewrite order)
		_dagRuleSet = new ArrayList<>();
		for( HopRewriteRule rewrite : rewrites )
			_dagRuleSet.add( rewrite );
		_sbRuleSet = new ArrayList<>();
	}
	
	/**
	 * Construct a program rewriter for a given rewrite which is passed from outside.
	 * 
	 * @param rewrites the statement block rewrite rules
	 */
	public ProgramRewriter(StatementBlockRewriteRule... rewrites) {
		//initialize HOP DAG rewrite ruleSet (with fixed rewrite order)
		_dagRuleSet = new ArrayList<>();
		_sbRuleSet = new ArrayList<>();
		for( StatementBlockRewriteRule rewrite : rewrites )
			_sbRuleSet.add( rewrite );
	}
	
	/**
	 * Construct a program rewriter for the given rewrite sets which are passed from outside.
	 * 
	 * @param hRewrites HOP rewrite rules
	 * @param sbRewrites statement block rewrite rules
	 */
	public ProgramRewriter(ArrayList<HopRewriteRule> hRewrites, ArrayList<StatementBlockRewriteRule> sbRewrites) {
		//initialize HOP DAG rewrite ruleSet (with fixed rewrite order)
		_dagRuleSet = new ArrayList<>();
		_dagRuleSet.addAll( hRewrites );
		
		_sbRuleSet = new ArrayList<>();
		_sbRuleSet.addAll( sbRewrites );
	}
	
	public void removeHopRewrite(Class<? extends HopRewriteRule> clazz) {
		_dagRuleSet.removeIf(r -> r.getClass().equals(clazz));
	}
	
	public void removeStatementBlockRewrite(Class<? extends StatementBlockRewriteRule> clazz) {
		_sbRuleSet.removeIf(r -> r.getClass().equals(clazz));
	}
	
	public ProgramRewriteStatus rewriteProgramHopDAGs(DMLProgram dmlp) {
		return rewriteProgramHopDAGs(dmlp, true);
	}
	
	public ProgramRewriteStatus rewriteProgramHopDAGs(DMLProgram dmlp, boolean splitDags) {
		return rewriteProgramHopDAGs(dmlp, splitDags, new ProgramRewriteStatus());
	}
	
	public ProgramRewriteStatus rewriteProgramHopDAGs(DMLProgram dmlp, boolean splitDags, ProgramRewriteStatus state) {
		// for each namespace, handle function statement blocks
		for (String namespaceKey : dmlp.getNamespaces().keySet())
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey,fname);
				rewriteHopDAGsFunction(fsblock, state, splitDags);
			}
		
		// handle regular statement blocks in "main" method
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			rRewriteStatementBlockHopDAGs(current, state);
		}
		if( !_sbRuleSet.isEmpty() )
			dmlp.setStatementBlocks(rRewriteStatementBlocks(
				dmlp.getStatementBlocks(), state, splitDags));
		
		return state;
	}
	
	public void rewriteHopDAGsFunction(FunctionStatementBlock fsb, boolean splitDags) {
		rewriteHopDAGsFunction(fsb, new ProgramRewriteStatus(), splitDags);
	}
	
	public void rewriteHopDAGsFunction(FunctionStatementBlock fsb, ProgramRewriteStatus state, boolean splitDags) {
		rRewriteStatementBlockHopDAGs(fsb, state);
		if( !_sbRuleSet.isEmpty() )
			rRewriteStatementBlock(fsb, state, splitDags);
	}
	
	public void rRewriteStatementBlockHopDAGs(StatementBlock current, ProgramRewriteStatus state) {
		//ensure robustness for calls from outside
		if( state == null )
			state = new ProgramRewriteStatus();
		
		if (current instanceof FunctionStatementBlock)
		{
			FunctionStatementBlock fsb = (FunctionStatementBlock)current;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for (StatementBlock sb : fstmt.getBody())
				rRewriteStatementBlockHopDAGs(sb, state);
		}
		else if (current instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock) current;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			wsb.setPredicateHops(rewriteHopDAG(wsb.getPredicateHops(), state));
			for (StatementBlock sb : wstmt.getBody())
				rRewriteStatementBlockHopDAGs(sb, state);
		}	
		else if (current instanceof IfStatementBlock)
		{
			IfStatementBlock isb = (IfStatementBlock) current;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			isb.setPredicateHops(rewriteHopDAG(isb.getPredicateHops(), state));
			for (StatementBlock sb : istmt.getIfBody())
				rRewriteStatementBlockHopDAGs(sb, state);
			for (StatementBlock sb : istmt.getElseBody())
				rRewriteStatementBlockHopDAGs(sb, state);
		}
		else if (current instanceof ForStatementBlock) //incl parfor
		{
			ForStatementBlock fsb = (ForStatementBlock) current;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			fsb.setFromHops(rewriteHopDAG(fsb.getFromHops(), state));
			fsb.setToHops(rewriteHopDAG(fsb.getToHops(), state));
			fsb.setIncrementHops(rewriteHopDAG(fsb.getIncrementHops(), state));
			for (StatementBlock sb : fstmt.getBody())
				rRewriteStatementBlockHopDAGs(sb, state);
		}
		else //generic (last-level)
		{
			current.setHops( rewriteHopDAG(current.getHops(), state) );
		}
	}
	
	public ArrayList<Hop> rewriteHopDAG(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		for( HopRewriteRule r : _dagRuleSet ) {
			Hop.resetVisitStatus( roots ); //reset for each rule
			roots = r.rewriteHopDAGs(roots, state);
			if( CHECK )
				HopDagValidator.validateHopDag(roots, r);
		}
		return roots;
	}
	
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if( root == null )
			return null;
		
		for( HopRewriteRule r : _dagRuleSet ) {
			root.resetVisitStatus(); //reset for each rule
			root = r.rewriteHopDAG(root, state);
			if( CHECK )
				HopDagValidator.validateHopDag(root, r);
		}
		return root;
	}
	
	public ArrayList<StatementBlock> rRewriteStatementBlocks(ArrayList<StatementBlock> sbs, ProgramRewriteStatus status, boolean splitDags) {
		//ensure robustness for calls from outside
		if( status == null )
			status = new ProgramRewriteStatus();
		
		//apply rewrite rules to list of statement blocks
		List<StatementBlock> tmp = sbs; 
		for( StatementBlockRewriteRule r : _sbRuleSet )
			if( splitDags || !r.createsSplitDag() )
				tmp = r.rewriteStatementBlocks(tmp, status);
		
		//recursively rewrite statement blocks (with potential expansion)
		List<StatementBlock> tmp2 = new ArrayList<>();
		for( StatementBlock sb : tmp )
			tmp2.addAll( rRewriteStatementBlock(sb, status, splitDags) );
		
		//apply rewrite rules to list of statement blocks (with potential contraction)
		for( StatementBlockRewriteRule r : _sbRuleSet )
			if( splitDags || !r.createsSplitDag() )
				tmp2 = r.rewriteStatementBlocks(tmp2, status);
		
		//prepare output list
		sbs.clear();
		sbs.addAll(tmp2);
		return sbs;
	}
	
	public ArrayList<StatementBlock> rRewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus status, boolean splitDags) {
		ArrayList<StatementBlock> ret = new ArrayList<>();
		ret.add(sb);
		
		//recursive invocation
		if (sb instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			fstmt.setBody(rRewriteStatementBlocks(fstmt.getBody(), status, splitDags));
		}
		else if (sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			wstmt.setBody(rRewriteStatementBlocks(wstmt.getBody(), status, splitDags));
		}	
		else if (sb instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			istmt.setIfBody(rRewriteStatementBlocks(istmt.getIfBody(), status, splitDags));
			istmt.setElseBody(rRewriteStatementBlocks(istmt.getElseBody(), status, splitDags));
		}
		else if (sb instanceof ForStatementBlock) { //incl parfor
			//maintain parfor context information (e.g., for checkpointing)
			boolean prestatus = status.isInParforContext();
			if( sb instanceof ParForStatementBlock )
				status.setInParforContext(true);
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			fstmt.setBody(rRewriteStatementBlocks(fstmt.getBody(), status, splitDags));
			status.setInParforContext(prestatus);
		}
		
		//apply rewrite rules to individual statement blocks
		for( StatementBlockRewriteRule r : _sbRuleSet ) {
			if( !splitDags && r.createsSplitDag() )
				continue;
			ArrayList<StatementBlock> tmp = new ArrayList<>();
			for( StatementBlock sbc : ret )
				tmp.addAll( r.rewriteStatementBlock(sbc, status) );
			
			//take over set of rewritten sbs
			ret.clear();
			ret.addAll(tmp);
		}
		
		return ret;
	}
}
