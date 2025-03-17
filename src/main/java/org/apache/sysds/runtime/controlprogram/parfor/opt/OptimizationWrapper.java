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

package org.apache.sysds.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.ipa.InterProceduralAnalysis;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.hops.recompile.Recompiler.ResetType;
import org.apache.sysds.hops.rewrite.HopRewriteRule;
import org.apache.sysds.hops.rewrite.ProgramRewriteStatus;
import org.apache.sysds.hops.rewrite.ProgramRewriter;
import org.apache.sysds.hops.rewrite.RewriteConstantFolding;
import org.apache.sysds.hops.rewrite.RewriteRemoveUnnecessaryBranches;
import org.apache.sysds.hops.rewrite.StatementBlockRewriteRule;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ParForStatementBlock;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.POptMode;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.opt.Optimizer.CostModelType;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.apache.sysds.utils.stats.ParForStatistics;
import org.apache.sysds.utils.stats.Timing;


/**
 * Wrapper to ParFOR cost estimation and optimizer. This is intended to be the 
 * only public access to the optimizer package.
 * 
 * NOTE: There are two main alternatives for invocation of this OptimizationWrapper:
 * (1) During compilation (after creating rtprog), (2) on execute of all top-level ParFOR PBs.
 * We decided to use (2) (and carry the SBs during execution) due to the following advantages
 *   - Known Statistics: problem size of top-level parfor known, in general, less unknown statistics
 *   - No Overhead: preventing overhead for non-parfor scripts (finding top-level parfors)
 *   - Simplicity: no need of finding top-level parfors 
 * 
 */
public class OptimizationWrapper 
{
	private static final Log LOG = LogFactory.getLog(OptimizationWrapper.class.getName());
	
	//internal parameters
	public static final double PAR_FACTOR_INFRASTRUCTURE = 1.0;


	/**
	 * Called once per top-level parfor (during runtime, on parfor execute)
	 * in order to optimize the specific parfor program block.
	 * 
	 * NOTE: this is the default way to invoke parfor optimizers.
	 * 
	 * @param type ?
	 * @param sb parfor statement block
	 * @param pb parfor program block
	 * @param ec execution context
	 * @param numRuns number of optimizations performed so far
	 */
	public static void optimize( POptMode type, ParForStatementBlock sb, ParForProgramBlock pb, ExecutionContext ec, int numRuns ) 
	{
		Timing time = new Timing(true);
		
		LOG.debug("ParFOR Opt: Running optimization for ParFOR("+pb.getID()+")");
		
		
		//set max contraints if not specified
		int ck = UtilFunctions.toInt( Math.max( InfrastructureAnalyzer.getCkMaxCP(),
			InfrastructureAnalyzer.getCkMaxMR() ) * PAR_FACTOR_INFRASTRUCTURE );
		double cm = InfrastructureAnalyzer.getCmMax() * OptimizerUtils.MEM_UTIL_FACTOR; 
		
		//execute optimizer
		optimize( type, ck, cm, sb, pb, ec, numRuns );
		
		double timeVal = time.stop();
		LOG.debug("ParFOR Opt: Finished optimization for PARFOR("+pb.getID()+") in "+timeVal+"ms.");
		if( DMLScript.STATISTICS ) {
			ParForStatistics.incrementOptimCount();
			ParForStatistics.incrementOptimTime((long)timeVal);
		}
	}

	private static void optimize( POptMode otype, int ck, double cm,
		ParForStatementBlock sb, ParForProgramBlock pb, ExecutionContext ec, int numRuns ) 
	{
		//create specified optimizer
		Optimizer opt = createOptimizer( otype );
		CostModelType cmtype = opt.getCostModelType();
		LOG.trace("ParFOR Opt: Created optimizer ("+otype+","+opt.getCostModelType());
		
		OptTree tree = null;
		
		//recompile parfor body 
		if( ConfigurationManager.isDynamicRecompilation() )
		{
			ForStatement fs = (ForStatement) sb.getStatement(0);
			
			//debug output before recompilation
			if( LOG.isDebugEnabled() ) {
				try {
					tree = OptTreeConverter.createOptTree(ck, cm, sb, pb, ec); 
					LOG.debug("ParFOR Opt: Input plan (before recompilation):\n" + tree.explain(false));
				}
				catch(Exception ex)
				{
					throw new DMLRuntimeException("Unable to create opt tree.", ex);
				}
			}
			
			//constant propagation into parfor body 
			//(input scalars to parfor are guaranteed read only, but need to ensure safe-replace on multiple reopt
			//separate propagation required because recompile in-place without literal replacement)
			try{
				LocalVariableMap constVars = ProgramRecompiler.getReusableScalarVariables(sb.getDMLProg(), sb, ec.getVariables());
				ProgramRecompiler.replaceConstantScalarVariables(sb, constVars);
			}
			catch(Exception ex){
				throw new DMLRuntimeException(ex);
			}
			
			//program rewrites (e.g., constant folding, branch removal) according to replaced literals
			try {
				ProgramRewriter rewriter = createProgramRewriterWithRuleSets();
				ProgramRewriteStatus state = new ProgramRewriteStatus();
				rewriter.rRewriteStatementBlockHopDAGs( sb, state );
				fs.setBody(rewriter.rRewriteStatementBlocks(fs.getBody(), state, true));
				if( state.getRemovedBranches() ){
					LOG.debug("ParFOR Opt: Removed branches during program rewrites, rebuilding runtime program");
					pb.setChildBlocks(ProgramRecompiler.generatePartitialRuntimeProgram(pb.getProgram(), fs.getBody()));
				}
			}
			catch(Exception ex){
				throw new DMLRuntimeException(ex);
			}
			
			//recompilation of parfor body and called functions (if safe)
			try{
				//core parfor body recompilation (based on symbol table entries)
				//* clone of variables in order to allow for statistics propagation across DAGs
				//(tid=0, because deep copies created after opt)
				LocalVariableMap tmp = (LocalVariableMap) ec.getVariables().clone();
				ResetType reset = ConfigurationManager.isCodegenEnabled() ? 
					ResetType.RESET_KNOWN_DIMS : ResetType.RESET;
				Recompiler.recompileProgramBlockHierarchy(pb.getChildBlocks(), tmp, 0, true, reset);
				
				//inter-procedural optimization (based on previous recompilation)
				if( pb.hasFunctions() ) {
					InterProceduralAnalysis ipa = new InterProceduralAnalysis(sb);
					Set<String> fcand = ipa.analyzeSubProgram();
					
					if( !fcand.isEmpty() ) {
						//regenerate runtime program of modified functions
						for( String func : fcand )
						{
							String[] funcparts = DMLProgram.splitFunctionKey(func);
							FunctionProgramBlock fpb = pb.getProgram().getFunctionProgramBlock(funcparts[0], funcparts[1]);
							//reset recompilation flags according to recompileOnce because it is only safe if function is recompileOnce 
							//because then recompiled for every execution (otherwise potential issues if func also called outside parfor)
							ResetType reset2 = fpb.isRecompileOnce() ? reset : ResetType.NO_RESET;
							Recompiler.recompileProgramBlockHierarchy(fpb.getChildBlocks(), new LocalVariableMap(), 0, true, reset2);
						}
					}
				}
			}
			catch(Exception ex){
				throw new DMLRuntimeException(ex);
			}
		}
		
		//create opt tree (before optimization)
		try {
			tree = OptTreeConverter.createOptTree(ck, cm, sb, pb, ec); 
			if(LOG.isDebugEnabled())
				LOG.debug("ParFOR Opt: Input plan (before optimization):\n" + tree.explain(false));
		}
		catch(Exception ex) {
			throw new DMLRuntimeException("Unable to create opt tree.", ex);
		}
		
		//create cost estimator
		CostEstimator est = createCostEstimator( cmtype, tree, ec.getVariables() );
		LOG.trace("ParFOR Opt: Created cost estimator ("+cmtype+")");
		
		//core optimize
		opt.optimize(sb, pb, tree, est, numRuns, ec);
		if(LOG.isDebugEnabled())
			LOG.debug("ParFOR Opt: Optimized plan (after optimization): \n" + tree.explain(false));
	}

	private static Optimizer createOptimizer( POptMode otype ) {
		switch( otype ) {
			case HEURISTIC:   return new OptimizerHeuristic();
			case RULEBASED:   return new OptimizerRuleBased();
			case CONSTRAINED: return new OptimizerConstrained();
			default:
				throw new DMLRuntimeException("Undefined optimizer: '"+otype+"'.");
		}
	}

	private static CostEstimator createCostEstimator( CostModelType cmtype, OptTree tree, LocalVariableMap vars )  {
		switch( cmtype ) {
			case STATIC_MEM_METRIC:
				return new CostEstimatorHops(tree.getPlanMapping());
			case RUNTIME_METRICS:
				return new CostEstimatorRuntime(tree.getPlanMapping(), (LocalVariableMap)vars.clone());
			default:
				throw new DMLRuntimeException("Undefined cost model type: '"+cmtype+"'.");
		}
	}

	private static ProgramRewriter createProgramRewriterWithRuleSets()
	{
		//create hop rewrite set
		ArrayList<HopRewriteRule> hRewrites = new ArrayList<>();
		hRewrites.add( new RewriteConstantFolding() );
		
		//create statementblock rewrite set
		ArrayList<StatementBlockRewriteRule> sbRewrites = new ArrayList<>();
		sbRewrites.add( new RewriteRemoveUnnecessaryBranches() );
		
		ProgramRewriter rewriter = new ProgramRewriter( hRewrites, sbRewrites );
		
		return rewriter;
	}
}
