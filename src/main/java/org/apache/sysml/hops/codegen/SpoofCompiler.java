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

package org.apache.sysml.hops.codegen;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map.Entry;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.codegen.cplan.CNode;
import org.apache.sysml.hops.codegen.cplan.CNodeCell;
import org.apache.sysml.hops.codegen.cplan.CNodeData;
import org.apache.sysml.hops.codegen.cplan.CNodeMultiAgg;
import org.apache.sysml.hops.codegen.cplan.CNodeOuterProduct;
import org.apache.sysml.hops.codegen.cplan.CNodeRow;
import org.apache.sysml.hops.codegen.cplan.CNodeTernary;
import org.apache.sysml.hops.codegen.cplan.CNodeTernary.TernaryType;
import org.apache.sysml.hops.codegen.cplan.CNodeTpl;
import org.apache.sysml.hops.codegen.template.TemplateBase;
import org.apache.sysml.hops.codegen.template.TemplateBase.CloseType;
import org.apache.sysml.hops.codegen.template.TemplateBase.TemplateType;
import org.apache.sysml.hops.codegen.template.CPlanCSERewriter;
import org.apache.sysml.hops.codegen.template.CPlanMemoTable;
import org.apache.sysml.hops.codegen.template.PlanSelection;
import org.apache.sysml.hops.codegen.template.PlanSelectionFuseCostBased;
import org.apache.sysml.hops.codegen.template.PlanSelectionFuseAll;
import org.apache.sysml.hops.codegen.template.PlanSelectionFuseNoRedundancy;
import org.apache.sysml.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysml.hops.codegen.template.CPlanMemoTable.MemoTableEntrySet;
import org.apache.sysml.hops.codegen.template.TemplateUtils;
import org.apache.sysml.hops.recompile.RecompileStatus;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.OpOp1;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.hops.rewrite.ProgramRewriteStatus;
import org.apache.sysml.hops.rewrite.ProgramRewriter;
import org.apache.sysml.hops.rewrite.RewriteCommonSubexpressionElimination;
import org.apache.sysml.hops.rewrite.RewriteRemoveUnnecessaryCasts;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.ForStatement;
import org.apache.sysml.parser.ForStatementBlock;
import org.apache.sysml.parser.FunctionStatement;
import org.apache.sysml.parser.FunctionStatementBlock;
import org.apache.sysml.parser.IfStatement;
import org.apache.sysml.parser.IfStatementBlock;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.WhileStatement;
import org.apache.sysml.parser.WhileStatementBlock;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.codegen.CodegenUtils;
import org.apache.sysml.runtime.codegen.SpoofCellwise.CellType;
import org.apache.sysml.runtime.codegen.SpoofRowwise.RowType;
import org.apache.sysml.runtime.controlprogram.ForProgramBlock;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.IfProgramBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.utils.Explain;
import org.apache.sysml.utils.Statistics;

public class SpoofCompiler
{
	private static final Log LOG = LogFactory.getLog(SpoofCompiler.class.getName());
	
	//internal configuration flags
	public static boolean LDEBUG                      = false;
	public static CompilerType JAVA_COMPILER          = CompilerType.JANINO; 
	public static IntegrationType INTEGRATION         = IntegrationType.RUNTIME;
	public static final boolean RECOMPILE_CODEGEN     = true;
	public static final boolean PRUNE_REDUNDANT_PLANS = true;
	public static PlanCachePolicy PLAN_CACHE_POLICY   = PlanCachePolicy.CSLH;
	public static final int PLAN_CACHE_SIZE           = 1024; //max 1K classes 
	public static final PlanSelector PLAN_SEL_POLICY  = PlanSelector.FUSE_COST_BASED; 

	public enum CompilerType {
		JAVAC,
		JANINO,
	}
	
	public enum IntegrationType {
		HOPS,
		RUNTIME,
	}
	
	public enum PlanSelector {
		FUSE_ALL,             //maximal fusion, possible w/ redundant compute
		FUSE_NO_REDUNDANCY,   //fusion without redundant compute 
		FUSE_COST_BASED,      //cost-based decision on materialization points
	}

	public enum PlanCachePolicy {
		CONSTANT, //plan cache, with always compile literals
		CSLH,     //plan cache, with context-sensitive literal replacement heuristic
		NONE;     //no plan cache
		
		public static PlanCachePolicy get(boolean planCache, boolean compileLiterals) {
			return !planCache ? NONE : compileLiterals ? CONSTANT : CSLH;
		}
	}
	
	static {
		// for internal debugging only
		if( LDEBUG ) {
			Logger.getLogger("org.apache.sysml.hops.codegen")
				  .setLevel((Level) Level.TRACE);
		}
	}
	
	//plan cache for cplan->compiled source to avoid unnecessary codegen/source code compile
	//for equal operators from (1) different hop dags and (2) repeated recompilation 
	//note: if PLAN_CACHE_SIZE is exceeded, we evict the least-recently-used plan (LRU policy)
	private static final PlanCache planCache = new PlanCache(PLAN_CACHE_SIZE);
	
	private static ProgramRewriter rewriteCSE = new ProgramRewriter(
			new RewriteCommonSubexpressionElimination(true),
			new RewriteRemoveUnnecessaryCasts());
	
	public static void generateCode(DMLProgram dmlprog) 
		throws LanguageException, HopsException, DMLRuntimeException
	{
		// for each namespace, handle function statement blocks
		for (String namespaceKey : dmlprog.getNamespaces().keySet()) {
			for (String fname : dmlprog.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock fsblock = dmlprog.getFunctionStatementBlock(namespaceKey,fname);
				generateCodeFromStatementBlock(fsblock);
			}
		}
		
		// handle regular statement blocks in "main" method
		for (int i = 0; i < dmlprog.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlprog.getStatementBlock(i);
			generateCodeFromStatementBlock(current);
		}
	}
	
	public static void generateCode(Program rtprog) 
		throws LanguageException, HopsException, DMLRuntimeException, LopsException, IOException
	{
		// handle all function program blocks
		for( FunctionProgramBlock pb : rtprog.getFunctionProgramBlocks().values() )
			generateCodeFromProgramBlock(pb);
		
		// handle regular program blocks in "main" method
		for( ProgramBlock pb : rtprog.getProgramBlocks() )
			generateCodeFromProgramBlock(pb);
	}
	
	public static void generateCodeFromStatementBlock(StatementBlock current)
		throws HopsException, DMLRuntimeException
	{
		if (current instanceof FunctionStatementBlock)
		{
			FunctionStatementBlock fsb = (FunctionStatementBlock)current;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for (StatementBlock sb : fstmt.getBody())
				generateCodeFromStatementBlock(sb);
		}
		else if (current instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock) current;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			wsb.setPredicateHops(optimize(wsb.getPredicateHops(), false));
			for (StatementBlock sb : wstmt.getBody())
				generateCodeFromStatementBlock(sb);
		}
		else if (current instanceof IfStatementBlock)
		{
			IfStatementBlock isb = (IfStatementBlock) current;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			isb.setPredicateHops(optimize(isb.getPredicateHops(), false));
			for (StatementBlock sb : istmt.getIfBody())
				generateCodeFromStatementBlock(sb);
			for (StatementBlock sb : istmt.getElseBody())
				generateCodeFromStatementBlock(sb);
		}
		else if (current instanceof ForStatementBlock) //incl parfor
		{
			ForStatementBlock fsb = (ForStatementBlock) current;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			fsb.setFromHops(optimize(fsb.getFromHops(), false));
			fsb.setToHops(optimize(fsb.getToHops(), false));
			fsb.setIncrementHops(optimize(fsb.getIncrementHops(), false));
			for (StatementBlock sb : fstmt.getBody())
				generateCodeFromStatementBlock(sb);
		}
		else //generic (last-level)
		{
			current.set_hops( generateCodeFromHopDAGs(current.get_hops()) );
			current.updateRecompilationFlag();
		}
	}
	
	public static void generateCodeFromProgramBlock(ProgramBlock current)
		throws HopsException, DMLRuntimeException, LopsException, IOException
	{
		if (current instanceof FunctionProgramBlock)
		{
			FunctionProgramBlock fsb = (FunctionProgramBlock)current;
			for (ProgramBlock pb : fsb.getChildBlocks())
				generateCodeFromProgramBlock(pb);
		}
		else if (current instanceof WhileProgramBlock)
		{
			WhileProgramBlock wpb = (WhileProgramBlock) current;
			WhileStatementBlock wsb = (WhileStatementBlock)wpb.getStatementBlock();
			
			if( wsb!=null && wsb.getPredicateHops()!=null )
				wpb.setPredicate(generateCodeFromHopDAGsToInst(wsb.getPredicateHops()));
			for (ProgramBlock sb : wpb.getChildBlocks())
				generateCodeFromProgramBlock(sb);
		}
		else if (current instanceof IfProgramBlock)
		{
			IfProgramBlock ipb = (IfProgramBlock) current;
			IfStatementBlock isb = (IfStatementBlock) ipb.getStatementBlock();
			if( isb!=null && isb.getPredicateHops()!=null )
				ipb.setPredicate(generateCodeFromHopDAGsToInst(isb.getPredicateHops()));
			for (ProgramBlock pb : ipb.getChildBlocksIfBody())
				generateCodeFromProgramBlock(pb);
			for (ProgramBlock pb : ipb.getChildBlocksElseBody())
				generateCodeFromProgramBlock(pb);
		}
		else if (current instanceof ForProgramBlock) //incl parfor
		{
			ForProgramBlock fpb = (ForProgramBlock) current;
			ForStatementBlock fsb = (ForStatementBlock) fpb.getStatementBlock();
			if( fsb!=null && fsb.getFromHops()!=null )
				fpb.setFromInstructions(generateCodeFromHopDAGsToInst(fsb.getFromHops()));
			if( fsb!=null && fsb.getToHops()!=null )
				fpb.setToInstructions(generateCodeFromHopDAGsToInst(fsb.getToHops()));
			if( fsb!=null && fsb.getIncrementHops()!=null )
				fpb.setIncrementInstructions(generateCodeFromHopDAGsToInst(fsb.getIncrementHops()));
			for (ProgramBlock pb : fpb.getChildBlocks())
				generateCodeFromProgramBlock(pb);
		}
		else //generic (last-level)
		{
			StatementBlock sb = current.getStatementBlock();
			current.setInstructions( generateCodeFromHopDAGsToInst(sb, sb.get_hops()) );
		}
	}

	public static ArrayList<Hop> generateCodeFromHopDAGs(ArrayList<Hop> roots) 
		throws HopsException, DMLRuntimeException
	{
		if( roots == null )
			return roots;

		ArrayList<Hop> optimized = SpoofCompiler.optimize(roots, false);
		Hop.resetVisitStatus(roots);
		Hop.resetVisitStatus(optimized);
		
		return optimized;
	}
	
	public static ArrayList<Instruction> generateCodeFromHopDAGsToInst(StatementBlock sb, ArrayList<Hop> roots) 
		throws DMLRuntimeException, HopsException, LopsException, IOException 
	{
		//create copy of hop dag, call codegen, and generate instructions
		return Recompiler.recompileHopsDag(sb, roots, 
			new LocalVariableMap(), new RecompileStatus(true), false, false, 0);
	}
	
	public static ArrayList<Instruction> generateCodeFromHopDAGsToInst(Hop root) 
		throws DMLRuntimeException, HopsException, LopsException, IOException 
	{
		//create copy of hop dag, call codegen, and generate instructions
		return Recompiler.recompileHopsDag(root, 
			new LocalVariableMap(), new RecompileStatus(true), false, false, 0);
	}
	
	
	/**
	 * Main interface of sum-product optimizer, predicate dag.
	 * 
	 * @param root dag root node
	 * @param recompile true if invoked during dynamic recompilation
	 * @return dag root node of modified dag
	 * @throws DMLRuntimeException if optimization failed
	 */
	public static Hop optimize( Hop root, boolean recompile ) throws DMLRuntimeException {
		if( root == null )
			return root;
		
		return optimize(new ArrayList<Hop>(
			Collections.singleton(root)), recompile).get(0);
	}
	
	/**
	 * Main interface of sum-product optimizer, statement block dag.
	 * 
	 * @param roots dag root nodes
	 * @param recompile true if invoked during dynamic recompilation
	 * @return dag root nodes of modified dag 
	 * @throws DMLRuntimeException if optimization failed
	 */
	public static ArrayList<Hop> optimize(ArrayList<Hop> roots, boolean recompile) 
		throws DMLRuntimeException 
	{
		if( roots == null || roots.isEmpty() )
			return roots;
	
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		ArrayList<Hop> ret = roots;
		
		try
		{
			//context-sensitive literal replacement (only integers during recompile)
			boolean compileLiterals = (PLAN_CACHE_POLICY==PlanCachePolicy.CONSTANT) || !recompile;
			
			//construct codegen plans
			HashMap<Long, Pair<Hop[],CNodeTpl>>  cplans = constructCPlans(roots, compileLiterals);
			
			//cleanup codegen plans (remove unnecessary inputs, fix hop-cnodedata mapping,
			//remove empty templates with single cnodedata input, remove spurious lookups,
			//perform common subexpression elimination)
			cplans = cleanupCPlans(cplans);
			
			//explain before modification
			if( LOG.isTraceEnabled() && !cplans.isEmpty() ) { //existing cplans
				LOG.trace("Codegen EXPLAIN (before optimize): \n"+Explain.explainHops(roots));
			}
			
			//source code generation for all cplans
			HashMap<Long, Pair<Hop[],Class<?>>> clas = new HashMap<Long, Pair<Hop[],Class<?>>>();
			for( Entry<Long, Pair<Hop[],CNodeTpl>> cplan : cplans.entrySet() ) 
			{
				Pair<Hop[],CNodeTpl> tmp = cplan.getValue();
				Class<?> cla = planCache.getPlan(tmp.getValue());
				
				if( cla == null ) {
					//generate java source code
					String src = tmp.getValue().codegen(false);
					
					//explain debug output cplans or generated source code
					if( LOG.isTraceEnabled() || DMLScript.EXPLAIN.isHopsType(recompile) ) {
						LOG.info("Codegen EXPLAIN (generated cplan for HopID: " 
							+ cplan.getKey() + ", line "+tmp.getValue().getBeginLine() + "):");
						LOG.info(tmp.getValue().getClassname()
							+ Explain.explainCPlan(cplan.getValue().getValue()));
					}
					if( LOG.isTraceEnabled() || DMLScript.EXPLAIN.isRuntimeType(recompile) ) {
						LOG.info("Codegen EXPLAIN (generated code for HopID: "
							+ cplan.getKey() + ", line "+tmp.getValue().getBeginLine() + "):");
						LOG.info(src);
					}
					
					//compile generated java source code
					cla = CodegenUtils.compileClass("codegen."+
							tmp.getValue().getClassname(), src);
					
					//maintain plan cache
					if( PLAN_CACHE_POLICY!=PlanCachePolicy.NONE )
						planCache.putPlan(tmp.getValue(), cla);
				}
				else if( DMLScript.STATISTICS ) {
					Statistics.incrementCodegenPlanCacheHits();
				}
				
				//make class available and maintain hits
				if(cla != null)
					clas.put(cplan.getKey(), new Pair<Hop[],Class<?>>(tmp.getKey(),cla));
				if( DMLScript.STATISTICS )
					Statistics.incrementCodegenPlanCacheTotal();
			}
			
			//create modified hop dag (operator replacement and CSE)
			if( !cplans.isEmpty() ) 
			{
				//generate final hop dag
				ret = constructModifiedHopDag(roots, cplans, clas);
				
				//run common subexpression elimination and other rewrites
				ret = rewriteCSE.rewriteHopDAGs(ret, new ProgramRewriteStatus());	
				
				//explain after modification
				if( LOG.isTraceEnabled() ) {
					LOG.trace("Codegen EXPLAIN (after optimize): \n"+Explain.explainHops(roots));
				}
			}
		}
		catch( Exception ex ) {
			LOG.error("Codegen failed to optimize the following HOP DAG: \n" + 
				Explain.explainHops(roots));
			throw new DMLRuntimeException(ex);
		}
		
		if( DMLScript.STATISTICS ) {
			Statistics.incrementCodegenDAGCompile();
			Statistics.incrementCodegenCompileTime(System.nanoTime()-t0);
		}
		
		Hop.resetVisitStatus(roots);
			
		return ret;
	}

	public static void cleanupCodeGenerator() {
		if( PLAN_CACHE_POLICY != PlanCachePolicy.NONE ) {
			CodegenUtils.clearClassCache(); //class cache
			planCache.clear(); //plan cache
		}
	}
	
	/**
	 * Factory method for alternative plan selection policies.
	 * 
	 * @return plan selector
	 */
	public static PlanSelection createPlanSelector() {
		switch( PLAN_SEL_POLICY ) {
			case FUSE_ALL: 
				return new PlanSelectionFuseAll();
			case FUSE_NO_REDUNDANCY: 
				return new PlanSelectionFuseNoRedundancy();
			case FUSE_COST_BASED:
				return new PlanSelectionFuseCostBased();
			default:	
				throw new RuntimeException("Unsupported "
					+ "plan selector: "+PLAN_SEL_POLICY);
		}
	}
	
	public static void setExecTypeSpecificJavaCompiler() {
		JAVA_COMPILER = OptimizerUtils.isSparkExecutionMode() ?
			CompilerType.JANINO : CompilerType.JAVAC;
	}
	
	////////////////////
	// Codegen plan construction

	private static HashMap<Long, Pair<Hop[],CNodeTpl>> constructCPlans(ArrayList<Hop> roots, boolean compileLiterals) throws DMLException
	{
		//explore cplan candidates
		CPlanMemoTable memo = new CPlanMemoTable();
		for( Hop hop : roots )
			rExploreCPlans(hop, memo, compileLiterals);
		
		//select optimal cplan candidates
		memo.pruneSuboptimal(roots);
		
		//construct actual cplan representations
		//note: we do not use the hop visit status due to jumps over fused operators which would
		//corrupt subsequent resets, leaving partial hops dags in visited status
		LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> ret = new LinkedHashMap<Long, Pair<Hop[],CNodeTpl>>();
		HashSet<Long> visited = new HashSet<Long>();
		for( Hop hop : roots )
			rConstructCPlans(hop, memo, ret, compileLiterals, visited);
		
		return ret;
	}
	
	private static void rExploreCPlans(Hop hop, CPlanMemoTable memo, boolean compileLiterals) 
		throws DMLException
	{
		//top-down memoization of processed dag nodes
		if( memo.contains(hop.getHopID()) || memo.containsHop(hop) )
			return;
		
		//recursive candidate exploration
		for( Hop c : hop.getInput() )
			rExploreCPlans(c, memo, compileLiterals);
		
		//open initial operator plans, if possible
		for( TemplateBase tpl : TemplateUtils.TEMPLATES )
			if( tpl.open(hop) ) {
				MemoTableEntrySet P = new MemoTableEntrySet(tpl.getType(), false);
				memo.addAll(hop, enumPlans(hop, -1, P, tpl, memo));
			}
		
		//fuse and merge operator plans
		for( Hop c : hop.getInput() ) {
			if( memo.contains(c.getHopID()) )
				for( MemoTableEntry me : memo.getDistinct(c.getHopID()) ) {
					TemplateBase tpl = TemplateUtils.createTemplate(me.type, me.closed);
					if( tpl.fuse(hop, c) ) {
						int pos = hop.getInput().indexOf(c);
						MemoTableEntrySet P = new MemoTableEntrySet(tpl.getType(), pos, c.getHopID(), tpl.isClosed());
						memo.addAll(hop, enumPlans(hop, pos, P, tpl, memo));
					}
				}	
		}
		
		//prune subsumed / redundant plans
		if( PRUNE_REDUNDANT_PLANS )
			memo.pruneRedundant(hop.getHopID());
		
		//close operator plans, if required
		if( memo.contains(hop.getHopID()) ) {
			Iterator<MemoTableEntry> iter = memo.get(hop.getHopID()).iterator();
			while( iter.hasNext() ) {
				MemoTableEntry me = iter.next();
				TemplateBase tpl = TemplateUtils.createTemplate(me.type);
				CloseType ccode = tpl.close(hop);
				if( ccode == CloseType.CLOSED_INVALID )
					iter.remove();
				else if( ccode == CloseType.CLOSED_VALID )
					me.closed = true;
			}
		}
		
		//mark visited even if no plans found (e.g., unsupported ops)
		memo.addHop(hop);
	}
	
	private static MemoTableEntrySet enumPlans(Hop hop, int pos, MemoTableEntrySet P, TemplateBase tpl, CPlanMemoTable memo) {
		for(int k=0; k<hop.getInput().size(); k++)
			if( k != pos ) {
				Hop input2 = hop.getInput().get(k);
				if( memo.contains(input2.getHopID()) && !memo.get(input2.getHopID()).get(0).closed
					&& TemplateUtils.isType(memo.get(input2.getHopID()).get(0).type, tpl.getType(), TemplateType.CellTpl)
					&& tpl.merge(hop, input2) && (tpl.getType()!=TemplateType.RowTpl || pos==-1 
						|| TemplateUtils.hasCommonRowTemplateMatrixInput(hop.getInput().get(pos), input2, memo)))
					P.crossProduct(k, -1L, input2.getHopID());
				else
					P.crossProduct(k, -1L);
			}
		return P;
	}
	
	private static void rConstructCPlans(Hop hop, CPlanMemoTable memo, HashMap<Long, Pair<Hop[],CNodeTpl>> cplans, boolean compileLiterals, HashSet<Long> visited) 
		throws DMLException
	{
		//top-down memoization of processed dag nodes
		if( hop == null || visited.contains(hop.getHopID()) )
			return;
		
		//generate cplan for existing memo table entry
		if( memo.containsTopLevel(hop.getHopID()) ) {
			cplans.put(hop.getHopID(), TemplateUtils
				.createTemplate(memo.getBest(hop.getHopID()).type)
				.constructCplan(hop, memo, compileLiterals));
			if( DMLScript.STATISTICS )
				Statistics.incrementCodegenCPlanCompile(1); 
		}
		
		//process children recursively, but skip compiled operator
		if( cplans.containsKey(hop.getHopID()) ) {
			for( Hop c : cplans.get(hop.getHopID()).getKey() )
				rConstructCPlans(c, memo, cplans, compileLiterals, visited);
		}
		else {
			for( Hop c : hop.getInput() )
				rConstructCPlans(c, memo, cplans, compileLiterals, visited);	
		}
		
		visited.add(hop.getHopID());
	}
	
	////////////////////
	// Codegen hop dag construction

	private static ArrayList<Hop> constructModifiedHopDag(ArrayList<Hop> orig, 
			HashMap<Long, Pair<Hop[],CNodeTpl>> cplans, HashMap<Long, Pair<Hop[],Class<?>>> cla)
	{
		HashSet<Long> memo = new HashSet<Long>();
		for( int i=0; i<orig.size(); i++ ) {
			Hop hop = orig.get(i); //w/o iterator because modified
			rConstructModifiedHopDag(hop, cplans, cla, memo);
		}
		return orig;
	}
	
	private static void rConstructModifiedHopDag(Hop hop,  HashMap<Long, Pair<Hop[],CNodeTpl>> cplans,
			HashMap<Long, Pair<Hop[],Class<?>>> clas, HashSet<Long> memo)
	{
		if( memo.contains(hop.getHopID()) )
			return; //already processed
		
		Hop hnew = hop;
		if( clas.containsKey(hop.getHopID()) ) 
		{
			//replace sub-dag with generated operator
			Pair<Hop[], Class<?>> tmpCla = clas.get(hop.getHopID());
			CNodeTpl tmpCNode = cplans.get(hop.getHopID()).getValue();
			hnew = new SpoofFusedOp(hop.getName(), hop.getDataType(), hop.getValueType(), 
					tmpCla.getValue(), false, tmpCNode.getOutputDimType());
			Hop[] inHops = tmpCla.getKey();
			for( int i=0; i<inHops.length; i++ ) {
				if( tmpCNode instanceof CNodeOuterProduct 
					&& inHops[i].getHopID()==((CNodeData)tmpCNode.getInput().get(2)).getHopID()
					&& !TemplateUtils.hasTransposeParentUnderOuterProduct(inHops[i]) ) {
					hnew.addInput(HopRewriteUtils.createTranspose(inHops[i]));
				}
				else
					hnew.addInput(inHops[i]); //add inputs
			}
			
			//modify output parameters 
			HopRewriteUtils.setOutputParameters(hnew, hop.getDim1(), hop.getDim2(), 
					hop.getRowsInBlock(), hop.getColsInBlock(), hop.getNnz());
			if(tmpCNode instanceof CNodeOuterProduct && ((CNodeOuterProduct)tmpCNode).isTransposeOutput() )
				hnew = HopRewriteUtils.createTranspose(hnew);
			else if( tmpCNode instanceof CNodeMultiAgg ) {
				ArrayList<Hop> roots = ((CNodeMultiAgg)tmpCNode).getRootNodes();
				hnew.setDataType(DataType.MATRIX);
				HopRewriteUtils.setOutputParameters(hnew, 1, roots.size(), 
					inHops[0].getRowsInBlock(), inHops[0].getColsInBlock(), -1);
				//inject artificial right indexing operations for all parents of all nodes
				for( int i=0; i<roots.size(); i++ ) {
					Hop hnewi = (roots.get(i) instanceof AggUnaryOp) ? 
						HopRewriteUtils.createScalarIndexing(hnew, 1, i+1) :
						HopRewriteUtils.createMatrixIndexing(hnew, 1, i+1);
					HopRewriteUtils.rewireAllParentChildReferences(roots.get(i), hnewi);
				}
			}
			else if( tmpCNode instanceof CNodeCell && ((CNodeCell)tmpCNode).requiredCastDtm() ) {
				HopRewriteUtils.setOutputParametersForScalar(hnew);
				hnew = HopRewriteUtils.createUnary(hnew, OpOp1.CAST_AS_MATRIX);
			}
			
			if( !(tmpCNode instanceof CNodeMultiAgg) )
				HopRewriteUtils.rewireAllParentChildReferences(hop, hnew);
			memo.add(hnew.getHopID());
		}
		
		//process hops recursively (parent-child links modified)
		for( int i=0; i<hnew.getInput().size(); i++ ) {
			Hop c = hnew.getInput().get(i);
			rConstructModifiedHopDag(c, cplans, clas, memo);
		}
		memo.add(hnew.getHopID());
	}
	
	/**
	 * Cleanup generated cplans in order to remove unnecessary inputs created
	 * during incremental construction. This is important as it avoids unnecessary 
	 * redundant computation. 
	 * 
	 * @param cplans set of cplans
	 */
	private static HashMap<Long, Pair<Hop[],CNodeTpl>> cleanupCPlans(HashMap<Long, Pair<Hop[],CNodeTpl>> cplans) 
	{
		HashMap<Long, Pair<Hop[],CNodeTpl>> cplans2 = new HashMap<Long, Pair<Hop[],CNodeTpl>>();
		CPlanCSERewriter cse = new CPlanCSERewriter();
		
		for( Entry<Long, Pair<Hop[],CNodeTpl>> e : cplans.entrySet() ) {
			CNodeTpl tpl = e.getValue().getValue();
			Hop[] inHops = e.getValue().getKey();
			
			//perform common subexpression elimination
			tpl = cse.eliminateCommonSubexpressions(tpl);
			
			//update input hops (order-preserving)
			HashSet<Long> inputHopIDs = tpl.getInputHopIDs(false);
			inHops = Arrays.stream(inHops)
				.filter(p -> inputHopIDs.contains(p.getHopID()))
				.toArray(Hop[]::new);
			cplans2.put(e.getKey(), new Pair<Hop[],CNodeTpl>(inHops, tpl));
			
			//remove invalid plans with column indexing on main input
			if( tpl instanceof CNodeCell ) {
				CNodeData in1 = (CNodeData)tpl.getInput().get(0);
				if( rHasLookupRC1(tpl.getOutput(), in1) || isLookupRC1(tpl.getOutput(), in1) ) {
					cplans2.remove(e.getKey());
					if( LOG.isTraceEnabled() )
						LOG.trace("Removed cplan due to invalid rc1 indexing on main input.");
				}
			}
			else if( tpl instanceof CNodeMultiAgg ) {
				CNodeData in1 = (CNodeData)tpl.getInput().get(0);
				for( CNode output : ((CNodeMultiAgg)tpl).getOutputs() )
					if( rHasLookupRC1(output, in1) || isLookupRC1(output, in1) ) {
						cplans2.remove(e.getKey());
						if( LOG.isTraceEnabled() )
							LOG.trace("Removed cplan due to invalid rc1 indexing on main input.");
					}
			}
			
			//remove spurious lookups on main input of cell template
			if( tpl instanceof CNodeCell || tpl instanceof CNodeOuterProduct ) {
				CNodeData in1 = (CNodeData)tpl.getInput().get(0);
				rFindAndRemoveLookup(tpl.getOutput(), in1);
			}
			else if( tpl instanceof CNodeMultiAgg ) {
				CNodeData in1 = (CNodeData)tpl.getInput().get(0);
				rFindAndRemoveLookupMultiAgg((CNodeMultiAgg)tpl, in1);
			}
			
			//remove cplan w/ single op and w/o agg
			if( (tpl instanceof CNodeCell && ((CNodeCell)tpl).getCellType()==CellType.NO_AGG
					&& TemplateUtils.hasSingleOperation(tpl) )
				|| (tpl instanceof CNodeRow && (((CNodeRow)tpl).getRowType()==RowType.NO_AGG
					|| ((CNodeRow)tpl).getRowType()==RowType.NO_AGG_B1)
					&& TemplateUtils.hasSingleOperation(tpl))
				|| TemplateUtils.hasNoOperation(tpl) ) 
				cplans2.remove(e.getKey());
				
			//remove cplan if empty
			if( tpl.getOutput() instanceof CNodeData )
				cplans2.remove(e.getKey());
			
			//rename inputs (for codegen and plan caching)
			tpl.renameInputs();
		}
		
		return cplans2;
	}
	
	private static void rFindAndRemoveLookupMultiAgg(CNodeMultiAgg node, CNodeData mainInput) {
		//process all outputs individually
		for( CNode output : node.getOutputs() )
			rFindAndRemoveLookup(output, mainInput);
		
		//handle special case, of lookup being itself the output node
		for( int i=0; i < node.getOutputs().size(); i++) {
			CNode tmp = node.getOutputs().get(i);
			if( TemplateUtils.isLookup(tmp) && tmp.getInput().get(0) instanceof CNodeData
				&& ((CNodeData)tmp.getInput().get(0)).getHopID()==mainInput.getHopID() )
				node.getOutputs().set(i, tmp.getInput().get(0));
		}
	}
	
	private static void rFindAndRemoveLookup(CNode node, CNodeData mainInput) {
		for( int i=0; i<node.getInput().size(); i++ ) {
			CNode tmp = node.getInput().get(i);
			if( TemplateUtils.isLookup(tmp) && tmp.getInput().get(0) instanceof CNodeData
				&& ((CNodeData)tmp.getInput().get(0)).getHopID()==mainInput.getHopID() )
			{
				node.getInput().set(i, tmp.getInput().get(0));
			}
			else
				rFindAndRemoveLookup(tmp, mainInput);
		}
	}
	
	private static boolean rHasLookupRC1(CNode node, CNodeData mainInput) {
		boolean ret = false;
		for( int i=0; i<node.getInput().size() && !ret; i++ ) {
			CNode tmp = node.getInput().get(i);
			if( isLookupRC1(tmp, mainInput) )
				ret = true;
			else
				ret |= rHasLookupRC1(tmp, mainInput);
		}
		return ret;
	}
	
	private static boolean isLookupRC1(CNode node, CNodeData mainInput) {
		return (node instanceof CNodeTernary && ((CNodeTernary)node).getType()==TernaryType.LOOKUP_RC1 
				&& node.getInput().get(0) instanceof CNodeData
				&& ((CNodeData)node.getInput().get(0)).getHopID() == mainInput.getHopID());
	}
	
	/**
	 * This plan cache maps CPlans to compiled and loaded classes in order
	 * to reduce javac and JIT compilation overhead. It uses a simple LRU 
	 * eviction policy if the maximum number of entries is exceeded. In case 
	 * of evictions, this cache also triggers the eviction of corresponding 
	 * class cache entries (1:N). 
	 * <p>
	 * Note: The JVM is free to garbage collect and unload classes that are no
	 * longer referenced.
	 * 
	 */
	private static class PlanCache {
		private final LinkedHashMap<CNode, Class<?>> _plans;
		private final int _maxSize;
		
		public PlanCache(int maxSize) {
			 _plans = new LinkedHashMap<CNode, Class<?>>();	
			 _maxSize = maxSize;
		}
		
		public synchronized Class<?> getPlan(CNode key) {
			//constant time get and maintain usage order
			Class<?> value = _plans.remove(key);
			if( value != null ) 
				_plans.put(key, value); 
			return value;
		}
		
		public synchronized void putPlan(CNode key, Class<?> value) {
			if( _plans.size() >= _maxSize ) {
				//remove least recently used (i.e., first) entry
				Iterator<Entry<CNode, Class<?>>> iter = _plans.entrySet().iterator();
				Class<?> rmCla = iter.next().getValue();
				CodegenUtils.clearClassCache(rmCla); //class cache
				iter.remove(); //plan cache
			}
			_plans.put(key, value);
		}
		
		public synchronized void clear() {
			_plans.clear();
		}
	}
}
