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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.codegen.cplan.CNode;
import org.apache.sysml.hops.codegen.cplan.CNodeCell;
import org.apache.sysml.hops.codegen.cplan.CNodeData;
import org.apache.sysml.hops.codegen.cplan.CNodeOuterProduct;
import org.apache.sysml.hops.codegen.cplan.CNodeTernary;
import org.apache.sysml.hops.codegen.cplan.CNodeTernary.TernaryType;
import org.apache.sysml.hops.codegen.cplan.CNodeTpl;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysml.hops.codegen.template.TemplateBase;
import org.apache.sysml.hops.codegen.template.TemplateBase.CloseType;
import org.apache.sysml.hops.codegen.template.TemplateBase.TemplateType;
import org.apache.sysml.hops.codegen.template.CPlanMemoTable;
import org.apache.sysml.hops.codegen.template.PlanSelection;
import org.apache.sysml.hops.codegen.template.PlanSelectionFuseAll;
import org.apache.sysml.hops.codegen.template.PlanSelectionFuseNoRedundancy;
import org.apache.sysml.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysml.hops.codegen.template.CPlanMemoTable.MemoTableEntrySet;
import org.apache.sysml.hops.codegen.template.TemplateUtils;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.OpOp1;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.hops.rewrite.ProgramRewriteStatus;
import org.apache.sysml.hops.rewrite.ProgramRewriter;
import org.apache.sysml.hops.rewrite.RewriteCommonSubexpressionElimination;
import org.apache.sysml.hops.rewrite.RewriteRemoveUnnecessaryCasts;
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
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.codegen.CodegenUtils;
import org.apache.sysml.runtime.codegen.SpoofCellwise.CellType;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.utils.Explain;
import org.apache.sysml.utils.Statistics;

public class SpoofCompiler 
{
	private static final Log LOG = LogFactory.getLog(SpoofCompiler.class.getName());
	
	//internal configuration flags
	public static boolean LDEBUG = false;
	public static final boolean RECOMPILE_CODEGEN = true;
	public static PlanCache PLAN_CACHE_POLICY = PlanCache.CSLH;
	public static final PlanSelector PLAN_SEL_POLICY = PlanSelector.FUSE_ALL; 
	public static final boolean PRUNE_REDUNDANT_PLANS = true;
	
	public enum PlanSelector {
		FUSE_ALL,             //maximal fusion, possible w/ redundant compute
		FUSE_NO_REDUNDANCY,   //fusion without redundant compute 
		FUSE_COST_BASED,      //cost-based decision on materialization points
	}

	public enum PlanCache {
		CONSTANT, //plan cache, with always compile literals
		CSLH,     //plan cache, with context-sensitive literal replacement heuristic
		NONE;     //no plan cache
		
		public static PlanCache getPolicy(boolean planCache, boolean compileLiterals) {
			return !planCache ? NONE : compileLiterals ? CONSTANT : CSLH;
		}
	}
	
	//plan cache for cplan->compiled source to avoid unnecessary codegen/source code compile
	//for equal operators from (1) different hop dags and (2) repeated recompilation 
	private static ConcurrentHashMap<CNode, Class<?>> planCache = new ConcurrentHashMap<CNode, Class<?>>();
	
	private static ProgramRewriter rewriteCSE = new ProgramRewriter(
			new RewriteCommonSubexpressionElimination(true),
			new RewriteRemoveUnnecessaryCasts());
	
	public static void generateCode(DMLProgram dmlp) 
		throws LanguageException, HopsException, DMLRuntimeException
	{
		// for each namespace, handle function statement blocks
		for (String namespaceKey : dmlp.getNamespaces().keySet()) {
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey,fname);
				generateCodeFromStatementBlock(fsblock);
			}
		}
		
		// handle regular statement blocks in "main" method
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			generateCodeFromStatementBlock(current);
		}
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
			wsb.setPredicateHops(optimize(wsb.getPredicateHops(), true));
			for (StatementBlock sb : wstmt.getBody())
				generateCodeFromStatementBlock(sb);
		}	
		else if (current instanceof IfStatementBlock)
		{
			IfStatementBlock isb = (IfStatementBlock) current;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			isb.setPredicateHops(optimize(isb.getPredicateHops(), true));
			for (StatementBlock sb : istmt.getIfBody())
				generateCodeFromStatementBlock(sb);
			for (StatementBlock sb : istmt.getElseBody())
				generateCodeFromStatementBlock(sb);
		}
		else if (current instanceof ForStatementBlock) //incl parfor
		{
			ForStatementBlock fsb = (ForStatementBlock) current;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			fsb.setFromHops(optimize(fsb.getFromHops(), true));
			fsb.setToHops(optimize(fsb.getToHops(), true));
			fsb.setIncrementHops(optimize(fsb.getIncrementHops(), true));
			for (StatementBlock sb : fstmt.getBody())
				generateCodeFromStatementBlock(sb);
		}
		else //generic (last-level)
		{
			current.set_hops( generateCodeFromHopDAGs(current.get_hops()) );
			current.updateRecompilationFlag();
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
		
		return optimize(new ArrayList<Hop>(Arrays.asList(root)), recompile).get(0);
	}
	
	public static void cleanupCodeGenerator() {
		if( PLAN_CACHE_POLICY != PlanCache.NONE ) {
			CodegenUtils.clearClassCache(); //class cache
			planCache.clear(); //plan cache
		}
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
			boolean compileLiterals = (PLAN_CACHE_POLICY==PlanCache.CONSTANT) || !recompile;
			
			//construct codegen plans
			HashMap<Long, Pair<Hop[],CNodeTpl>>  cplans = constructCPlans(roots, compileLiterals);
			
			//cleanup codegen plans (remove unnecessary inputs, fix hop-cnodedata mapping,
			//remove empty templates with single cnodedata input, remove spurious lookups)
			cplans = cleanupCPlans(cplans);
			
			//explain before modification
			if( LDEBUG && !cplans.isEmpty() ) { //existing cplans
				LOG.info("Codegen EXPLAIN (before optimize): \n"+Explain.explainHops(roots));
			}
			
			//source code generation for all cplans
			HashMap<Long, Pair<Hop[],Class<?>>> clas = new HashMap<Long, Pair<Hop[],Class<?>>>();
			for( Entry<Long, Pair<Hop[],CNodeTpl>> cplan : cplans.entrySet() ) {
				Pair<Hop[],CNodeTpl> tmp = cplan.getValue();
				
				if( PLAN_CACHE_POLICY==PlanCache.NONE || !planCache.containsKey(tmp.getValue()) ) {
					//generate java source code
					String src = tmp.getValue().codegen(false);
					
					//explain debug output cplans or generated source code
					if( LDEBUG || DMLScript.EXPLAIN.isHopsType(recompile) ) {
						LOG.info("Codegen EXPLAIN (generated cplan for HopID: " +  cplan.getKey() +"):");
						LOG.info(tmp.getValue().getClassname()
								+Explain.explainCPlan(cplan.getValue().getValue()));
					}
					if( LDEBUG || DMLScript.EXPLAIN.isRuntimeType(recompile) ) {
						LOG.info("Codegen EXPLAIN (generated code for HopID: " +  cplan.getKey() +"):");
						LOG.info(src);
					}
					
					//compile generated java source code
					Class<?> cla = CodegenUtils.compileClass(tmp.getValue().getClassname(), src);
					planCache.put(tmp.getValue(), cla);
				}
				else if( LDEBUG || DMLScript.STATISTICS ) {
					Statistics.incrementCodegenPlanCacheHits();
				}
				
				Class<?> cla = planCache.get(tmp.getValue());
				if(cla != null)
					clas.put(cplan.getKey(), new Pair<Hop[],Class<?>>(tmp.getKey(),cla));
				
				if( LDEBUG || DMLScript.STATISTICS )
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
				if( LDEBUG ) {
					LOG.info("Codegen EXPLAIN (after optimize): \n"+Explain.explainHops(roots));
				}
			}
		}
		catch( Exception ex ) {
			throw new DMLRuntimeException(ex);
		}
		
		if( DMLScript.STATISTICS ) {
			Statistics.incrementCodegenDAGCompile();
			Statistics.incrementCodegenCompileTime(System.nanoTime()-t0);
		}
			
		return ret;
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
		LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> ret = new LinkedHashMap<Long, Pair<Hop[],CNodeTpl>>();
		Hop.resetVisitStatus(roots);
		for( Hop hop : roots )
			rConstructCPlans(hop, memo, ret, compileLiterals);
		Hop.resetVisitStatus(roots);
		
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
			if( tpl.open(hop) )
				memo.add(hop, tpl.getType());
		
		//fuse and merge operator plans
		for( Hop c : hop.getInput() ) {
			if( memo.contains(c.getHopID()) )
				for( MemoTableEntry me : memo.getDistinct(c.getHopID()) ) {
					TemplateBase tpl = TemplateUtils.createTemplate(me.type, me.closed);
					if( tpl.fuse(hop, c) ) {
						int pos = hop.getInput().indexOf(c);
						MemoTableEntrySet P = new MemoTableEntrySet(tpl.getType(), pos, c.getHopID(), tpl.isClosed());
						for(int k=0; k<hop.getInput().size(); k++)
							if( k != pos ) {
								Hop input2 = hop.getInput().get(k);
								if( memo.contains(input2.getHopID()) && !memo.get(input2.getHopID()).get(0).closed
									&& memo.get(input2.getHopID()).get(0).type == TemplateType.CellTpl && tpl.merge(hop, input2) ) 
									P.crossProduct(k, -1L, input2.getHopID());
								else
									P.crossProduct(k, -1L);
							}
						memo.addAll(hop, P);
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
	
	private static void rConstructCPlans(Hop hop, CPlanMemoTable memo, HashMap<Long, Pair<Hop[],CNodeTpl>> cplans, boolean compileLiterals) 
		throws DMLException
	{		
		//top-down memoization of processed dag nodes
		if( hop.isVisited() )
			return;
		
		//generate cplan for existing memo table entry
		if( memo.containsTopLevel(hop.getHopID()) ) {
			cplans.put(hop.getHopID(), TemplateUtils
				.createTemplate(memo.getBest(hop.getHopID()).type)
				.constructCplan(hop, memo, compileLiterals));
			if( DMLScript.STATISTICS )
				Statistics.incrementCodegenCPlanCompile(1); 
		}
		
		//process childs recursively
		for( Hop c : hop.getInput() )
			rConstructCPlans(c, memo, cplans, compileLiterals);
		hop.setVisited();
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
			else if( tmpCNode instanceof CNodeCell && ((CNodeCell)tmpCNode).requiredCastDtm() ) {
				HopRewriteUtils.setOutputParametersForScalar(hnew);
				hnew = HopRewriteUtils.createUnary(hnew, OpOp1.CAST_AS_MATRIX);
			}
			
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
	private static HashMap<Long, Pair<Hop[],CNodeTpl>> cleanupCPlans(HashMap<Long, Pair<Hop[],CNodeTpl>> cplans) {
		HashMap<Long, Pair<Hop[],CNodeTpl>> cplans2 = new HashMap<Long, Pair<Hop[],CNodeTpl>>();
		for( Entry<Long, Pair<Hop[],CNodeTpl>> e : cplans.entrySet() ) {
			CNodeTpl tpl = e.getValue().getValue();
			Hop[] inHops = e.getValue().getKey();
			
			//collect cplan leaf node names
			HashSet<Long> leafs = new HashSet<Long>();
			rCollectLeafIDs(tpl.getOutput(), leafs);
			
			//create clean cplan w/ minimal inputs
			if( inHops.length == leafs.size() )
				cplans2.put(e.getKey(), e.getValue());
			else {
				tpl.cleanupInputs(leafs);
				ArrayList<Hop> tmp = new ArrayList<Hop>();
				for( Hop hop : inHops ) {
					if( hop!= null && leafs.contains(hop.getHopID()) )
						tmp.add(hop);
				}
				cplans2.put(e.getKey(), new Pair<Hop[],CNodeTpl>(
						tmp.toArray(new Hop[0]),tpl));
			}
			
			//remove spurious lookups on main input of cell template
			if( tpl instanceof CNodeCell || tpl instanceof CNodeOuterProduct ) {
				CNodeData in1 = (CNodeData)tpl.getInput().get(0);
				rFindAndRemoveLookup(tpl.getOutput(), in1);
			}
			
			//remove invalid plans with column indexing on main input
			if( tpl instanceof CNodeCell ) {
				CNodeData in1 = (CNodeData)tpl.getInput().get(0);
				if( rHasLookupRC1(tpl.getOutput(), in1) )
					cplans2.remove(e.getKey());
			}
			
			//remove cplan w/ single op and w/o agg
			if( tpl instanceof CNodeCell && ((CNodeCell)tpl).getCellType()==CellType.NO_AGG
				&& TemplateUtils.hasSingleOperation(tpl) ) 
				cplans2.remove(e.getKey());
				
			//remove cplan if empty
			if( tpl.getOutput() instanceof CNodeData )
				cplans2.remove(e.getKey());
		}
		
		return cplans2;
	}
	
	private static void rCollectLeafIDs(CNode node, HashSet<Long> leafs) {
		//collect leaf variable names
		if( node instanceof CNodeData && !((CNodeData)node).isLiteral() )
			leafs.add(((CNodeData) node).getHopID());
		
		//recursively process cplan
		for( CNode c : node.getInput() )
			rCollectLeafIDs(c, leafs);
	}
	
	private static void rFindAndRemoveLookup(CNode node, CNodeData mainInput) {
		for( int i=0; i<node.getInput().size(); i++ ) {
			CNode tmp = node.getInput().get(i);
			if( tmp instanceof CNodeUnary && (((CNodeUnary)tmp).getType()==UnaryType.LOOKUP_R 
					|| ((CNodeUnary)tmp).getType()==UnaryType.LOOKUP_RC)
				&& tmp.getInput().get(0) instanceof CNodeData
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
			if( tmp instanceof CNodeTernary && ((CNodeTernary)tmp).getType()==TernaryType.LOOKUP_RC1 
				&& tmp.getInput().get(0) instanceof CNodeData
				&& ((CNodeData)tmp.getInput().get(0)).getHopID() == mainInput.getHopID())
				ret = true;
			else
				ret |= rHasLookupRC1(tmp, mainInput);
		}
		return ret;
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
			default:	
				throw new RuntimeException("Unsupported "
					+ "plan selector: "+PLAN_SEL_POLICY);
		}
	}
}
