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
import org.apache.sysml.hops.codegen.cplan.CNodeTpl;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysml.hops.codegen.template.BaseTpl;
import org.apache.sysml.hops.codegen.template.CellTpl;
import org.apache.sysml.hops.codegen.template.CplanRegister;
import org.apache.sysml.hops.codegen.template.OuterProductTpl;
import org.apache.sysml.hops.codegen.template.RowAggTpl;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
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
	
	public static boolean OPTIMIZE = true;
	
	//internal configuration flags
	public static final boolean LDEBUG = false;
	public static final boolean SUM_PRODUCT = false;
	public static final boolean RECOMPILE = true;
	public static boolean USE_PLAN_CACHE = true;
	public static boolean ALWAYS_COMPILE_LITERALS = false;
	public static final boolean ALLOW_SPARK_OPS = false;
	
	//plan cache for cplan->compiled source to avoid unnecessary codegen/source code compile
	//for equal operators from (1) different hop dags and (2) repeated recompilation 
	private static ConcurrentHashMap<CNode, Class<?>> planCache = new ConcurrentHashMap<CNode, Class<?>>();
	
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
		if( USE_PLAN_CACHE ) {
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
	@SuppressWarnings("unused")
	public static ArrayList<Hop> optimize(ArrayList<Hop> roots, boolean recompile) 
		throws DMLRuntimeException 
	{
		if( roots == null || roots.isEmpty() || !OPTIMIZE )
			return roots;
	
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		ArrayList<Hop> ret = roots;
		
		try
		{
			//context-sensitive literal replacement (only integers during recompile)
			boolean compileLiterals = ALWAYS_COMPILE_LITERALS || !recompile;
			
			//construct codegen plans
			HashMap<Long, Pair<Hop[],CNodeTpl>>  cplans = constructCPlans(roots, compileLiterals);
			
			//cleanup codegen plans (remove unnecessary inputs, fix hop-cnodedata mapping,
			//remove empty templates with single cnodedata input, remove spurious lookups)
			cplans = cleanupCPlans(cplans);
					
			//explain before modification
			if( LDEBUG && cplans.size() > 0 ) { //existing cplans
				LOG.info("Codegen EXPLAIN (before optimize): \n"+Explain.explainHops(roots));
			}
			
			//source code generation for all cplans
			HashMap<Long, Pair<Hop[],Class<?>>> clas = new HashMap<Long, Pair<Hop[],Class<?>>>();
			for( Entry<Long, Pair<Hop[],CNodeTpl>> cplan : cplans.entrySet() ) {
				Pair<Hop[],CNodeTpl> tmp = cplan.getValue();
				
				if( !USE_PLAN_CACHE || !planCache.containsKey(tmp.getValue()) ) {
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
			
			//generate final hop dag
			ret = constructModifiedHopDag(roots, cplans, clas);
			
			//explain after modification
			if( LDEBUG && cplans.size() > 0 ) { //existing cplans
				LOG.info("Codegen EXPLAIN (after optimize): \n"+Explain.explainHops(roots));
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
		LinkedHashMap<Long, Pair<Hop[],CNodeTpl>> ret = new LinkedHashMap<Long, Pair<Hop[],CNodeTpl>>();
		for( Hop hop : roots ) {
			CplanRegister perRootCplans = new CplanRegister();
			HashSet<Long> memo = new HashSet<Long>();
			rConstructCPlans(hop, perRootCplans, memo, compileLiterals);
			
			for (Entry<Long, Pair<Hop[],CNodeTpl>> entry : perRootCplans.getTopLevelCplans().entrySet())
				if(!ret.containsKey(entry.getKey()))
					ret.put(entry.getKey(), entry.getValue());
		}
		return ret;
	}
	
	private static void rConstructCPlans(Hop hop, CplanRegister cplanReg, HashSet<Long> memo, boolean compileLiterals) throws DMLException
	{		
		if( memo.contains(hop.getHopID()) )
			return;
		
		//construct template instances
		BaseTpl[] templates = new BaseTpl[]{
				new RowAggTpl(), new CellTpl(), new OuterProductTpl()};
		
		//process hop with all templates
		for( BaseTpl tpl : templates ) {
			if( tpl.openTpl(hop) && tpl.findTplBoundaries(hop,cplanReg) ) {
				cplanReg.insertCpplans(tpl.getType(), 
					tpl.constructTplCplan(compileLiterals));
			}		
		}
		
		//process childs recursively
		memo.add(hop.getHopID());
		for( Hop c : hop.getInput() )
			rConstructCPlans(c, cplanReg, memo, compileLiterals);
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
			for( Hop in : tmpCla.getKey() ) {
				hnew.addInput(in); //add inputs
			}
			hnew.setOutputBlocksizes(hop.getRowsInBlock() , hop.getColsInBlock());
			hnew.setDim1(hop.getDim1());
			hnew.setDim2(hop.getDim2());
			if(tmpCNode instanceof CNodeOuterProduct && ((CNodeOuterProduct)tmpCNode).isTransposeOutput() ) {
				hnew = HopRewriteUtils.createTranspose(hnew);
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
				for( Hop hop : inHops )
					if( leafs.contains(hop.getHopID()) )
						tmp.add(hop);
				cplans2.put(e.getKey(), new Pair<Hop[],CNodeTpl>(
						tmp.toArray(new Hop[0]),tpl));
			}
			
			//remove spurious lookups on main input of cell template
			if( tpl instanceof CNodeCell || tpl instanceof CNodeOuterProduct ) {
				CNode in1 = tpl.getInput().get(0);
				rFindAndRemoveLookup(tpl.getOutput(), in1.getVarname());
			}
			
			//remove cplan w/ single op and w/o agg
			if( tpl instanceof CNodeCell && ((CNodeCell)tpl).getCellType()==CellType.NO_AGG
				&& tpl.getOutput() instanceof CNodeUnary && tpl.getOutput().getInput().get(0) instanceof CNodeData) 
				cplans2.remove(e.getKey());
		
			//remove cplan if empty
			if( tpl.getOutput() instanceof CNodeData )
				cplans2.remove(e.getKey());
		}
		
		return cplans2;
	}
	
	private static void rCollectLeafIDs(CNode node, HashSet<Long> leafs) {
		//collect leaf variable names
		if( node instanceof CNodeData )
			leafs.add(((CNodeData) node).getHopID());
		
		//recursively process cplan
		for( CNode c : node.getInput() )
			rCollectLeafIDs(c, leafs);
	}
	
	private static void rFindAndRemoveLookup(CNode node, String nodeName) {
		for( int i=0; i<node.getInput().size(); i++ ) {
			CNode tmp = node.getInput().get(i);
			if( tmp instanceof CNodeUnary && (((CNodeUnary)tmp).getType()==UnaryType.LOOKUP_R 
					|| ((CNodeUnary)tmp).getType()==UnaryType.LOOKUP_RC)
				&& tmp.getInput().get(0).getVarname().equals(nodeName) )
			{
				node.getInput().set(i, tmp.getInput().get(0));
			}
			else
				rFindAndRemoveLookup(tmp, nodeName);
		}
	}
}
