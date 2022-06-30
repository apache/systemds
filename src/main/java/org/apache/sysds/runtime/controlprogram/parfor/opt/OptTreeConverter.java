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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.MultiThreadedHop;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.ParForStatement;
import org.apache.sysds.parser.ParForStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.IfProgramBlock;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.opt.OptNode.ExecType;
import org.apache.sysds.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import org.apache.sysds.runtime.controlprogram.parfor.opt.OptNode.ParamType;
import org.apache.sysds.runtime.controlprogram.parfor.opt.Optimizer.PlanInputType;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.EvalNaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.cpfile.MatrixIndexingCPFileInstruction;
import org.apache.sysds.runtime.instructions.spark.SPInstruction;

/**
 * Converter for creating an internal plan representation for a given runtime program
 * and to modify/create the runtime program according to the optimized plan.
 * 
 * NOTE: currently only one abstract and one runtime plan at a time.
 * This implies that only one parfor optimization can happen at a time.
 */
public class OptTreeConverter
{
	//internal configuration flags
	public static boolean INCLUDE_FUNCTIONS = true;

	public static OptTree createOptTree( int ck, double cm, PlanInputType type, ParForStatementBlock pfsb, ParForProgramBlock pfpb, ExecutionContext ec )  {
		switch( type ) {
			case ABSTRACT_PLAN: {
				OptTreePlanMappingAbstract hlMap = new OptTreePlanMappingAbstract();
				hlMap.putRootProgram(pfsb.getDMLProg(), pfpb.getProgram());
				Set<String> memo = new HashSet<>();
				OptNode root = rCreateAbstractOptNode(pfsb, pfpb, ec.getVariables(), true, hlMap, memo);
				root.checkAndCleanupRecursiveFunc(new HashSet<String>()); //create consistency between recursive info
				root.checkAndCleanupLeafNodes(); //prune unnecessary nodes
				return new OptTree(ck, cm, type, root, hlMap, null);
			}
			case RUNTIME_PLAN: {
				OptTreePlanMappingRuntime rtMap = new OptTreePlanMappingRuntime();
				OptNode root = rCreateOptNode( pfpb, ec.getVariables(), true, rtMap, true );
				return new OptTree(ck, cm, type, root, null, rtMap);
			}
			default:
				throw new DMLRuntimeException("Optimizer plan input type "+type+" not supported.");
		}
	}

	public static OptTree createAbstractOptTree( int ck, double cm, ParForStatementBlock pfsb,
		ParForProgramBlock pfpb, OptTreePlanMappingAbstract hlMap, Set<String> memo, ExecutionContext ec ) 
	{
		OptTree tree = null;
		OptNode root = null;
		
		try {
			root = rCreateAbstractOptNode( pfsb, pfpb, ec.getVariables(), true, hlMap, memo );
			tree = new OptTree(ck, cm, root);
		}
		catch(HopsException ex) {
			throw new DMLRuntimeException(ex);
		}
		
		return tree;
	}

	public static OptNode rCreateOptNode( ProgramBlock pb, LocalVariableMap vars, boolean topLevel, OptTreePlanMappingRuntime rtMap, boolean storeObjs ) 
	{
		OptNode node = null;
		
		if( pb instanceof IfProgramBlock ) {
			IfProgramBlock ipb = (IfProgramBlock) pb;
			node = new OptNode( NodeType.IF );
			if(storeObjs)
				rtMap.putMapping(ipb, node);
			node.setExecType(ExecType.CP);
			//process if condition
			OptNode ifn = new OptNode(NodeType.GENERIC);
			node.addChilds( createOptNodes( ipb.getPredicate(), vars, rtMap, storeObjs ) );
			node.addChild( ifn );
			for( ProgramBlock lpb : ipb.getChildBlocksIfBody() )
				ifn.addChild( rCreateOptNode(lpb, vars, topLevel, rtMap, storeObjs) );
			//process else condition
			if( ipb.getChildBlocksElseBody() != null && ipb.getChildBlocksElseBody().size()>0 ) {
				OptNode efn = new OptNode(NodeType.GENERIC);
				node.addChild( efn );
				for( ProgramBlock lpb : ipb.getChildBlocksElseBody() )
					efn.addChild( rCreateOptNode(lpb, vars, topLevel, rtMap, storeObjs) );
			}
		}
		else if( pb instanceof WhileProgramBlock ) {
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			node = new OptNode( NodeType.WHILE );
			if(storeObjs)
				rtMap.putMapping(wpb, node);
			node.setExecType(ExecType.CP);
			//process predicate instruction
			node.addChilds( createOptNodes( wpb.getPredicate(), vars, rtMap, storeObjs ) );
			//process body
			for( ProgramBlock lpb : wpb.getChildBlocks() )
				node.addChild( rCreateOptNode(lpb, vars, topLevel, rtMap, storeObjs) );
		}
		else if( pb instanceof ForProgramBlock && !(pb instanceof ParForProgramBlock) ) {
			ForProgramBlock fpb = (ForProgramBlock) pb;
			node = new OptNode( NodeType.FOR );
			if(storeObjs)
				rtMap.putMapping(fpb, node);
			node.setExecType(ExecType.CP);
			
			//determine number of iterations
			long N = OptimizerUtils.getNumIterations(fpb, vars, CostEstimator.FACTOR_NUM_ITERATIONS);
			node.addParam(ParamType.NUM_ITERATIONS, String.valueOf(N));
			
			node.addChilds( createOptNodes( fpb.getFromInstructions(), vars, rtMap, storeObjs ) );
			node.addChilds( createOptNodes( fpb.getToInstructions(), vars, rtMap, storeObjs ) );
			node.addChilds( createOptNodes( fpb.getIncrementInstructions(), vars, rtMap, storeObjs ) );
			
			//process body
			for( ProgramBlock lpb : fpb.getChildBlocks() )
				node.addChild( rCreateOptNode(lpb, vars, topLevel, rtMap, storeObjs) );
		}
		else if( pb instanceof ParForProgramBlock ) {
			ParForProgramBlock fpb = (ParForProgramBlock) pb;
			node = new OptNode( NodeType.PARFOR );
			if(storeObjs)
				rtMap.putMapping(fpb, node);
			node.setK( fpb.getDegreeOfParallelism() );
			long N = fpb.getNumIterations();
			node.addParam(ParamType.NUM_ITERATIONS, (N!=-1) ? 
				String.valueOf(N) : String.valueOf(CostEstimator.FACTOR_NUM_ITERATIONS));
			
			switch(fpb.getExecMode()) {
				case LOCAL:
					node.setExecType(ExecType.CP);
					break;
				case REMOTE_SPARK:
				case REMOTE_SPARK_DP:
					node.setExecType(ExecType.SPARK);
					break;
				default:
					node.setExecType(null);
			}
			
			if( !topLevel ) {
				node.addChilds( createOptNodes( fpb.getFromInstructions(), vars, rtMap, storeObjs ) );
				node.addChilds( createOptNodes( fpb.getToInstructions(), vars, rtMap, storeObjs ) );
				node.addChilds( createOptNodes( fpb.getIncrementInstructions(), vars, rtMap, storeObjs ) );
			}
			
			//process body
			for( ProgramBlock lpb : fpb.getChildBlocks() )
				node.addChild( rCreateOptNode(lpb, vars, false, rtMap, storeObjs) );
			
			//parameters, add required parameters
		}
		else if ( pb instanceof BasicProgramBlock ) {
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			node = new OptNode(NodeType.GENERIC);
			if(storeObjs)
				rtMap.putMapping(pb, node);
			node.addChilds( createOptNodes(bpb.getInstructions(), vars, rtMap, storeObjs) );
			node.setExecType(ExecType.CP);
		}
			
		return node;
	}

	public static ArrayList<OptNode> createOptNodes (ArrayList<Instruction> instset, LocalVariableMap vars, OptTreePlanMappingRuntime rtMap, boolean storeObjs) {
		ArrayList<OptNode> tmp = new ArrayList<>(instset.size());
		for( Instruction inst : instset )
			tmp.add( createOptNode(inst, vars, rtMap, storeObjs) );
		return tmp;
	}

	public static OptNode createOptNode( Instruction inst, LocalVariableMap vars, OptTreePlanMappingRuntime rtMap, boolean storeObjs ) {
		OptNode node = new OptNode(NodeType.INST);
		String instStr = inst.toString();
		String opstr = instStr.split(Instruction.OPERAND_DELIM)[1];
		if(storeObjs)
			rtMap.putMapping(inst, node);
		node.addParam(ParamType.OPSTRING,opstr);
		
		//exec type
		switch( inst.getType() )
		{
			case CONTROL_PROGRAM:
				node.setExecType(ExecType.CP);
				//exec operations
				//CPInstruction cpinst = (CPInstruction) inst;
				//node.addParam(ParamType.OPTYPE,cpinst.getCPInstructionType().toString());
				break;
			default:
				// In initial prototype, parfor is not supported for spark, so this exception will be thrown
				throw new DMLRuntimeException("Unsupported instruction type.");
		}
		
		return node;
	}

	public static OptNode rCreateAbstractOptNode( StatementBlock sb, ProgramBlock pb,
		LocalVariableMap vars, boolean topLevel, OptTreePlanMappingAbstract hlMap, Set<String> memo ) 
	{
		OptNode node = null;
		
		if( pb instanceof IfProgramBlock && sb instanceof IfStatementBlock )
		{
			IfProgramBlock ipb = (IfProgramBlock) pb;
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement is = (IfStatement) isb.getStatement(0);
			
			node = new OptNode( NodeType.IF );
			hlMap.putProgMapping(sb, pb, node);
			node.setExecType(ExecType.CP);
			node.setLineNumbers(isb.getBeginLine(), isb.getEndLine());
			
			//handle predicate
			isb.getPredicateHops().resetVisitStatus();
			node.addChilds( rCreateAbstractOptNodes( isb.getPredicateHops(), vars, hlMap, memo ) );
			
			//process if branch
			OptNode ifn = new OptNode(NodeType.GENERIC);
			hlMap.putProgMapping(sb, pb, ifn);
			ifn.setExecType(ExecType.CP);
			node.addChild( ifn );
			int len = is.getIfBody().size();
			for( int i=0; i<ipb.getChildBlocksIfBody().size() && i<len; i++ )
			{
				ProgramBlock lpb = ipb.getChildBlocksIfBody().get(i);
				StatementBlock lsb = is.getIfBody().get(i);
				ifn.addChild( rCreateAbstractOptNode(lsb, lpb, vars, false, hlMap, memo) );
			}
			//process else branch
			if( ipb.getChildBlocksElseBody() != null ) {
				OptNode efn = new OptNode(NodeType.GENERIC);
				hlMap.putProgMapping(sb, pb, efn);
				efn.setExecType(ExecType.CP);
				node.addChild( efn );
				int len2 = is.getElseBody().size();
				for( int i=0; i<ipb.getChildBlocksElseBody().size() && i<len2; i++ )
				{
					ProgramBlock lpb = ipb.getChildBlocksElseBody().get(i);
					StatementBlock lsb = is.getElseBody().get(i);
					efn.addChild( rCreateAbstractOptNode(lsb, lpb, vars, false, hlMap, memo) );
				}
			}
		}
		else if( pb instanceof WhileProgramBlock && sb instanceof WhileStatementBlock )
		{
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			WhileStatementBlock wsb = (WhileStatementBlock)sb;
			WhileStatement ws = (WhileStatement) wsb.getStatement(0);
			
			node = new OptNode( NodeType.WHILE );
			hlMap.putProgMapping(sb, pb, node);
			node.setExecType(ExecType.CP);
			node.setLineNumbers(wsb.getBeginLine(), wsb.getEndLine());
			
			//handle predicate
			wsb.getPredicateHops().resetVisitStatus();
			node.addChilds( rCreateAbstractOptNodes( wsb.getPredicateHops(), vars, hlMap, memo ) );
			
			//process body
			int len = ws.getBody().size();
			for( int i=0; i<wpb.getChildBlocks().size() && i<len; i++ ) {
				ProgramBlock lpb = wpb.getChildBlocks().get(i);
				StatementBlock lsb = ws.getBody().get(i);
				node.addChild( rCreateAbstractOptNode(lsb, lpb, vars, false, hlMap, memo) );
			}
		}
		else if( pb instanceof ForProgramBlock && sb instanceof ForStatementBlock && !(pb instanceof ParForProgramBlock) )
		{
			ForProgramBlock fpb = (ForProgramBlock) pb;
			ForStatementBlock fsb = (ForStatementBlock)sb;
			ForStatement fs = (ForStatement) fsb.getStatement(0);
			
			node = new OptNode( NodeType.FOR );
			hlMap.putProgMapping(sb, pb, node);
			node.setExecType(ExecType.CP);
			node.setLineNumbers(fsb.getBeginLine(), fsb.getEndLine());
			
			//determine number of iterations
			long N = OptimizerUtils.getNumIterations(fpb, vars, CostEstimator.FACTOR_NUM_ITERATIONS);
			node.addParam(ParamType.NUM_ITERATIONS, String.valueOf(N));
			
			//handle predicate
			fsb.getFromHops().resetVisitStatus();
			fsb.getToHops().resetVisitStatus();
			if( fsb.getIncrementHops()!=null )
				fsb.getIncrementHops().resetVisitStatus();
			node.addChilds( rCreateAbstractOptNodes( fsb.getFromHops(), vars, hlMap, memo ) );
			node.addChilds( rCreateAbstractOptNodes( fsb.getToHops(), vars, hlMap, memo ) );
			if( fsb.getIncrementHops()!=null )
				node.addChilds( rCreateAbstractOptNodes( fsb.getIncrementHops(), vars, hlMap, memo ) );
			
			//process body
			int len = fs.getBody().size();
			for( int i=0; i<fpb.getChildBlocks().size() && i<len; i++ ) {
				ProgramBlock lpb = fpb.getChildBlocks().get(i);
				StatementBlock lsb = fs.getBody().get(i);
				node.addChild( rCreateAbstractOptNode(lsb, lpb, vars, false, hlMap, memo) );
			}
		}
		else if( pb instanceof ParForProgramBlock && sb instanceof ParForStatementBlock )
		{
			ParForProgramBlock fpb = (ParForProgramBlock) pb;
			ParForStatementBlock fsb = (ParForStatementBlock)sb;
			ParForStatement fs = (ParForStatement) fsb.getStatement(0);
			node = new OptNode( NodeType.PARFOR );
			node.setLineNumbers(fsb.getBeginLine(), fsb.getEndLine());
			hlMap.putProgMapping(sb, pb, node);
			node.setK( fpb.getDegreeOfParallelism() );
			long N = fpb.getNumIterations();
			node.addParam(ParamType.NUM_ITERATIONS, (N!=-1) ? String.valueOf(N) :
				String.valueOf(CostEstimator.FACTOR_NUM_ITERATIONS));
			
			switch(fpb.getExecMode()) {
				case LOCAL:
					node.setExecType(ExecType.CP);
					break;
				case REMOTE_SPARK:
				case REMOTE_SPARK_DP:
					node.setExecType(ExecType.SPARK);
					break;		
				case UNSPECIFIED:
					node.setExecType(null);
			}
			
			if( !topLevel ) {
				fsb.getFromHops().resetVisitStatus();
				fsb.getToHops().resetVisitStatus();
				if( fsb.getIncrementHops()!=null )
					fsb.getIncrementHops().resetVisitStatus();
				node.addChilds( rCreateAbstractOptNodes( fsb.getFromHops(), vars, hlMap, memo ) );
				node.addChilds( rCreateAbstractOptNodes( fsb.getToHops(), vars, hlMap, memo ) );
				if( fsb.getIncrementHops()!=null )
					node.addChilds( rCreateAbstractOptNodes( fsb.getIncrementHops(), vars, hlMap, memo ) );
			}
			
			//process body
			int len = fs.getBody().size();
			for( int i=0; i<fpb.getChildBlocks().size() && i<len; i++ ) {
				ProgramBlock lpb = fpb.getChildBlocks().get(i);
				StatementBlock lsb = fs.getBody().get(i);
				node.addChild( rCreateAbstractOptNode(lsb, lpb, vars, false, hlMap, memo) );
			}
			
			//parameters, add required parameters
			Map<String,String> lparams = fpb.getParForParams();
			node.addParam(ParamType.DATA_PARTITIONER, lparams.get(ParForStatementBlock.DATA_PARTITIONER));
			node.addParam(ParamType.TASK_PARTITIONER, lparams.get(ParForStatementBlock.TASK_PARTITIONER));
			node.addParam(ParamType.RESULT_MERGE, lparams.get(ParForStatementBlock.RESULT_MERGE));
			//TODO task size
		}
		else if( pb instanceof BasicProgramBlock ) {
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			sb = pb.getStatementBlock();
			
			//process all hops
			node = new OptNode(NodeType.GENERIC);
			hlMap.putProgMapping(sb, pb, node);
			node.addChilds( createAbstractOptNodes(sb.getHops(), vars, hlMap, memo) );
			node.setExecType(ExecType.CP);
			node.setLineNumbers(sb.getBeginLine(), sb.getEndLine());
			
			//TODO remove this workaround once this information can be obtained from hops/lops compiler
			if( node.isCPOnly() ) {
				boolean isSparkExec = OptimizerUtils.isSparkExecutionMode();
				if( isSparkExec && containsSparkInstruction(bpb, false))
					node.setExecType(ExecType.SPARK);
			}
		}
		
		//final cleanup
		node.checkAndCleanupLeafNodes(); //NOTE: required because this function is also used to create subtrees
		
		return node;
	}

	public static ArrayList<OptNode> createAbstractOptNodes(ArrayList<Hop> hops, LocalVariableMap vars, OptTreePlanMappingAbstract hlMap, Set<String> memo ) {
		ArrayList<OptNode> ret = new ArrayList<>(); 
		
		//reset all hops
		Hop.resetVisitStatus(hops);
		
		//created and add actual opt nodes
		if( hops != null )
			for( Hop hop : hops )
				ret.addAll(rCreateAbstractOptNodes(hop, vars, hlMap, memo));
		
		return ret;
	}

	public static ArrayList<OptNode> rCreateAbstractOptNodes(Hop hop, LocalVariableMap vars, OptTreePlanMappingAbstract hlMap, Set<String> memo) {
		ArrayList<OptNode> ret = new ArrayList<>(); 
		ArrayList<Hop> in = hop.getInput();
	
		if( hop.isVisited() )
			return ret;
		
		//general case
		if( !(hop instanceof DataOp || hop instanceof LiteralOp || hop instanceof FunctionOp) )
		{
			OptNode node = new OptNode(NodeType.HOP);
			String opstr = hop.getOpString();
			node.addParam(ParamType.OPSTRING,opstr);
			
			//handle execution type
			Types.ExecType et = (hop.getExecType()!=null) ? 
					   hop.getExecType() : Types.ExecType.CP;
			switch( et ) {
				case CP:case GPU:
					node.setExecType(ExecType.CP); break;
				case SPARK:
					node.setExecType(ExecType.SPARK); break;
				// TODO: create execution mode for parfor loop
				case FED:
					node.setExecType(ExecType.CP); break;
				default:
					throw new DMLRuntimeException("Unsupported optnode exec type: "+et);
			}
			
			//handle degree of parallelism
			if( et == Types.ExecType.CP && hop instanceof MultiThreadedHop ){
				MultiThreadedHop mtop = (MultiThreadedHop) hop;
				node.setK( OptimizerUtils.getConstrainedNumThreads(mtop.getMaxNumThreads()) );
			}
			
			//assign node to return
			hlMap.putHopMapping(hop, node);
			ret.add(node);
		}
		//process function calls
		else if (hop instanceof FunctionOp && INCLUDE_FUNCTIONS )
		{
			FunctionOp fhop = (FunctionOp) hop;
			String fname = fhop.getFunctionName();
			String fnspace = fhop.getFunctionNamespace();
			String fKey = fhop.getFunctionKey();
			Object[] prog = hlMap.getRootProgram();

			OptNode node = new OptNode(NodeType.FUNCCALL);
			hlMap.putHopMapping(fhop, node); 
			node.setExecType(ExecType.CP);
			node.addParam(ParamType.OPSTRING, fKey);
			
			if( !fnspace.equals(DMLProgram.INTERNAL_NAMESPACE) )
			{
				FunctionProgramBlock fpb = ((Program)prog[1]).getFunctionProgramBlock(fnspace, fname);
				FunctionStatementBlock fsb = ((DMLProgram)prog[0]).getFunctionStatementBlock(fnspace, fname);
				FunctionStatement fs = (FunctionStatement) fsb.getStatement(0);
				
				//process body; NOTE: memo prevents inclusion of functions multiple times
				if( !memo.contains(fKey) )
				{
					memo.add(fKey); 
					int len = fs.getBody().size();
					for( int i=0; i<fpb.getChildBlocks().size() && i<len; i++ ) {
						ProgramBlock lpb = fpb.getChildBlocks().get(i);
						StatementBlock lsb = fs.getBody().get(i);
						node.addChild( rCreateAbstractOptNode(lsb, lpb, vars, false, hlMap, memo) );
					}
					memo.remove(fKey);
				}
				else
					node.addParam(ParamType.RECURSIVE_CALL, "true");
			}
			
			ret.add(node);
		}
		
		if( in != null )
			for( Hop hin : in ) 
				if( !(hin instanceof DataOp || hin instanceof LiteralOp ) ) //no need for opt nodes
					ret.addAll(rCreateAbstractOptNodes(hin, vars, hlMap, memo));

		hop.setVisited();
		
		return ret;
	}

	public static boolean rContainsSparkInstruction( ProgramBlock pb, boolean inclFunctions )
	{
		boolean ret = false;
		
		if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock tmp = (WhileProgramBlock)pb;
			ret = containsSparkInstruction(tmp.getPredicate(), true);
			if( ret ) return ret;
			for (ProgramBlock pb2 : tmp.getChildBlocks()) {
				ret = rContainsSparkInstruction(pb2, inclFunctions);
				if( ret ) return ret;
			}
		}
		else if (pb instanceof IfProgramBlock) {
			IfProgramBlock tmp = (IfProgramBlock)pb;
			ret = containsSparkInstruction(tmp.getPredicate(), true);
			if( ret ) return ret;
			for( ProgramBlock pb2 : tmp.getChildBlocksIfBody() ){
				ret = rContainsSparkInstruction(pb2, inclFunctions);
				if( ret ) return ret;
			}
			for( ProgramBlock pb2 : tmp.getChildBlocksElseBody() ){
				ret = rContainsSparkInstruction(pb2, inclFunctions);
				if( ret ) return ret;
			}
		}
		else if (pb instanceof ForProgramBlock) { //includes ParFORProgramBlock
			ForProgramBlock tmp = (ForProgramBlock)pb;
			ret = containsSparkInstruction(tmp.getFromInstructions(), true);
			ret |= containsSparkInstruction(tmp.getToInstructions(), true);
			ret |= containsSparkInstruction(tmp.getIncrementInstructions(), true);
			if( ret ) return ret;
			for( ProgramBlock pb2 : tmp.getChildBlocks() ){
				ret = rContainsSparkInstruction(pb2, inclFunctions);
				if( ret ) return ret;
			}
		}
		else if (  pb instanceof FunctionProgramBlock ) {
			//do nothing
		}
		else if( pb instanceof BasicProgramBlock ) {
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			ret =   containsSparkInstruction(bpb, true)
				|| (inclFunctions && containsFunctionCallInstruction(bpb));
		}

		return ret;
	}

	public static boolean containsSparkInstruction( BasicProgramBlock pb, boolean inclCPFile ) {
		return containsSparkInstruction(pb.getInstructions(), inclCPFile);
	}

	public static boolean containsSparkInstruction( ArrayList<Instruction> instSet, boolean inclCPFile ) {
		return instSet.stream().anyMatch(inst -> inst instanceof SPInstruction
			|| (inclCPFile && inst instanceof MatrixIndexingCPFileInstruction));
	}

	public static boolean containsFunctionCallInstruction( BasicProgramBlock pb ) {
		return pb.getInstructions().stream()
			.anyMatch(inst -> inst instanceof FunctionCallCPInstruction
					|| inst instanceof EvalNaryCPInstruction);
	}

	public static void replaceProgramBlock(OptNode parent, OptNode n, ProgramBlock pbOld, ProgramBlock pbNew, OptTreePlanMappingAbstract hlMap) {
		ProgramBlock pbParent = null;
		if( parent.getNodeType()==NodeType.FUNCCALL ) {
			FunctionOp fop = (FunctionOp) hlMap.getMappedHop(parent.getID());
			pbParent = ((Program)hlMap.getRootProgram()[1]).getFunctionProgramBlock(fop.getFunctionNamespace(), fop.getFunctionName());
		}
		else
			pbParent = (ProgramBlock) hlMap.getMappedProg( parent.getID() )[1];
		
		if( pbParent instanceof IfProgramBlock ) {
			IfProgramBlock ipb = (IfProgramBlock) pbParent;
			replaceProgramBlock( ipb.getChildBlocksIfBody(), pbOld, pbNew );
			replaceProgramBlock( ipb.getChildBlocksElseBody(), pbOld, pbNew );
		}
		else if( pbParent instanceof WhileProgramBlock ) {
			WhileProgramBlock wpb = (WhileProgramBlock) pbParent;
			replaceProgramBlock( wpb.getChildBlocks(), pbOld, pbNew );
		}
		else if( pbParent instanceof ForProgramBlock || pbParent instanceof ParForProgramBlock ) {
			ForProgramBlock fpb = (ForProgramBlock) pbParent;
			replaceProgramBlock( fpb.getChildBlocks(), pbOld, pbNew );
		}
		else if( pbParent instanceof FunctionProgramBlock ) {
			FunctionProgramBlock fpb = (FunctionProgramBlock) pbParent;
			replaceProgramBlock( fpb.getChildBlocks(), pbOld, pbNew );
		}
		else
			throw new DMLRuntimeException("Optimizer doesn't support "+pbParent.getClass().getName());
		
		//update repository
		hlMap.replaceMapping(pbNew, n);
	}
	
	public static void replaceProgramBlock(OptNode parent, OptNode n, ProgramBlock pbOld, ProgramBlock pbNew, OptTreePlanMappingRuntime rtMap) {
		ProgramBlock pbParent = null;
		pbParent = (ProgramBlock) rtMap.getMappedObject( parent.getID() );
		
		if( pbParent instanceof IfProgramBlock ) {
			IfProgramBlock ipb = (IfProgramBlock) pbParent;
			replaceProgramBlock( ipb.getChildBlocksIfBody(), pbOld, pbNew );
			replaceProgramBlock( ipb.getChildBlocksElseBody(), pbOld, pbNew );
		}
		else if( pbParent instanceof WhileProgramBlock ) {
			WhileProgramBlock wpb = (WhileProgramBlock) pbParent;
			replaceProgramBlock( wpb.getChildBlocks(), pbOld, pbNew );
		}
		else if( pbParent instanceof ForProgramBlock || pbParent instanceof ParForProgramBlock ) {
			ForProgramBlock fpb = (ForProgramBlock) pbParent;
			replaceProgramBlock( fpb.getChildBlocks(), pbOld, pbNew );
		}
		else if( pbParent instanceof FunctionProgramBlock ) {
			FunctionProgramBlock fpb = (FunctionProgramBlock) pbParent;
			replaceProgramBlock( fpb.getChildBlocks(), pbOld, pbNew );
		}
		else
			throw new DMLRuntimeException("Optimizer doesn't support "+pbParent.getClass().getName());
		
		//update repository
		rtMap.replaceMapping(pbNew, n);
	}

	public static void replaceProgramBlock(List<ProgramBlock> pbs, ProgramBlock pbOld, ProgramBlock pbNew) {
		int len = pbs.size();
		for( int i=0; i<len; i++ )
			if( pbs.get(i) == pbOld )
				pbs.set(i, pbNew);
	}
}
