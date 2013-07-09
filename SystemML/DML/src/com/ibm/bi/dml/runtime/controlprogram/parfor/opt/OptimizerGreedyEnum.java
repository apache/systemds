package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.Collection;
import java.util.HashSet;

import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.POptMode;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ParamType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;

/**
 * 
 * NOTE: Due to common subexpresion elimination, many to many relationship between hops and instructions,
 * we need a "generate and evaluate" approach for costing potential rewrites.
 */

class OptimizerGreedyEnum extends Optimizer
{
	public static final int MAX_ITERATIONS = 10;
	

	@Override
	public CostModelType getCostModelType() 
	{
		return CostModelType.RUNTIME_METRICS;
	}


	@Override
	public PlanInputType getPlanInputType() 
	{
		return PlanInputType.RUNTIME_PLAN;
	}
	

	@Override
	public POptMode getOptMode() 
	{
		return POptMode.GREEDY;
	}
	
	@Override
	public boolean optimize(ParForStatementBlock sb, ParForProgramBlock pb, OptTree pPlan, CostEstimator est, ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		System.out.println("--- GREEDY OPTIMIZER -------");
		
		//preparations
		boolean change = false;
		OptTree lPlan = OptTreeConverter.createAbstractOptTree(pPlan.getCK(), pPlan.getCM(), sb, pb, new HashSet<String>(), ec);
		OptNode lRoot = lPlan.getRoot();
		OptNode pRoot = pPlan.getRoot();
		double T = est.getEstimate(TestMeasure.EXEC_TIME, pRoot);
		double M = est.getEstimate(TestMeasure.MEMORY_USAGE, pRoot);
		
		//TODO optimize reuse of enumerated/computed intermediates
		//     (as soon as all rewrites are successfully integrated)
		
		//TODO compute local time bound (early pruning) similar to compute local par/mem bound
		
		///////
		// begin core opt algorithm
		///////
		boolean converged = false; //true, if no further improvement
		int i = 0; //count of processed iterations
		while( !converged && i < MAX_ITERATIONS )
		{
			System.out.println("--- GREEDY OPTIMIZER: iteration "+i+": T="+T/1000+"s, M="+M/(1024*1024)+"MB");
			
			MemoTable memo = new MemoTable();
			
			Collection<OptNode> nodes = lRoot.getRelevantNodeList();
			for( OptNode n : nodes ) //for each unfixed node in tree
			{
				boolean topLevel = (n == lRoot);
				// compute local bounds (per node)
				double lck = est.computeLocalParBound(lPlan, n);
				double lcm = est.computeLocalMemoryBound(lPlan, n);
				
				System.out.println("lck="+lck);
				System.out.println("lcm="+lcm);
				
				// enum plans per node, wrt given par constraint
				Collection<OptNode> C = enumPlans(n, lck);
				
				// create and eval plans
				for( OptNode ac : C )
				{
					//System.out.println("Create and eval plan for node: "+ac.getID());
					ProgramBlock cPB = ProgramRecompiler.recompile( ac ); 
					OptNode rc = OptTreeConverter.rCreateOptNode(cPB, ec.getVariables(), topLevel, false); 

					double cM = est.getEstimate(TestMeasure.MEMORY_USAGE, rc);
					System.out.println("cM="+cM);
					if( cM <= lcm ) //check memory constraint
					{
						double cT = est.getEstimate(TestMeasure.EXEC_TIME, rc);
						System.out.println("cT="+cT);
						memo.putMemoTableEntry(ac.getID(), new MemoTableEntry(ac.getID(),rc, cPB, cM, cT), true);
					}
				}
			}
			
			// greedy selection of best node change
			if( memo.hasCandidates() )
			{
				Collection<MemoTableEntry> C = memo.getCandidates();
				double minT = T; //init as time of current plan 
				MemoTableEntry minC = null;
				for( MemoTableEntry c : C )
				{
					System.out.println("Local plan>\n"+c.getRtOptNode().explain(0, false));
					OptNode tmpPNode = OptTreeConverter.exchangeTemporary( pRoot, c.getID(), c.getRtOptNode() );
					System.out.println("After exchange>\n"+tmpPNode.explain(0, false));
					double tmpT = est.getEstimate(TestMeasure.EXEC_TIME, tmpPNode);
					
					System.out.println("tmpT="+tmpT+", minT="+minT);
					if( tmpT < minT )
					{
						minT = tmpT;
						minC = c;
					}
					OptTreeConverter.revertTemporaryChange( c.getID() );
				}
				if( minC != null ) //found candidate that improve overall exec time
				{
					ProgramRecompiler.exchangeProgram( minC.getID(), minC.getRtProgramBlock() );
					pRoot = OptTreeConverter.exchangePermanently( pRoot, minC.getID(), minC.getRtOptNode() ); 
					T = minT;
					M = est.getEstimate(TestMeasure.MEMORY_USAGE, pRoot);
					change = true;
				}
				else
				{
					System.out.println("Warning: Unsuccessful convergence (no more global candiate despite existing local candidates).");
					converged = true;
				}
				i++; //iteration cnt
			}
			else
			{
				System.out.println("Successful convergence.");
				converged = true;
			}
		}
		///////
		// end core opt algorithm
		///////
		
		//TODO invoke heuristic optimizer at the end for final polishing and reuse of rewrites?
		//apply some simple rewrites: e.g., remove nested parfor (if not needed)
		
		return change;
	}

	
	public boolean test(ParForStatementBlock sb, ParForProgramBlock pb, OptTree plan) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//OptNode lroot = plan.getRoot();

		// Dummy execution context
		// TODO: fix this!
		ExecutionContext ec = new ExecutionContext();
		
		OptTree absPlan = OptTreeConverter.createAbstractOptTree(plan.getCK(), plan.getCM(), sb, pb, new HashSet<String>(), ec);
		OptNode absRoot = absPlan.getRoot();
		
		Collection<OptNode> nodes = absRoot.getNodeList();
		for( OptNode n : nodes )
			if( n.getNodeType()==NodeType.HOP )
			{
				System.out.println(n.getParam(ParamType.OPSTRING)); 
				if(n.getInstructionName().contains("a(+*)") ) //"rand"
				{
					n.setExecType(ExecType.CP);
					ProgramRecompiler.recompilePartialPlan( n );
				}
			}

		return true;
		
	}

}
