/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn.ropt;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.cost.CostEstimationWrapper;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.compile.Recompiler;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptTreeConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.yarn.DMLYarnClient;
import com.ibm.bi.dml.yarn.ropt.YarnOptimizerUtils.GridEnumType;

/**
 * TODO parallel version with exposed numThreads parameter
 * TODO change all memory units to long byte
 * 
 */
public class ResourceOptimizer 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final Log LOG = LogFactory.getLog(ResourceOptimizer.class);
	
	/**
	 * 
	 * @param prog
	 * @param cc
	 * @param cptype
	 * @param mrtype
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static ResourceConfig optimizeResourceConfig( ArrayList<ProgramBlock> prog, YarnClusterConfig cc, GridEnumType cptype, GridEnumType mrtype ) 
		throws DMLRuntimeException
	{
		ResourceConfig ROpt = null;
		
		try
		{
			//enumerate grid points for given types (refers to jvm max heap)
			ArrayList<Double> SRc = enumerateGridPoints(prog, cc, cptype);
			ArrayList<Double> SRm = enumerateGridPoints(prog, cc, mrtype);
			double min = YarnOptimizerUtils.toB(cc.getMinAllocationMB()) / DMLYarnClient.MEM_FACTOR;
			
			//init resource config and global costs
			ROpt = new ResourceConfig(prog, min);
			double costOpt = Double.MAX_VALUE;
			
			for( Double rc : SRc ) //enumerate CP memory rc
			{
				//baseline compile and pruning
				ArrayList<ProgramBlock> B = compileProgram(prog, null, rc, min); //unrolled Bp
				ArrayList<ProgramBlock> Bp = pruneProgramBlocks( B );
				
				//init local memo table [resource, cost]
				double[][] memo = initLocalMemoTable( Bp, min );
				
				for( int i=0; i<Bp.size(); i++ ) //for all relevant blocks
				{
					ProgramBlock pb = Bp.get(i);
					
					for( Double rm : SRm ) //for each MR memory 
					{
						//recompile program block 
						recompileProgramBlock(pb, rc, rm);
						
						//local costing and memo table maintenance (cost entire program to account for 
						//in-memory status of variables and loops)
						double lcost = CostEstimationWrapper.getTimeEstimate(pb.getProgram(), new ExecutionContext());
						if( lcost < memo[i][1] ) { //accept new local opt
							memo[i][0] = rm;
							memo[i][1] = lcost;
 						}
					}
				}			
				
				//global costing 
				double[][] gmemo = initGlobalMemoTable(B, Bp, memo, min);
				recompileProgramBlocks(B, rc, gmemo);
				double gcost = CostEstimationWrapper.getTimeEstimate(B.get(0).getProgram(), new ExecutionContext());
				if( gcost <= costOpt ){ //accept new global opt
					ROpt.setCPResource(rc.longValue());
					ROpt.setMRResources(B, gmemo);
					costOpt = gcost;
				}
			}	
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		return ROpt;
	}
	
	
	/**
	 * 
	 * @param prog
	 * @param cc
	 * @param type
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws HopsException 
	 */
	private static ArrayList<Double> enumerateGridPoints( ArrayList<ProgramBlock> prog, YarnClusterConfig cc, GridEnumType type ) 
		throws DMLRuntimeException, HopsException
	{
		//compute effective memory
		double min = YarnOptimizerUtils.toB(cc.getMinAllocationMB()) / DMLYarnClient.MEM_FACTOR;
		double max = YarnOptimizerUtils.toB(cc.getMaxAllocationMB()) / DMLYarnClient.MEM_FACTOR;
		
		//create enumerator
		GridEnumeration ge = null;
		switch( type ){
			case EQUI_GRID:
				ge = new GridEnumerationEqui(prog, min, max); break;
			case EXP_GRID:
				ge = new GridEnumerationExp(prog, min, max); break;
			case HYBRID_MEM_EQUI_GRID:
				ge = new GridEnumerationHybrid(prog, min, max); break;
		}
		
		//generate points 
		return ge.enumerateGridPoints();
	}
	
	/**
	 * 
	 * @param prog
	 * @param B
	 * @return
	 * @throws IOException 
	 * @throws DMLUnsupportedOperationException 
	 * @throws LopsException 
	 * @throws HopsException 
	 * @throws DMLRuntimeException 
	 */
	private static ArrayList<ProgramBlock> compileProgram( ArrayList<ProgramBlock> prog, ArrayList<ProgramBlock> B, double cp, double mr ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		if( B == null ) //init 
		{
			B = new ArrayList<ProgramBlock>();
			
			InfrastructureAnalyzer.setLocalMaxMemory( (long)cp );
			InfrastructureAnalyzer.setRemoteMaxMemoryMap( (long)mr );
			InfrastructureAnalyzer.setRemoteMaxMemoryReduce( (long)mr );	
		}
		
		for( ProgramBlock pb : prog )
			compileProgram( pb, B, cp, mr );
		
		return B;
	}
	
	/**
	 * 
	 * @param pb
	 * @param Bp
	 * @return
	 * @throws IOException 
	 * @throws DMLUnsupportedOperationException 
	 * @throws LopsException 
	 * @throws HopsException 
	 * @throws DMLRuntimeException 
	 */
	private static ArrayList<ProgramBlock> compileProgram( ProgramBlock pb, ArrayList<ProgramBlock> B, double cp, double mr ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		if (pb instanceof FunctionProgramBlock)
		{
			FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
			compileProgram(fpb.getChildBlocks(), B, cp, mr);
		}
		else if (pb instanceof WhileProgramBlock)
		{
			//TODO while predicate 
			WhileProgramBlock fpb = (WhileProgramBlock)pb;
			compileProgram(fpb.getChildBlocks(), B, cp, mr);
		}	
		else if (pb instanceof IfProgramBlock)
		{
			//TODO if predicate 
			IfProgramBlock fpb = (IfProgramBlock)pb;
			compileProgram(fpb.getChildBlocksIfBody(), B, cp, mr);
			compileProgram(fpb.getChildBlocksElseBody(), B, cp, mr);
		}
		else if (pb instanceof ForProgramBlock) //incl parfor
		{
			//TODO while predicate 
			ForProgramBlock fpb = (ForProgramBlock)pb;
			compileProgram(fpb.getChildBlocks(), B, cp, mr);
		}
		else
		{
			StatementBlock sb = pb.getStatementBlock();
			ArrayList<Instruction> inst = Recompiler.recompileHopsDag(sb, sb.get_hops(), 
					                                   new LocalVariableMap(), false, 0);
			pb.setInstructions( inst );
			B.add(pb);
		}
		
		return B;
	}
	
	
	/**
	 * 
	 * @param pbs
	 * @param cp
	 * @param memo
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	private static void recompileProgramBlocks( ArrayList<ProgramBlock> pbs, double cp, double[][] memo ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		for( int i=0; i<pbs.size(); i++ )
		{
			ProgramBlock pb = pbs.get(i);
			double mr = memo[i][0];
			recompileProgramBlock(pb, cp, mr);
		}
	}
	
	/**
	 * 
	 * @param pb
	 * @param cp
	 * @param mr
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	private static void recompileProgramBlock( ProgramBlock pb, double cp, double mr ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		//init compiler memory budget
		InfrastructureAnalyzer.setRemoteMaxMemoryMap( (long)mr );
		InfrastructureAnalyzer.setRemoteMaxMemoryReduce( (long)mr );
		
		//recompile instructions
		StatementBlock sb = pb.getStatementBlock();
		ArrayList<Instruction> inst = Recompiler.recompileHopsDag(sb, sb.get_hops(), 
				                                   new LocalVariableMap(), false, 0);
		pb.setInstructions( inst );
	}
	
	/**
	 * 
	 * @param B
	 * @return
	 */
	private static ArrayList<ProgramBlock> pruneProgramBlocks( ArrayList<ProgramBlock> B )
	{
		ArrayList<ProgramBlock> Bp = new ArrayList<ProgramBlock>();
		for( ProgramBlock pb : B )
			if( OptTreeConverter.containsMRJobInstruction(pb.getInstructions(), false) ){
				Bp.add( pb );
			}
		
		return Bp;		
	}
	
	/**
	 * 
	 * @param Bp
	 * @param min
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	private static double[][] initLocalMemoTable( ArrayList<ProgramBlock> Bp, double min ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//allocate memo structure
		int len = Bp.size();
		double[][] memo = new double[len][2];
		
		//init with min resource and current costs
		for( int i=0; i<len; i++ )
		{
			ProgramBlock pb = Bp.get(i);
			memo[i][0] = min;
			memo[i][1] = CostEstimationWrapper.getTimeEstimate(pb.getProgram(), new ExecutionContext());
		}
		
		return memo;
	}
	
	/**
	 * 
	 * @param B
	 * @param Bp
	 * @param lmemo
	 * @param min
	 * @return
	 */
	private static double[][] initGlobalMemoTable( ArrayList<ProgramBlock> B, ArrayList<ProgramBlock> Bp, double[][] lmemo, double min )
	{
		//allocate memo structure
		int len = B.size();
		int lenp = Bp.size(); //lenp<=len
		double[][] memo = new double[len][2];
		
		//init with min resource and current costs
		int j = 0;
		for( int i=0; i<len && j<lenp; i++ )
		{
			ProgramBlock pb = B.get(i);
			if( pb != Bp.get(j) ){
				memo[i][0] = min;
				memo[i][1] = -1;
				j++; continue;
			}
			
			memo[i][0] = lmemo[j][0];
			memo[i][1] = -1;
		}
		
		return memo;
	
	}
	
	
	////////
	// old code
	
	public static long jvmToPhy(long jvm, boolean mrRealRun) {
		long ret = (long) Math.ceil((double)jvm * DMLYarnClient.MEM_FACTOR);
		if (mrRealRun) {
			long lowerBound = (long)YarnClusterAnalyzer.getMinMRContarinerPhyMB() * 1024 * 1024;
			if (ret < lowerBound)
				return lowerBound;
		}
		return ret;
	}
	
	public static long budgetToJvm(double budget) {
		return (long) Math.ceil(budget / OptimizerUtils.MEM_UTIL_FACTOR);
	}
	

	public static double phyToBudget(long physical) throws IOException {
		return (double)physical / DMLYarnClient.MEM_FACTOR * OptimizerUtils.MEM_UTIL_FACTOR;
	}
	
	/*

	public static final double MINIMAL_CP_BUDGET_ALLOWED = 512 * 1024 * 1024; //TODO get min allocation constraints
	public static final double MINIMAL_MR_BUDGET_ALLOWED = 512 * 1024 * 1024;
	
	
	public static double jvmToBudget(long jvm) {
		return (double)jvm * OptimizerUtils.MEM_UTIL_FACTOR;
	}
	
	
	public static int budgetToPhysicalMB(double budget) {
		return (int) Math.ceil(budget / OptimizerUtils.MEM_UTIL_FACTOR * DMLYarnClient.MEM_FACTOR / 1024 / 1024);
	}
	
	public static int budgetToJvmMB(double budget) {
		return (int) Math.ceil(budget / OptimizerUtils.MEM_UTIL_FACTOR / 1024 / 1024);
	}
	
	public static int byteToMB(long byteSize) {
		return (int) Math.ceil(byteSize / 1024 / 1024);
	}
	
	
	public static class DMLMultiTimeCompiler {
		public YarnConfiguration conf;
		public String[] dmlScriptArgs = null;
		
		public int compileCount;
		
		public DMLMultiTimeCompiler(YarnConfiguration conf, String[] dmlScriptArgs) {
			this.conf = conf;
			this.dmlScriptArgs = dmlScriptArgs;
			compileCount = 0;
		}
		
		public int getCompileCount() {
			return compileCount;
		}
		
		public void tryCompile(CompilationMode compileMode, long cpJvm, HashMap<Long, Double> mrBudget) 
				throws DMLException, ParseException, IOException {
			
			//System.out.println("try compiling CP heap " + OptimizerUtils.toMB(cpJvm) + ", mr budget " + MemOptimizer.serializeRemoteBudgetPlan(mrBudget, true));
			if (compileCount == 0) {	// Initial compile, compile from script
				YarnOptimizerUtils.setCompileMode(compileMode);
				YarnClusterAnalyzer.setRemoteMaxMemPlan(mrBudget);
				InfrastructureAnalyzer.setLocalMaxMemory(cpJvm);
				DMLScript.executeScript(conf, dmlScriptArgs);
			} else {					// Later compile, same scope as dynamic recompilation
				if (dmlProg == null)
					throw new DMLException("Initial compile didn't go to HadoopExecution");
				
				YarnOptimizerUtils.setCompileMode(compileMode);
				YarnClusterAnalyzer.setRemoteMaxMemPlan(mrBudget);
				InfrastructureAnalyzer.setLocalMaxMemory(cpJvm);
				
				YarnClusterAnalyzer.resetSBProbedSet();
				
				//apply hop rewrites (dynamic rewrites, after IPA)
				dmlt.resetHopsDAGVisitStatus(dmlProg);
				ProgramRewriter rewriter2 = new ProgramRewriter(false, true);
				rewriter2.rewriteProgramHopDAGs(dmlProg);
				dmlt.resetHopsDAGVisitStatus(dmlProg);
				
				// Compute memory estimates for all the hops. These estimates are used
				// subsequently in various optimizations, e.g. CP vs. MR scheduling and parfor.
				if (YarnOptimizerUtils.getCompileMode() == CompilationMode.O2_COMPILE_ONLY_AND_HOP_ESTIMATE) {
					ResourceOptimizer.hopMemEstimates.clear();
					// FIXME MB dmlt.refreshMemEstimates(dmlProg, MemOptimizer.hopMemEstimates);
				} else
				{
					// FIXME MB dmlt.refreshMemEstimates(dmlProg, null);
				}
				
				dmlt.printHops(dmlProg);
				
				dmlt.resetHopsDAGVisitStatus(dmlProg);
				// FIXME MB DMLScript.executeHadoop(dmlt, dmlProg, dmlConf);
			}
			compileCount++;
		}
	}
	
	// Output and updated in each compilation attempt
	public static boolean hasMRJobs;
	public static double cost;
	public static ArrayList<Double> hopMemEstimates = new ArrayList<Double>();	// Budget estimates in Byte from hop
	public static DMLTranslator dmlt;
	public static DMLProgram dmlProg = null;
	public static DMLConfig dmlConf;
	
	// Input of each compilation attempt and runtime execution
	double cpBudget;
	HashMap<Long, Double> mrBudget = new HashMap<Long, Double>();
		
	// Fixed after initial compilation
	ArrayList<Double> hopMemEstSorted;		// Sorted budget estimates in Byte from hop, might have -1 in the end !!!
	double maxBudget;		// Max budget possible by the largest node in the cluster
	long maxSbId;			// Max (inclusive) statement block Id assigned in initial compile
	
	//ArrayList<Long> probed;	// probed id during compile
	//long maxId;		// maximum id created during compile, inclusive
	
	// Final optimal decision
	double optCPBudgetByte;
	HashMap<Long, Double> optMRBudget = new HashMap<Long, Double>();
	
	public YarnConfiguration conf;
	
	DMLMultiTimeCompiler compiler;
	
	
	public int getCPPhysicalMemMB() {
		return budgetToPhysicalMB(optCPBudgetByte);
	}
	
	public long getCPJvmMem() {
		return budgetToJvm(optCPBudgetByte);
	}
	
	
	public void init(String[] args, int startIndex, YarnConfiguration conf, 
			YarnClient yarnClient) throws IOException, DMLException, YarnException, ParseException {

		String[] dmlScriptArgs = new String [args.length - startIndex];
		for (int i = 0; i < args.length - startIndex; i++)
			dmlScriptArgs[i] = args[startIndex + i];
		dmlScriptArgs = new GenericOptionsParser(conf, dmlScriptArgs).getRemainingArgs();
		compiler = new DMLMultiTimeCompiler(conf, dmlScriptArgs);
		
		this.conf = conf;
		YarnClusterAnalyzer.analyzeYarnCluster(yarnClient, conf, true);
		
		optCPBudgetByte = -1;
		
		// Maximum budget = min(max node, max RM assignment)
		maxBudget = YarnClusterAnalyzer.getNodesMaxBudgetSorted().get(0);
		if (maxBudget <= MINIMAL_CP_BUDGET_ALLOWED || maxBudget <= MINIMAL_MR_BUDGET_ALLOWED)
			throw new IOException("Max cluster node is so small? " + OptimizerUtils.toMB(maxBudget) + "MB");
		
		long maxYarnAllocate = YarnClusterAnalyzer.getMaxPhyAllocate();
		double maxAllocate = ResourceOptimizer.phyToBudget(maxYarnAllocate);
		if (maxBudget > maxAllocate)
			maxBudget = maxAllocate;
		maxBudget--;	// Avoid round up error when converted to physical memory
		
		//----------- The initial compilation attempt ---------------
		mrBudget.clear();
		mrBudget.put((long)-1, maxBudget);
		
		compiler.tryCompile(CompilationMode.O2_COMPILE_ONLY_AND_HOP_ESTIMATE, budgetToJvm(maxBudget), mrBudget);
		
		// FIXME MB maxSbId = StatementBlock.getIDAssignedCount();
		hopMemEstSorted = new ArrayList<Double> (hopMemEstimates);
		Collections.sort(hopMemEstSorted, Collections.reverseOrder());
		
		// Print some results
		System.out.println(maxSbId + " statement blocks created after runtime plan generated");
		System.out.print("Memory requirement of the hops (MB): ");
		for (Double d : hopMemEstSorted)
			System.out.print(OptimizerUtils.toMB(d) + ",");
		System.out.println();
	}
	
	
	public double optimizeGrid(ArrayList<Double> cpGrid, ArrayList<Double> mrGrid, boolean verbose) throws DMLException, ParseException, IOException {
		// cpMem -> (sbId -> mrMem or cost)
		HashMap<Double, HashMap<Long, ArrayList<Double>>> memSpace = new HashMap<Double, HashMap<Long, ArrayList<Double>>> ();
		HashMap<Double, HashMap<Long, ArrayList<Double>>> costSpace = new HashMap<Double, HashMap<Long, ArrayList<Double>>> ();
		HashMap<Double, HashMap<Long, Double>> optMRMemChoice = new HashMap<Double, HashMap<Long, Double>> ();
		
		// cpMem -> optCost
		HashMap<Double, Double> cpOptCosts = new HashMap<Double, Double> ();
		
		double globalOptCost = -1;
		// Search the CP space
		for (Double tryCp : cpGrid) {
			// Try minimal MR memory as baseline plan
			mrBudget.clear();
			mrBudget.put((long)-1, MINIMAL_MR_BUDGET_ALLOWED);
			for (long i = 1; i <= maxSbId; i++)
				mrBudget.put(i, MINIMAL_MR_BUDGET_ALLOWED);
			
			if (verbose)
				System.out.println("try cp = " + OptimizerUtils.toMB(tryCp));
			// Baseline plan
			compiler.tryCompile(CompilationMode.O1_COMPILE_ONLY_SILENT, budgetToJvm(tryCp), mrBudget);
			
			// FIXME MB 
			if (!InfrastructureAnalyzer.checkValidMemPlan(hasMRJobs)) {
				if (verbose)
					System.out.println("   base plan invalid");
				continue;	// not enough cluster memory for MR
			}
			
			
			ArrayList<Long> probed = null; // FIXME MB new ArrayList<Long> (InfrastructureAnalyzer.getSBProbedSet());
			Collections.sort(probed);
			if (hasMRJobs != (probed.size() > 0))
				throw new RuntimeException("Probed but no MR job? " + probed.size() + ", " + hasMRJobs);
			// FIXME MB if (StatementBlock.getIDAssignedCount() != maxSbId)
			// FIXME MB 	throw new RuntimeException("statement block max id different " + maxSbId + ", " + StatementBlock.getIDAssignedCount());
			
			double baseCost = cost;
			double optCostCurCp = baseCost;
			if (verbose) {
				System.out.println("   base cost " + baseCost);
				System.out.print("   probed sbId: ");
				
				for (Long id : probed)
					System.out.print(id + ",");
				System.out.println();
			}
			
			// Initialize the cost results 
			HashMap<Long, ArrayList<Double>> cpMems = new HashMap<Long, ArrayList<Double>>();
			HashMap<Long, ArrayList<Double>> cpCosts = new HashMap<Long, ArrayList<Double>>();
			HashMap<Long, Double> optChoice = new HashMap<Long, Double> ();
			memSpace.put(tryCp, cpMems);
			costSpace.put(tryCp, cpCosts);
			optMRMemChoice.put(tryCp, optChoice);
			for (Long sbId : probed) {
				ArrayList<Double> mems = new ArrayList<Double>();
				ArrayList<Double> costs = new ArrayList<Double>();
				mems.add(MINIMAL_MR_BUDGET_ALLOWED);
				costs.add(baseCost);
				cpMems.put(sbId, mems);
				cpCosts.put(sbId, costs);
				optChoice.put(sbId, MINIMAL_MR_BUDGET_ALLOWED);
			}
			
			// Search the MR space
			for (Long sbId : probed) {	// Search for the optimal MR memory for each statment block
				if (verbose)
					System.out.println("   try sbId = " + sbId);
				double minCostInCurSb = baseCost;
				for (Double tryMr : mrGrid) {
					if (tryMr == MINIMAL_MR_BUDGET_ALLOWED) {
						if (verbose)
							System.out.println("      skip mr = " + OptimizerUtils.toMB(MINIMAL_MR_BUDGET_ALLOWED));
						continue;
					}
					mrBudget.put(sbId, tryMr);
					if (verbose)
						System.out.print("      try mr = " + OptimizerUtils.toMB(tryMr));
					compiler.tryCompile(CompilationMode.O1_COMPILE_ONLY_SILENT, budgetToJvm(tryCp), mrBudget);
					
					if (!YarnClusterAnalyzer.checkValidMemPlan(hasMRJobs)) {
						if (verbose)
							System.out.println(", break at mr = " + OptimizerUtils.toMB(tryMr));
						break;	// not enough cluster memory for MR
					}
					if (verbose)
						System.out.println(", get cost " + cost);
					
					// FIXME MB if (YarnClusterAnalyzer.getSBProbedSet().size() != probed.size())
					// FIXME MB 	throw new RuntimeException("A different probed set " + InfrastructureAnalyzer.getSBProbedSet().size() + ", " + probed.size());
					if (!hasMRJobs)
						throw new RuntimeException("No mr jobs?!");
					// FIXME MB if (StatementBlock.getIDAssignedCount() != maxSbId)
					// FIXME MB 	throw new RuntimeException("statement block max id different " + maxSbId + ", " + StatementBlock.getIDAssignedCount());
					
					cpMems.get(sbId).add(tryMr);
					cpCosts.get(sbId).add(cost);
					if (minCostInCurSb > cost) {
						minCostInCurSb = cost;
						optChoice.put(sbId, tryMr);
					}
				}
				mrBudget.put(sbId, MINIMAL_MR_BUDGET_ALLOWED);
				
				optCostCurCp -= baseCost - minCostInCurSb;
				if (verbose)
					System.out.println("      sbId " + sbId + " reduce the cost by " + (baseCost - minCostInCurSb) + 
						" using mr = " + OptimizerUtils.toMB(optChoice.get(sbId)));
			}
			
			cpOptCosts.put(tryCp, optCostCurCp);
			if (globalOptCost == -1 || globalOptCost > optCostCurCp) {
				globalOptCost = optCostCurCp;
				optCPBudgetByte = tryCp;
			}
			if (verbose)
				System.out.println("   minimal cost " + optCostCurCp + " using cp = " + OptimizerUtils.toMB(tryCp) + "\n");
		}
		
		// Set the opt mr settings
		optMRBudget.clear();
		for (Map.Entry<Long, Double> entry : optMRMemChoice.get(optCPBudgetByte).entrySet())
			optMRBudget.put(entry.getKey(), entry.getValue());
		
		//if (verbose)
		//	System.out.println("Search done, optimal cost " + globalOptCost + " with cp = " + OptimizerUtils.toMB(optCPBudgetByte));
		return globalOptCost;
	}


	*/
}
