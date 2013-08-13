package com.ibm.bi.dml.hops.cost;

import java.util.HashSet;

import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.instructions.CPInstructionParser;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRInstructionParser;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.CPInstruction.CPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.BinaryMRInstructionBase;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CM_N_COVInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.GroupedAggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MMTSJMRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.PickByCountInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RandInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.TertiaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.UnaryMRInstructionBase;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction.MRINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;

/**
 * 
 */
public class CostEstimatorStaticRuntime extends CostEstimator
{
	//time-conversion
	private static final long DEFAULT_FLOPS = 2L * 1024 * 1024 * 1024; //2GFLOPS
	private static final long UNKNOWN_TIME = -1;
	
	//floating point operations
	private static final double DEFAULT_NFLOP_RAND = 2;
	private static final double DEFAULT_NFLOP_NOOP = 10; 
	private static final double DEFAULT_NFLOP_UNKNOWN = 1; 
	private static final double DEFAULT_NFLOP_CP = 1; 	
	
	//MR job latency
	private static final double DEFAULT_MR_LATENCY_LOCAL = 2;
	private static final double DEFAULT_MR_LATENCY_REMOTE = 10;
	
	//IO throughput //TODO different formats
	private static final double DEFAULT_MBS_FSREAD_BINARYBLOCK_DENSE = 150;
	private static final double DEFAULT_MBS_FSWRITE_BINARYBLOCK_DENSE = 100;
	private static final double DEFAULT_MBS_HDFSREAD_BINARYBLOCK_DENSE = 90;
	private static final double DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_DENSE = 60;
	private static final double DEFAULT_MBS_FSREAD_BINARYBLOCK_SPARSE = 75;
	private static final double DEFAULT_MBS_FSWRITE_BINARYBLOCK_SPARSE = 50;
	private static final double DEFAULT_MBS_HDFSREAD_BINARYBLOCK_SPARSE = 45;
	private static final double DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_SPARSE = 30;
	
	@Override
	protected double getCPInstTimeEstimate( Instruction inst, VarStats[] vs, String[] args ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//load time into mem
		double ltime = 0;
		if( !vs[0]._inmem )
			ltime += getHDFSReadTime( vs[0]._rlen, vs[0]._clen, (vs[0]._nnz<0)? 1.0:(double)vs[0]._nnz/vs[0]._rlen/vs[0]._clen );
		if( !vs[1]._inmem )
			ltime += getHDFSReadTime( vs[1]._rlen, vs[1]._clen, (vs[1]._nnz<0)? 1.0:(double)vs[1]._nnz/vs[1]._rlen/vs[1]._clen );
				
		//exec time CP instruction
		double etime = getInstTimeEstimate(inst.toString(), vs, args);
	
		//write time caching
		double wtime = 0;
		//double wtime = getFSWriteTime( vs[2]._rlen, vs[2]._clen, (vs[2]._nnz<0)? 1.0:(double)vs[2]._nnz/vs[2]._rlen/vs[2]._clen );
		
		//total costs
		double costs = ltime + etime + wtime;
		
		//if( LOG.isDebugEnabled() )
		//	LOG.debug("Costs CP instruction = "+costs);
		
		return costs;
	}
	
	
	@Override
	protected double getMRJobInstTimeEstimate( Instruction inst, VarStats[] vs, String[] args ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		MRJobInstruction jinst = (MRJobInstruction) inst;
		
		//infrastructure properties
		boolean localJob = InfrastructureAnalyzer.isLocalMode();
		int maxPMap = InfrastructureAnalyzer.getRemoteParallelMapTasks(); 	
		int maxPRed = Math.min( InfrastructureAnalyzer.getRemoteParallelReduceTasks(),
				        Integer.parseInt(ConfigurationManager.getConfig().getTextValue(DMLConfig.NUM_REDUCERS)) );
		double blocksize = ((double)InfrastructureAnalyzer.getHDFSBlockSize())/(1024*1024);
		
		//job properties
		boolean mapOnly = jinst.isMapOnly();
		String rdInst = jinst.getIv_randInstructions();
		String rrInst = jinst.getIv_recordReaderInstructions();
		String mapInst = jinst.getIv_instructionsInMapper();
		String shfInst = jinst.getIv_shuffleInstructions();
		String aggInst = jinst.getIv_aggInstructions();
		String otherInst = jinst.getIv_otherInstructions();		
		byte[] inIx = getInputIndexes( jinst.getInputVars() );
		byte[] retIx = jinst.getIv_resultIndices();
		byte[] mapOutIx = getMapOutputIndexes(inIx, retIx, rdInst, mapInst, shfInst, aggInst, otherInst);
		int numMap = computeNumMapTasks(vs, inIx, blocksize, maxPMap, jinst.getJobType());
		int numPMap = Math.min(numMap, maxPMap);
		int numRed = computeNumReduceTasks( vs, mapOutIx, jinst.getJobType() );
		int numPRed = Math.min(numRed, maxPRed);
		
		//step 0: export if inputs in mem
		double exportCosts = 0; 
		for( int i=0; i<jinst.getInputVars().length; i++ )
			if( vs[i]._inmem )
				exportCosts += getHDFSWriteTime(vs[i]._rlen, vs[i]._clen, vs[i].getSparsity());
		
		//step 1: MR job latency
		double latencyCosts = 0;
		if( localJob )
			latencyCosts += DEFAULT_MR_LATENCY_LOCAL;
		else
			latencyCosts += DEFAULT_MR_LATENCY_REMOTE;

		//step 2: parallel read of inputs
		double hdfsReadCosts = 0;
		for( int i=0; i<jinst.getInputVars().length; i++ )
			hdfsReadCosts += (getHDFSReadTime(vs[i]._rlen, vs[i]._clen, vs[i].getSparsity()) / numPMap); 
				
		//step 3: parallel MR instructions
		String[] mapperInst = new String[]{rdInst, rrInst, mapInst};
		String[] reducerInst = new String[]{shfInst, aggInst, otherInst};	
		
		//map instructions
		double mapCosts = 0;
		double shuffleCosts = 0;
		double reduceCosts = 0;
		
		for( String instCat : mapperInst )
			if( instCat != null && instCat.length()>0 ) {
				String[] linst = instCat.split( Lops.INSTRUCTION_DELIMITOR );
				for( String tmp : linst ){
					Object[] o = extractMRInstStatistics(tmp, vs);
					mapCosts += getInstTimeEstimate(tmp, (VarStats[])o[0], (String[])o[1]);
				}
			}
		mapCosts /= numPMap;
		
		if( !mapOnly )
		{
			//shuffle costs
			//TODO account for 1) combiner and 2) specific job types (incl parallelism) 
			for( int i=0; i<mapOutIx.length; i++ )
			{
				shuffleCosts += ( getFSWriteTime(vs[mapOutIx[i]]._rlen, vs[mapOutIx[i]]._clen, vs[mapOutIx[i]].getSparsity()) / numPMap
						        + getFSReadTime(vs[mapOutIx[i]]._rlen, vs[mapOutIx[i]]._clen, vs[mapOutIx[i]].getSparsity()) / numPRed); 	
			}
						
			//reduce instructions
			for( String instCat : reducerInst )
				if( instCat != null && instCat.length()>0 ) {
					String[] linst = instCat.split( Lops.INSTRUCTION_DELIMITOR );
					for( String tmp : linst ){
						Object[] o = extractMRInstStatistics(tmp, vs);
						if(InstructionUtils.getMRType(tmp)==MRINSTRUCTION_TYPE.Aggregate)
							o[1] = new String[]{String.valueOf(numMap)};
						reduceCosts += getInstTimeEstimate(tmp, (VarStats[])o[0], (String[])o[1]);
					}
				}
			reduceCosts /= numPRed;
		}		
		
		//step 4: parallel write of outputs
		double hdfsWriteCosts = 0;
		for( int i=0; i<jinst.getOutputVars().length; i++ )
		{
			hdfsWriteCosts += getHDFSWriteTime(vs[retIx[i]]._rlen, vs[retIx[i]]._clen, vs[retIx[i]].getSparsity())
			       / ((mapOnly)? numPMap : numPRed); 
		}
		
		if( LOG.isDebugEnabled() )
		{
			LOG.debug("Costs Export = "+exportCosts);
			LOG.debug("Costs Latency = "+latencyCosts);
			LOG.debug("Costs HDFS Read = "+hdfsReadCosts);
			LOG.debug("Costs Map Exec = "+mapCosts);
			LOG.debug("Costs Shuffle = "+shuffleCosts);
			LOG.debug("Costs Reduce Exec = "+reduceCosts);
			LOG.debug("Costs HDFS Write = "+hdfsWriteCosts);
		}

		return exportCosts + latencyCosts + hdfsReadCosts + mapCosts 
		       + shuffleCosts + reduceCosts + hdfsWriteCosts;
	}		
	
	/**
	 * 
	 * @param inst
	 * @param stats
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	private Object[] extractMRInstStatistics( String inst, VarStats[] stats ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		Object[] ret = new Object[2]; //stats, attrs
		VarStats[] vs = new VarStats[3];
		String[] attr = null; 

		String[] parts = InstructionUtils.getInstructionParts(inst);
		String opcode = parts[0];
		
		
		if( opcode.equals("Rand") )
		{
			vs[0] = _unknownStats;
			vs[1] = _unknownStats;
			vs[2] = stats[Integer.parseInt(parts[2])];
			
			double minValue = Double.parseDouble(parts[7]);
			double maxValue = Double.parseDouble(parts[8]);
			double sparsity = Double.parseDouble(parts[9]);
			int type = 2; 
			if( minValue == 0.0 && maxValue == 0.0 )
				type = 0;
			else if( sparsity == 1.0 && minValue == maxValue )
				type = 1;
			attr = new String[]{String.valueOf(type)};
		}	
		else //general case
		{
			
			String inst2 = replaceInstructionPatch( inst );
			MRInstruction mrinst = MRInstructionParser.parseSingleInstruction(inst2);
			
			if( mrinst instanceof UnaryMRInstructionBase )
			{
				UnaryMRInstructionBase uinst = (UnaryMRInstructionBase) mrinst;
				vs[0] = stats[ uinst.input ];
				vs[1] = _unknownStats;
				vs[2] = stats[ uinst.output ];
				
				if( vs[0] == null ) //scalar input, e.g., print
					vs[0] = _scalarStats;
				if( vs[2] == null ) //scalar output
					vs[2] = _scalarStats;
				
				if( mrinst instanceof MMTSJMRInstruction )
				{
					String type = ((MMTSJMRInstruction)mrinst).getMMTSJType().toString();
					attr = new String[]{type};
				}  
				else if( mrinst instanceof CM_N_COVInstruction )
				{
					if( opcode.equals("cm") )
						attr = new String[]{parts[parts.length-2]};		
				}
				else if( mrinst instanceof GroupedAggregateInstruction )
				{
					if( opcode.equals("groupedagg") )
						attr = new String[]{String.valueOf(CMOperator.AggregateOperationTypes.valueOf(parts[parts.length-2].toUpperCase()).ordinal())};
					
				}
			}
			else if( mrinst instanceof BinaryMRInstructionBase )
			{
				BinaryMRInstructionBase binst = (BinaryMRInstructionBase) mrinst;
				vs[0] = stats[ binst.input1 ];
				vs[1] = stats[ binst.input2 ];
				vs[2] = stats[ binst.output ];
								
				if( vs[0] == null ) //scalar input, 
					vs[0] = _scalarStats;
				if( vs[1] == null ) //scalar input, 
					vs[1] = _scalarStats;
				if( vs[2] == null ) //scalar output
					vs[2] = _scalarStats;
			}
			else if( mrinst instanceof TertiaryInstruction )
			{
				TertiaryInstruction tinst = (TertiaryInstruction) mrinst;
				vs[0] = stats[ tinst.input1 ];
				vs[1] = stats[ tinst.input2 ];
				vs[2] = stats[ tinst.input3 ];
				
				if( vs[0] == null ) //scalar input, 
					vs[0] = _scalarStats;
				if( vs[1] == null ) //scalar input, 
					vs[1] = _scalarStats;
				if( vs[2] == null ) //scalar input
					vs[2] = _scalarStats;
			}
			else if( mrinst instanceof PickByCountInstruction )
			{
				PickByCountInstruction pinst = (PickByCountInstruction) mrinst;
				vs[0] = stats[ pinst.input1 ];
				vs[2] = stats[ pinst.output ];
				if( vs[0] == null ) //scalar input, 
					vs[0] = _scalarStats;
				if( vs[1] == null ) //scalar input, 
					vs[1] = _scalarStats;
				if( vs[2] == null ) //scalar input
					vs[2] = _scalarStats;
			}
		}
		
		//maintain var status (CP output always inmem)
		vs[2]._inmem = true;
		
		ret[0] = vs;
		ret[1] = attr;
		
		return ret;
	}
	
	

	/////////////////////
	// Utilities       //
	/////////////////////	
	
	/**
	 * 
	 * @param inputVars
	 * @return
	 */
	private byte[] getInputIndexes(String[] inputVars)
	{
		byte[] inIx = new byte[inputVars.length];
		for( int i=0; i<inIx.length; i++ )
			inIx[i] = (byte)i;
		return inIx;
	}
	
	/**
	 * 
	 * @param inIx
	 * @param retIx
	 * @param rdInst
	 * @param mapInst
	 * @param shfInst
	 * @param aggInst
	 * @param otherInst
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	private byte[] getMapOutputIndexes( byte[] inIx, byte[] retIx, String rdInst, String mapInst, String shfInst, String aggInst, String otherInst ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//note: this is a simplified version of MRJobConfiguration.setUpOutputIndexesForMapper
		
		//map indices
		HashSet<Byte> ixMap = new HashSet<Byte>();
		for( byte ix : inIx )
			ixMap.add(ix);
		
		if( rdInst!=null && rdInst.length()>0 ) {
			rdInst = replaceInstructionPatch(rdInst);
			RandInstruction[] ins = MRInstructionParser.parseRandInstructions(rdInst);
			for( RandInstruction inst : ins )
				for( byte ix : inst.getAllIndexes() )
					ixMap.add(ix);
		}
		
		if( mapInst!=null && mapInst.length()>0 ) {
			mapInst = replaceInstructionPatch(mapInst);
			Instruction[] ins = MRInstructionParser.parseMixedInstructions(mapInst);
			for( Instruction inst : ins )
				for( byte ix : inst.getAllIndexes() )
					ixMap.add(ix);
		}
		
		if( shfInst!=null && shfInst.length()>0 ) {
			shfInst = replaceInstructionPatch(shfInst);
			Instruction[] ins = MRInstructionParser.parseMixedInstructions(shfInst);
			for( Instruction inst : ins )
				for( byte ix : inst.getAllIndexes() )
					ixMap.add(ix);
		}
		
		//reduce indices
		HashSet<Byte> ixRed = new HashSet<Byte>();
		for( byte ix : retIx )
			ixRed.add(ix);
	
		if( aggInst!=null && aggInst.length()>0 ) {
			aggInst = replaceInstructionPatch(aggInst);
			Instruction[] ins = MRInstructionParser.parseAggregateInstructions(aggInst);
			for( Instruction inst : ins )
				for( byte ix : inst.getAllIndexes() )
					ixRed.add(ix);
		}
		
		if( otherInst!=null && otherInst.length()>0 ) {
			otherInst = replaceInstructionPatch(otherInst);
			Instruction[] ins = MRInstructionParser.parseMixedInstructions(otherInst);
			for( Instruction inst : ins )
				for( byte ix : inst.getAllIndexes() )
					ixRed.add(ix);
		}

		//difference
		ixMap.retainAll(ixRed);
			
		//copy result
		byte[] ret = new byte[ixMap.size()];
		int i = 0;
		for( byte ix : ixMap )
			ret[i++] = ix;
		
		return ret;
	}
	
	/**
	 * 
	 * @param vs
	 * @param inputIx
	 * @param blocksize
	 * @param maxPMap
	 * @param jobtype
	 * @return
	 */
	private int computeNumMapTasks( VarStats[] vs, byte[] inputIx, double blocksize, int maxPMap, JobType jobtype )
	{
		//special cases
		if( jobtype == JobType.RAND )
			return maxPMap;
			
		//input size, num blocks
		double mapInputSize = 0;
		int numBlocks = 0;
		for( int i=0; i<inputIx.length; i++ )
		{
			//input size
			boolean sparse = (vs[inputIx[i]].getSparsity()<MatrixBlockDSM.SPARCITY_TURN_POINT && vs[inputIx[i]]._clen>MatrixBlockDSM.SKINNY_MATRIX_TURN_POINT);
			mapInputSize += ((double)MatrixBlockDSM.estimateSizeOnDisk((long)vs[inputIx[i]]._rlen, (long)vs[inputIx[i]]._clen, (long)vs[inputIx[i]]._nnz, sparse)) / (1024*1024);	
		
			//num blocks
			int lret =  (int) Math.ceil((double)vs[inputIx[i]]._rlen/vs[inputIx[i]]._brlen)
	                   *(int) Math.ceil((double)vs[inputIx[i]]._clen/vs[inputIx[i]]._bclen);
			numBlocks = Math.max(lret, numBlocks);
		}
		
		return Math.max(1, Math.min( (int)Math.ceil(mapInputSize/blocksize),numBlocks ));
	}
	
	/**
	 * 
	 * @param vs
	 * @param mapOutIx
	 * @param jobtype
	 * @return
	 */
	private int computeNumReduceTasks( VarStats[] vs, byte[] mapOutIx, JobType jobtype )
	{
		int ret = -1;
		
		//TODO for jobtype==JobType.MMCJ common dim

		for( int i=0; i<mapOutIx.length; i++ )
		{
			int lret =  (int) Math.ceil((double)vs[mapOutIx[i]]._rlen/vs[mapOutIx[i]]._brlen)
			           *(int) Math.ceil((double)vs[mapOutIx[i]]._clen/vs[mapOutIx[i]]._bclen);
			ret = Math.max(lret, ret);
		}
		
		return Math.max(1, ret);
	}

	
	
	/////////////////////
	// I/O Costs       //
	/////////////////////	
	
	/**
	 * Returns the estimated read time from HDFS. 
	 * NOTE: Does not handle unknowns.
	 * 
	 * @param dm
	 * @param dn
	 * @param ds
	 * @return
	 */
	private double getHDFSReadTime( long dm, long dn, double ds )
	{
		boolean sparse = (ds<MatrixBlockDSM.SPARCITY_TURN_POINT && dn>MatrixBlockDSM.SKINNY_MATRIX_TURN_POINT);
		
		double ret = ((double)MatrixBlockDSM.estimateSizeOnDisk((long)dm, (long)dn, (long)ds*dm*dn, sparse)) / (1024*1024);  		
		if( sparse )
			ret /= DEFAULT_MBS_HDFSREAD_BINARYBLOCK_SPARSE;
		else //dense
			ret /= DEFAULT_MBS_HDFSREAD_BINARYBLOCK_DENSE;
		
		return ret;
	}
	
	/**
	 * 
	 * @param dm
	 * @param dn
	 * @param ds
	 * @return
	 */
	private double getHDFSWriteTime( long dm, long dn, double ds )
	{
		boolean sparse = (ds<MatrixBlockDSM.SPARCITY_TURN_POINT && dn>MatrixBlockDSM.SKINNY_MATRIX_TURN_POINT);
		
		double ret = ((double)MatrixBlockDSM.estimateSizeOnDisk((long)dm, (long)dn, (long)ds*dm*dn, sparse)) / (1024*1024);  		
		
		if( sparse )
			ret /= DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_SPARSE;
		else //dense
			ret /= DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_DENSE;
		
		return ret;
	}

	/**
	 * Returns the estimated read time from local FS. 
	 * NOTE: Does not handle unknowns.
	 * 
	 * @param dm
	 * @param dn
	 * @param ds
	 * @return
	 */
	private double getFSReadTime( long dm, long dn, double ds )
	{
		boolean sparse = (ds<MatrixBlockDSM.SPARCITY_TURN_POINT && dn>MatrixBlockDSM.SKINNY_MATRIX_TURN_POINT);
		
		double ret = ((double)MatrixBlockDSM.estimateSizeOnDisk((long)dm, (long)dn, (long)ds*dm*dn, sparse)) / (1024*1024);  		
		if( sparse )
			ret /= DEFAULT_MBS_FSREAD_BINARYBLOCK_SPARSE;
		else //dense
			ret /= DEFAULT_MBS_FSREAD_BINARYBLOCK_DENSE;
		
		return ret;
	}

	/**
	 * 
	 * @param dm
	 * @param dn
	 * @param ds
	 * @return
	 */
	private double getFSWriteTime( long dm, long dn, double ds )
	{
		boolean sparse = (ds<MatrixBlockDSM.SPARCITY_TURN_POINT && dn>MatrixBlockDSM.SKINNY_MATRIX_TURN_POINT);
		
		double ret = ((double)MatrixBlockDSM.estimateSizeOnDisk((long)dm, (long)dn, (long)ds*dm*dn, sparse)) / (1024*1024);  		
		
		if( sparse )
			ret /= DEFAULT_MBS_FSWRITE_BINARYBLOCK_SPARSE;
		else //dense
			ret /= DEFAULT_MBS_FSWRITE_BINARYBLOCK_DENSE;
		
		return ret;
	}

	
	/////////////////////
	// Operation Costs //
	/////////////////////
	
	/**
	 * 
	 * @param inst
	 * @param vs
	 * @param args
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	private double getInstTimeEstimate(String instStr, VarStats[] vs, String[] args) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		String opcode = InstructionUtils.getOpCode(instStr);
		return getInstTimeEstimate(opcode, 
				                   vs[0]._rlen, vs[0]._clen, (vs[0]._nnz<0)? 1.0:(double)vs[0]._nnz/vs[0]._rlen/vs[0]._clen, 
						           vs[1]._rlen, vs[1]._clen, (vs[1]._nnz<0)? 1.0:(double)vs[1]._nnz/vs[1]._rlen/vs[1]._clen, 
						           vs[2]._rlen, vs[2]._clen, (vs[2]._nnz<0)? 1.0:(double)vs[2]._nnz/vs[2]._rlen/vs[2]._clen,
						           args);
	}
	
	/**
	 * Returns the estimated instruction execution time, w/o data transfer and single-threaded.
	 * For scalars input dims must be set to 1 before invocation. 
	 * 
	 * NOTE: Does not handle unknowns.
	 * 
	 * @param opcode
	 * @param d1m
	 * @param d1n
	 * @param d1s
	 * @param d2m
	 * @param d2n
	 * @param d2s
	 * @param d3m
	 * @param d3n
	 * @param d3s
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	private double getInstTimeEstimate( String opcode, long d1m, long d1n, double d1s, long d2m, long d2n, double d2s, long d3m, long d3n, double d3s, String[] args ) throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		double nflops = getNFLOP(opcode, d1m, d1n, d1s, d2m, d2n, d2s, d3m, d3n, d3s, args);
		double time = nflops / DEFAULT_FLOPS;
		
		if( LOG.isDebugEnabled() )
			LOG.debug("Cost["+opcode+"] = "+time+"s, "+nflops+" flops ("+d1m+","+d1n+","+d1s+","+d2m+","+d2n+","+d2s+","+d3m+","+d3n+","+d3s+").");
		
		return time;
	}
	
	/**
	 * 
	 * @param optype
	 * @param d1m
	 * @param d1n
	 * @param d1s
	 * @param d2m
	 * @param d2n
	 * @param d2s
	 * @param d3m
	 * @param d3n
	 * @param d3s
	 * @param args
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	private double getNFLOP( String optype, long d1m, long d1n, double d1s, long d2m, long d2n, double d2s, long d3m, long d3n, double d3s, String[] args ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//operation costs in FLOP on matrix block level (for CP and MR instructions)
		//(excludes IO and parallelism; assumes known dims for all inputs, outputs )
	
		boolean leftSparse = (d1s<MatrixBlockDSM.SPARCITY_TURN_POINT && d1n>MatrixBlockDSM.SKINNY_MATRIX_TURN_POINT);
		boolean rightSparse = (d2s<MatrixBlockDSM.SPARCITY_TURN_POINT && d2n>MatrixBlockDSM.SKINNY_MATRIX_TURN_POINT);
		boolean onlyLeft = (d1m>=0 && d1n>=0 && d2m<0 && d2n<0 );
		boolean allExists = (d1m>=0 && d1n>=0 && d2m>=0 && d2n>=0 && d3m>=0 && d3n>=0 );
		
		//NOTE: all instruction types that are equivalent in CP and MR are only
		//included in CP to prevent redundancy
		CPINSTRUCTION_TYPE cptype = CPInstructionParser.String2CPInstructionType.get(optype);
		if( cptype != null ) //for CP Ops and equivalent MR ops 
		{
			//general approach: count of floating point *, /, +, -, ^, builtin ;
			switch(cptype) 
			{
			
				case AggregateBinary: //opcodes: ba+*, cov
					if( optype.equals("ba+*") ) { //matrix mult
						if( !leftSparse && !rightSparse )
							return 2 * (d1m * d1n * d1s * d2n);
						else if( !leftSparse && rightSparse )
							return 2 * (d1m * d1n * d1s * d2n * d2s);
						else if( leftSparse && !rightSparse )
							return 2 * (d1m * d1n * d1s * d2n);
						else //leftSparse && rightSparse
							return 2 * (d1m * d1n * d1s * d2n * d2s);
					}
					else if( optype.equals("cov") ) {
						//note: output always scalar, d3 used as weights block
						//if( allExists ), same runtime for 2 and 3 inputs
						return 23 * d1m; //(11+3*k+)
					}
					
					return 0;
				
				case AggregateUnary: //opcodes: uak+, uark+, uack+, uamean, uarmean, uacmean, 
									 //         uamax, uarmax, uarimax, uacmax, uamin, uarmin, uacmin, 
									 //         ua+, uar+, uac+, ua*, uatrace, uaktrace, rdiagM2V, 
					                 //         nrow, ncol, length, cm
					
					if( optype.equals("nrow") || optype.equals("ncol") || optype.equals("length") )
						return DEFAULT_NFLOP_NOOP;
					else if( optype.equals( "cm" ) ) {
						double xcm = 1;
						switch( Integer.parseInt(args[0]) ) {
							case 0: xcm=1; break; //count
							case 1: xcm=8; break; //mean
							case 2: xcm=16; break; //cm2
							case 3: xcm=31; break; //cm3
							case 4: xcm=51; break; //cm4
							case 5: xcm=16; break; //variance
						}
						return (leftSparse) ? xcm * (d1m * d1s + 1) : xcm * d1m;
					}
				    else if( optype.equals("uatrace") || optype.equals("uaktrace") )
				    	return 2 * d1m * d1n;
				    else if( optype.equals("rdiagM2V") )
				    	return d1m * d1n * DEFAULT_NFLOP_CP;
				    else if( optype.equals("ua+") || optype.equals("uar+") || optype.equals("uar+")  ){
				    	//sparse safe operations
				    	if( !leftSparse ) //dense
				    		return d1m * d1n;
				    	else //sparse
				    		return d1m * d1n * d1s;
				    }
				    else if( optype.equals("uak+") || optype.equals("uark+") || optype.equals("uark+"))
				    	return 4 * d1m * d1n; //1*k+
				    else if( optype.equals("uamean") || optype.equals("uarmean") || optype.equals("uacmean"))
						return 7 * d1m * d1n; //1*k+
				    else if(   optype.equals("uamax") || optype.equals("uarmax") || optype.equals("uacmax") 
				    		|| optype.equals("uamin") || optype.equals("uarmin") || optype.equals("uacmin")
				    		|| optype.equals("uarimax") || optype.equals("ua*") )
				    	return d1m * d1n;
					
				    return 0;	
				    
				case ArithmeticBinary: //opcodes: +, -, *, /, ^
					//note: covers scalar-scalar, scalar-matrix, matrix-matrix
					if( optype.equals("+") || optype.equals("-") //sparse safe
						&& ( leftSparse || rightSparse ) )
						return d1m*d1n*d1s + d2m*d2n*d2s;
					else
						return d3m*d3n;
					
				case Tertiary: //opcodes: ctable
					if( optype.equals("ctable") ){
						if( leftSparse )
							return d1m * d1n * d1s; //add
						else 
							return d1m * d1n;
					}
					return 0;
					
				case BooleanBinary: //opcodes: &&, ||
					return 1; //always scalar-scalar
						
				case BooleanUnary: //opcodes: !
					return 1; //always scalar-scalar

				case Builtin: //opcodes: log 
					//note: covers scalar-scalar, scalar-matrix, matrix-matrix
					//note: can be unary or binary
					if( allExists ) //binary
						return 3 * d3m * d3n;
					else //unary
						return d3m * d3n;
					
				case BuiltinBinary: //opcodes: max, min
					//note: covers scalar-scalar, scalar-matrix, matrix-matrix
					return d3m*d3n;

					
				case BuiltinUnary: //opcodes: exp, abs, sin, cos, tan, sqrt, plogp, print, print2, round
					if( optype.equals("print") || optype.equals("print2") ) //scalar only
						return 1;
					else
					{
						double xbu = 1; //default for all ops
						if( optype.equals("plogp") ) xbu = 2;
						else if( optype.equals("round") ) xbu = 4;
						
						if( optype.equals("sin") || optype.equals("tan") || optype.equals("round")
							|| optype.equals("abs") || optype.equals("sqrt") ) //sparse-safe
						{
							if( leftSparse ) //sparse
								return xbu * d1m * d1n * d1s;	
							else //dense
								return xbu * d1m * d1n;
						}
						else
							return xbu * d1m * d1n;
					}
										
				case Reorg: //opcodes: r', rdiagV2M
				case MatrixReshape: //opcodes: rshape
					if( leftSparse )
						return d1m * d1n * d1s;
					else
						return d1m * d1n;
					
				case Append: //opcodes: append
					return DEFAULT_NFLOP_CP * 
					       (((leftSparse) ? d1m * d1n * d1s : d1m * d1n ) +
					        ((rightSparse) ? d2m * d2n * d2s : d2m * d2n ));
					
				case RelationalBinary: //opcodes: ==, !=, <, >, <=, >=  
					//note: all relational ops are not sparsesafe
					return d3m * d3n; //covers all combinations of scalar and matrix  
					
				case File: //opcodes: rm, mv
					return DEFAULT_NFLOP_NOOP;
					
				case Variable: //opcodes: assignvar, cpvar, rmvar, rmfilevar, assignvarwithfile, attachfiletovar, valuepick, iqsize, read, write, createvar, setfilename, castAsMatrix
					if( optype.equals("write") ){
						if( !leftSparse )
							return d1m * d1n * DEFAULT_NFLOP_CP; 
						else
							return d1m * d1n * d1s * DEFAULT_NFLOP_CP;
					}
					else if ( optype.equals("inmem-iqm") )
						//note: assumes uniform distribution
						return 2 * d1m + //sum of weights
						       5 + 0.25d * d1m + //scan to lower quantile
						       8 * 0.5 * d1m; //scan from lower to upper quantile
					else
						return DEFAULT_NFLOP_NOOP;
			
				case Rand: //opcodes: Rand 
					switch(Integer.parseInt(args[0])) {
						case 0: return DEFAULT_NFLOP_NOOP; //empty matrix
						case 1: return d3m * d3n * DEFAULT_NFLOP_CP; //arrayfill
						case 2: return d3m * d3n * DEFAULT_NFLOP_RAND + 
							           d3m * d3n * d3s * (DEFAULT_NFLOP_RAND + 2);
					}
				case External: //opcodes: extfunct
					//note: should be invoked independently for multiple outputs
					return d1m * d1n * d1s * DEFAULT_NFLOP_UNKNOWN;
					
				case ParameterizedBuiltin: //opcodes: cdf, groupedagg, rmempty
					if( optype.equals("cdf") )
						return DEFAULT_NFLOP_UNKNOWN; //scalar call to commons.math
					else if( optype.equals("groupedagg") ){	
						double xga = 1;
						switch( Integer.parseInt(args[0]) ) {
							case 0: xga=4; break; //sum, see uk+
							case 1: xga=1; break; //count, see cm
							case 2: xga=8; break; //mean
							case 3: xga=16; break; //cm2
							case 4: xga=31; break; //cm3
							case 5: xga=51; break; //cm4
							case 6: xga=16; break; //variance
						}						
						return 2 * d1m + xga * d1m; //scan for min/max, groupedagg
					}	
					else if( optype.equals("rmempty") ){
						switch(Integer.parseInt(args[0])){
							case 0: //remove rows
								return ((leftSparse) ? d1m : d1m * Math.ceil(1.0d/d1s)/2) +
									   DEFAULT_NFLOP_CP * d3m * d2m;
							case 1: //remove cols
								return d1n * Math.ceil(1.0d/d1s)/2 + 
								       DEFAULT_NFLOP_CP * d3m * d2m;
						}
						
					}	
					return 0;
					
				case Sort: //opcodes: sort
					if( optype.equals("sort") ){
						//note: mergesort since comparator used
						double sortCosts = 0;
						if( onlyLeft )
							sortCosts = DEFAULT_NFLOP_CP * d1m + d1m;
						else //w/ weights
							sortCosts = DEFAULT_NFLOP_CP * ((leftSparse)?d1m*d1s:d1m); 
						return sortCosts + d1m*(int)(Math.log(d1m)/Math.log(2)) + //mergesort
										   DEFAULT_NFLOP_CP * d1m;
					}
					return 0;
					
				case MatrixIndexing: //opcodes: rangeReIndex, leftIndex
					if( optype.equals("leftIndex") ){
						return DEFAULT_NFLOP_CP * ((leftSparse)? d1m*d1n*d1s : d1m*d1n)
						       + 2 * DEFAULT_NFLOP_CP * ((rightSparse)? d2m*d2n*d2s : d2m*d2n );
					}
					else if( optype.equals("rangeReIndex") ){
						return DEFAULT_NFLOP_CP * ((leftSparse)? d2m*d2n*d2s : d2m*d2n );
					}
					return 0;
					
				case MMTSJ: //opcodes: tsmm
					//diff to ba+* only upper triangular matrix
					if( MMTSJType.valueOf(args[0])==MMTSJType.LEFT ) { //lefttranspose
						if( !rightSparse ) //dense						
							return d1m * d1n * d1s * d1n;
						else //sparse
							return d1m * d1n * d1s * d1n * d1s; 
					}
					else if(onlyLeft) { //righttranspose
						if( !leftSparse ) //dense
							return d1m * d1n * d1m;
						else //sparse
							return   d1m * d1n * d1s //reorg sparse
							       + d1m * d1n * d1s * d1n * d1s; //core tsmm
					}					
					return 0;
				
				case INVALID:
					return 0;
				
				default: 
					throw new DMLRuntimeException("CostEstimator: unsupported instruction type: "+optype);
			}
				
		}
		
		//if not found in CP instructions
		MRINSTRUCTION_TYPE mrtype = MRInstructionParser.String2MRInstructionType.get(optype);
		if ( mrtype != null ) //for specific MR ops
		{
			switch(mrtype)
			{
				case Aggregate: //opcodes: a+, ak+, a*, amax, amin, amean 
					//TODO should be aggregate unary
					int numMap = Integer.parseInt(args[0]);
					if( optype.equals("ak+") )
				    	return 4 * numMap * d1m * d1n * d1s;
				    else 
						return numMap * d1m * d1n * d1s;
					
				case AggregateBinary: //opcodes: cpmm, rmm, mvmult
					//note: copy from CP costs
					if( optype.equals("cpmm") || optype.equals("rmm") || optype.equals("mvmult") ) { //matrix mult
						if( !leftSparse && !rightSparse )
							return 2 * (d1m * d1n * d1s * d2n);
						else if( !leftSparse && rightSparse )
							return 2 * (d1m * d1n * d1s * d2n * d2s);
						else if( leftSparse && !rightSparse )
							return 2 * (d1m * d1n * d1s * d2n);
						else //leftSparse && rightSparse
							return 2 * (d1m * d1n * d1s * d2n * d2s);
					}
					return 0;
					
				case ArithmeticBinary: //opcodes: s-r, so, max, min, 
					                   //         >, >=, <, <=, ==, != 
					//TODO Should be relational 
				
					//note: all relational ops are not sparsesafe
					return d3m * d3n; //covers all combinations of scalar and matrix  
	
				case CombineUnary: //opcodes: combineunary
					return d1m * d1n * d1s;
					
				case CombineBinary: //opcodes: combinebinary
					return   d1m * d1n * d1s
					       + d2m * d2n * d2s;
					
				case CombineTertiary: //opcodes: combinetertiary
					return   d1m * d1n * d1s
				           + d2m * d2n * d2s
				           + d3m * d3n * d3s;
					
				case Unary: //opcodes: log, slog, pow 			
					//TODO requires opcode consolidation (builtin, arithmic)
					//note: covers scalar, matrix, matrix-scalar
					return d3m * d3n;
					
				case Tertiary: //opcodes: ctabletransform, ctabletransformscalarweight, ctabletransformhistogram, ctabletransformweightedhistogram
					//note: copy from cp
					if( leftSparse )
						return d1m * d1n * d1s; //add
					else 
						return d1m * d1n;
					
				case Reblock: //opcodes: rblk
					return DEFAULT_NFLOP_CP * ((leftSparse)? d1m*d1n*d1s : d1m*d1n); 
					
				case CM_N_COV: //opcodes: mean
					double xcm = 8;
					return (leftSparse) ? xcm * (d1m * d1s + 1) : xcm * d1m;
					
				case GroupedAggregate: //opcodes: groupedagg		
					//TODO: need to consolidate categories (ParameterizedBuiltin)
					//copy from CP opertion
					double xga = 1;
					switch( Integer.parseInt(args[0]) ) {
						case 0: xga=4; break; //sum, see uk+
						case 1: xga=1; break; //count, see cm
						case 2: xga=8; break; //mean
						case 3: xga=16; break; //cm2
						case 4: xga=31; break; //cm3
						case 5: xga=51; break; //cm4
						case 6: xga=16; break; //variance
					}						
					return 2 * d1m + xga * d1m; //scan for min/max, groupedagg
					
				case PickByCount: //opcodes: valuepick, rangepick
					break;
					//TODO
					//String2MRInstructionType.put( "valuepick"  , MRINSTRUCTION_TYPE.PickByCount);  // for quantile()
					//String2MRInstructionType.put( "rangepick"  , MRINSTRUCTION_TYPE.PickByCount);  // for interQuantile()
					
				case RangeReIndex: //opcodes: rangeReIndex, rangeReIndexForLeft
					//TODO: requires category consolidation
					if( optype.equals("rangeReIndex") )
						return DEFAULT_NFLOP_CP * ((leftSparse)? d2m*d2n*d2s : d2m*d2n );
					else //rangeReIndexForLeft
						return   DEFAULT_NFLOP_CP * ((leftSparse)? d1m*d1n*d1s : d1m*d1n)
					           + DEFAULT_NFLOP_CP * ((rightSparse)? d2m*d2n*d2s : d2m*d2n );
	
				case ZeroOut: //opcodes: zeroOut
					return   DEFAULT_NFLOP_CP * ((leftSparse)? d1m*d1n*d1s : d1m*d1n)
				           + DEFAULT_NFLOP_CP * ((rightSparse)? d2m*d2n*d2s : d2m*d2n );								
			}
		}
		else
		{
			throw new DMLRuntimeException("CostEstimator: unsupported instruction type: "+optype);
		}
		
		//TODO Parameterized Builtin Functions
		//String2CPFileInstructionType.put( "rmempty"	    , CPINSTRUCTION_TYPE.ParameterizedBuiltin);
		
		return -1; //should never come here.
	}
}
