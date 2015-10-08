/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.hops.cost;

import java.util.ArrayList;
import java.util.HashSet;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.MapMult;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheableData;
import com.ibm.bi.dml.runtime.controlprogram.caching.LazyWriteBuffer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.instructions.CPInstructionParser;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRInstructionParser;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.CPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.CPInstruction.CPINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.instructions.cp.FunctionCallCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.VariableCPInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.BinaryMRInstructionBase;
import com.ibm.bi.dml.runtime.instructions.mr.CM_N_COVInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.DataGenMRInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.GroupedAggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.IDistributedCacheConsumer;
import com.ibm.bi.dml.runtime.instructions.mr.MMTSJMRInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.MapMultChainInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.PickByCountInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.RemoveEmptyMRInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.TernaryInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.UnaryMRInstructionBase;
import com.ibm.bi.dml.runtime.instructions.mr.MRInstruction.MRINSTRUCTION_TYPE;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.mapred.DistributedCacheInput;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator;
import com.ibm.bi.dml.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import com.ibm.bi.dml.yarn.ropt.MRJobResourceInstruction;
import com.ibm.bi.dml.yarn.ropt.YarnClusterAnalyzer;

/**
 * 
 */
public class CostEstimatorStaticRuntime extends CostEstimator
{
	
	//time-conversion
	private static final long DEFAULT_FLOPS = 2L * 1024 * 1024 * 1024; //2GFLOPS
	//private static final long UNKNOWN_TIME = -1;
	
	//floating point operations
	private static final double DEFAULT_NFLOP_NOOP = 10; 
	private static final double DEFAULT_NFLOP_UNKNOWN = 1; 
	private static final double DEFAULT_NFLOP_CP = 1; 	
	private static final double DEFAULT_NFLOP_TEXT_IO = 350; 
	
	//MR job latency
	private static final double DEFAULT_MR_JOB_LATENCY_LOCAL = 2;
	private static final double DEFAULT_MR_JOB_LATENCY_REMOTE = 20;
	private static final double DEFAULT_MR_TASK_LATENCY_LOCAL = 0.001;
	private static final double DEFAULT_MR_TASK_LATENCY_REMOTE = 1.5;
	
	//IO READ throughput
	private static final double DEFAULT_MBS_FSREAD_BINARYBLOCK_DENSE = 200;
	private static final double DEFAULT_MBS_FSREAD_BINARYBLOCK_SPARSE = 100;
	private static final double DEFAULT_MBS_HDFSREAD_BINARYBLOCK_DENSE = 150;
	private static final double DEFAULT_MBS_HDFSREAD_BINARYBLOCK_SPARSE = 75;
	//IO WRITE throughput
	private static final double DEFAULT_MBS_FSWRITE_BINARYBLOCK_DENSE = 150;
	private static final double DEFAULT_MBS_FSWRITE_BINARYBLOCK_SPARSE = 75;
	private static final double DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_DENSE = 120;
	private static final double DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_SPARSE = 60;
	private static final double DEFAULT_MBS_HDFSWRITE_TEXT_DENSE = 40;
	private static final double DEFAULT_MBS_HDFSWRITE_TEXT_SPARSE = 30;
	
	@Override
	@SuppressWarnings("unused")
	protected double getCPInstTimeEstimate( Instruction inst, VarStats[] vs, String[] args ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		CPInstruction cpinst = (CPInstruction)inst;
		
		//load time into mem
		double ltime = 0;
		if( !vs[0]._inmem ){
			ltime += getHDFSReadTime( vs[0]._rlen, vs[0]._clen, vs[0].getSparsity() );
			//eviction costs
			if( CacheableData.CACHING_WRITE_CACHE_ON_READ &&
				LazyWriteBuffer.getWriteBufferSize()<MatrixBlock.estimateSizeOnDisk(vs[0]._rlen, vs[0]._clen, (long)((vs[0]._nnz<0)? vs[0]._rlen*vs[0]._clen:vs[0]._nnz)) )
			{
				ltime += Math.abs( getFSWriteTime( vs[0]._rlen, vs[0]._clen, vs[0].getSparsity() ));
			}
			vs[0]._inmem = true;
		}
		if( !vs[1]._inmem ){
			ltime += getHDFSReadTime( vs[1]._rlen, vs[1]._clen, vs[1].getSparsity() );
			//eviction costs
			if( CacheableData.CACHING_WRITE_CACHE_ON_READ &&
				LazyWriteBuffer.getWriteBufferSize()<MatrixBlock.estimateSizeOnDisk(vs[1]._rlen, vs[1]._clen, (long)((vs[1]._nnz<0)? vs[1]._rlen*vs[1]._clen:vs[1]._nnz)) )
			{
				ltime += Math.abs( getFSWriteTime( vs[1]._rlen, vs[1]._clen, vs[1].getSparsity()) );
			}
			vs[1]._inmem = true;
		}
		if( LOG.isDebugEnabled() && ltime!=0 ) {
			LOG.debug("Cost["+cpinst.getOpcode()+" - read] = "+ltime);
		}		
				
		//exec time CP instruction
		String opcode = (cpinst instanceof FunctionCallCPInstruction) ? InstructionUtils.getOpCode(cpinst.toString()) : cpinst.getOpcode();
		double etime = getInstTimeEstimate(opcode, vs, args, ExecType.CP);
		
		//write time caching
		double wtime = 0;
		//double wtime = getFSWriteTime( vs[2]._rlen, vs[2]._clen, (vs[2]._nnz<0)? 1.0:(double)vs[2]._nnz/vs[2]._rlen/vs[2]._clen );
		if( inst instanceof VariableCPInstruction && ((VariableCPInstruction)inst).getOpcode().equals("write") )
			wtime += getHDFSWriteTime(vs[2]._rlen, vs[2]._clen, vs[2].getSparsity(), ((VariableCPInstruction)inst).getInput3().getName() );
		
		if( LOG.isDebugEnabled() && wtime!=0 ) {
			LOG.debug("Cost["+cpinst.getOpcode()+" - write] = "+wtime);
		}
		
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
				        ConfigurationManager.getConfig().getIntValue(DMLConfig.NUM_REDUCERS) );
		double blocksize = ((double)InfrastructureAnalyzer.getHDFSBlockSize())/(1024*1024);
		
		//correction max number of mappers/reducers on yarn clusters
		if( InfrastructureAnalyzer.isYarnEnabled() ) {
			maxPMap = (int)Math.max( maxPMap, YarnClusterAnalyzer.getNumCores() );
			//artificially reduced by factor 2, in order to prefer map-side processing even if smaller degree of parallelism
			maxPRed = (int)Math.max( maxPRed, YarnClusterAnalyzer.getNumCores()/2 /2 );
		}
				
		//yarn-specific: take degree of parallelism into account
		if( jinst instanceof MRJobResourceInstruction ){
			int maxTasks = (int)((MRJobResourceInstruction)jinst).getMaxMRTasks();
			maxPMap = Math.min(maxPMap, maxTasks);
			maxPRed = Math.min(maxPRed, maxTasks);
		}
		
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
		int numEPMap = Math.max(Math.min(numMap, maxPMap/2),1); //effective map dop
		int numRed = computeNumReduceTasks( vs, mapOutIx, jinst.getJobType() );
		int numPRed = Math.min(numRed, maxPRed);
		int numEPRed = Math.max(Math.min(numRed, maxPRed/2),1); //effective reduce dop
				
		LOG.debug("Meta nmap = "+numMap+", nred = "+numRed+"; npmap = "+numPMap+", npred = "+numPRed+"; nepmap = "+numEPMap+", nepred = "+numEPRed);
	
		//step 0: export if inputs in mem
		double exportCosts = 0; 
		for( int i=0; i<jinst.getInputVars().length; i++ )
			if( vs[i]._inmem )
				exportCosts += getHDFSWriteTime(vs[i]._rlen, vs[i]._clen, vs[i].getSparsity());
		
		//step 1: MR job / task latency (normalization by effective dop)
		double jobLatencyCosts = localJob ? DEFAULT_MR_JOB_LATENCY_LOCAL : DEFAULT_MR_JOB_LATENCY_REMOTE;
		double taskLatencyCost = (numMap / numEPMap + numEPRed)
				               * (localJob ? DEFAULT_MR_TASK_LATENCY_LOCAL : DEFAULT_MR_TASK_LATENCY_REMOTE);	
		double latencyCosts = jobLatencyCosts + taskLatencyCost;
		
		//step 2: parallel read of inputs (normalization by effective dop)
		double hdfsReadCosts = 0;
		for( int i=0; i<jinst.getInputVars().length; i++ )
			hdfsReadCosts += getHDFSReadTime(vs[i]._rlen, vs[i]._clen, vs[i].getSparsity()); 
		 hdfsReadCosts /= numEPMap;
		
		//step 3: parallel MR instructions
		String[] mapperInst = new String[]{rdInst, rrInst, mapInst};
		String[] reducerInst = new String[]{shfInst, aggInst, otherInst};	
		
		//map instructions compute/distcache read (normalization by effective dop) 
		double mapDCReadCost = 0; //read through distributed cache
		double mapCosts = 0; //map compute cost
		double shuffleCosts = 0; 
		double reduceCosts = 0; //reduce compute costs
		
		for( String instCat : mapperInst )
			if( instCat != null && instCat.length()>0 ) {
				String[] linst = instCat.split( Lop.INSTRUCTION_DELIMITOR );
				for( String tmp : linst ){
					//map compute costs
					Object[] o = extractMRInstStatistics(tmp, vs);
					String opcode = InstructionUtils.getOpCode(tmp);
					mapCosts += getInstTimeEstimate(opcode, (VarStats[])o[0], (String[])o[1], ExecType.MR);
					//dist cache read costs
					int dcIndex = getDistcacheIndex(tmp);
					if( dcIndex >= 0 ) {
						mapDCReadCost += Math.min(getFSReadTime(vs[dcIndex]._rlen, vs[dcIndex]._clen, vs[dcIndex].getSparsity()),
								                  getFSReadTime(DistributedCacheInput.PARTITION_SIZE, 1, 1.0)) //32MB partitions
								         * numMap; //read in each task
					}
				}
			}
		mapCosts /= numEPMap;
		mapDCReadCost /= numEPMap;
		
		if( !mapOnly )
		{
			//shuffle costs (normalization by effective map/reduce dop)
			for( int i=0; i<mapOutIx.length; i++ )
			{
				shuffleCosts += ( getFSWriteTime(vs[mapOutIx[i]]._rlen, vs[mapOutIx[i]]._clen, vs[mapOutIx[i]].getSparsity()) / numEPMap
				                 + getFSWriteTime(vs[mapOutIx[i]]._rlen, vs[mapOutIx[i]]._clen, vs[mapOutIx[i]].getSparsity())*4 / numEPRed
						         + getFSReadTime(vs[mapOutIx[i]]._rlen, vs[mapOutIx[i]]._clen, vs[mapOutIx[i]].getSparsity()) / numEPRed); 	
			
				//correction of shuffle costs (necessary because the above shuffle does not consider the number of blocks)
				//TODO this is a workaround - we need to address the number of map output blocks in a more systematic way
				for( String instCat : reducerInst )
					if( instCat != null && instCat.length()>0 ) {
						String[] linst = instCat.split( Lop.INSTRUCTION_DELIMITOR );
						for( String tmp : linst ) {
							if(InstructionUtils.getMRType(tmp)==MRINSTRUCTION_TYPE.Aggregate)
								shuffleCosts += numMap * getFSWriteTime(vs[mapOutIx[i]]._rlen, vs[mapOutIx[i]]._clen, vs[mapOutIx[i]].getSparsity()) / numEPMap
										      + numPMap * getFSWriteTime(vs[mapOutIx[i]]._rlen, vs[mapOutIx[i]]._clen, vs[mapOutIx[i]].getSparsity()) / numEPMap
										      + numPMap * getFSReadTime(vs[mapOutIx[i]]._rlen, vs[mapOutIx[i]]._clen, vs[mapOutIx[i]].getSparsity()) / numEPRed;
						}
					}
			}
						
			//reduce instructions compute (normalization by effective dop)
			for( String instCat : reducerInst )
				if( instCat != null && instCat.length()>0 ) {
					String[] linst = instCat.split( Lop.INSTRUCTION_DELIMITOR );
					for( String tmp : linst ){
						Object[] o = extractMRInstStatistics(tmp, vs);
						if(InstructionUtils.getMRType(tmp)==MRINSTRUCTION_TYPE.Aggregate)
							o[1] = new String[]{String.valueOf(numMap)};
						String opcode = InstructionUtils.getOpCode(tmp);
						reduceCosts += getInstTimeEstimate(opcode, (VarStats[])o[0], (String[])o[1], ExecType.MR);
					}
				}
			reduceCosts /= numEPRed;
		}		
		
		//step 4: parallel write of outputs (normalization by effective dop)
		double hdfsWriteCosts = 0;
		for( int i=0; i<jinst.getOutputVars().length; i++ )
		{
			hdfsWriteCosts += getHDFSWriteTime(vs[retIx[i]]._rlen, vs[retIx[i]]._clen, vs[retIx[i]].getSparsity()); 
		}
		hdfsWriteCosts /= ((mapOnly)? numEPMap : numEPRed);
		
		//debug output
		if( LOG.isDebugEnabled() ) {
			LOG.debug("Costs Export = "+exportCosts);
			LOG.debug("Costs Latency = "+latencyCosts);
			LOG.debug("Costs HDFS Read = "+hdfsReadCosts);
			LOG.debug("Costs Distcache Read = "+mapDCReadCost);
			LOG.debug("Costs Map Exec = "+mapCosts);
			LOG.debug("Costs Shuffle = "+shuffleCosts);
			LOG.debug("Costs Reduce Exec = "+reduceCosts);
			LOG.debug("Costs HDFS Write = "+hdfsWriteCosts);
		}
	
		//aggregate individual cost factors
		return exportCosts + latencyCosts + 
			   hdfsReadCosts + mapCosts + mapDCReadCost + 
			   shuffleCosts +  
		       reduceCosts + hdfsWriteCosts; 
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
		
		
		if( opcode.equals(DataGen.RAND_OPCODE) )
		{
			vs[0] = _unknownStats;
			vs[1] = _unknownStats;
			vs[2] = stats[Integer.parseInt(parts[2])];
			
			int type = 2; 
			//awareness of instruction patching min/max
			if(    !parts[7].contains(Lop.VARIABLE_NAME_PLACEHOLDER) 
				&& !parts[8].contains(Lop.VARIABLE_NAME_PLACEHOLDER) )
			{
				double minValue = Double.parseDouble(parts[7]);
				double maxValue = Double.parseDouble(parts[8]);
				double sparsity = Double.parseDouble(parts[9]);
				if( minValue == 0.0 && maxValue == 0.0 )
					type = 0;
				else if( sparsity == 1.0 && minValue == maxValue )
					type = 1;
			}
			attr = new String[]{String.valueOf(type)};
		}	
		if( opcode.equals(DataGen.SEQ_OPCODE) )
		{
			vs[0] = _unknownStats;
			vs[1] = _unknownStats;
			vs[2] = stats[Integer.parseInt(parts[2])];
		}	
		else //general case
		{
			
			String inst2 = replaceInstructionPatch( inst );
			MRInstruction mrinst = MRInstructionParser.parseSingleInstruction(inst2);
			
			if( mrinst instanceof UnaryMRInstructionBase )
			{
				UnaryMRInstructionBase uinst = (UnaryMRInstructionBase) mrinst;
				vs[0] = uinst.input>=0 ? stats[ uinst.input ] : _unknownStats;
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
					{
						AggregateOperationTypes type = CMOperator.getAggOpType(parts[2], parts[3]);
						attr = new String[]{String.valueOf(type.ordinal())};
					}
					
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
				
				if( opcode.equals("rmempty") ) {
					RemoveEmptyMRInstruction rbinst = (RemoveEmptyMRInstruction) mrinst;
					attr = new String[]{rbinst.isRemoveRows()?"0":"1"};
				}
			}
			else if( mrinst instanceof TernaryInstruction )
			{
				TernaryInstruction tinst = (TernaryInstruction) mrinst;
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
			else if( mrinst instanceof MapMultChainInstruction)
			{
				MapMultChainInstruction minst = (MapMultChainInstruction) mrinst;
				vs[0] = stats[ minst.getInput1() ];
				vs[1] = stats[ minst.getInput2() ];
				if( minst.getInput3()>=0 )
					vs[2] = stats[ minst.getInput3() ];
				
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
			DataGenMRInstruction[] ins = MRInstructionParser.parseDataGenInstructions(rdInst);
			for( DataGenMRInstruction inst : ins )
				for( byte ix : inst.getAllIndexes() )
					ixMap.add(ix);
		}
		
		if( mapInst!=null && mapInst.length()>0 ) {
			mapInst = replaceInstructionPatch(mapInst);
			MRInstruction[] ins = MRInstructionParser.parseMixedInstructions(mapInst);
			for( MRInstruction inst : ins )
				for( byte ix : inst.getAllIndexes() )
					ixMap.add(ix);
		}
		
		//reduce indices
		HashSet<Byte> ixRed = new HashSet<Byte>();
		for( byte ix : retIx )
			ixRed.add(ix);
	

		if( shfInst!=null && shfInst.length()>0 ) {
			shfInst = replaceInstructionPatch(shfInst);
			MRInstruction[] ins = MRInstructionParser.parseMixedInstructions(shfInst);
			for( MRInstruction inst : ins )
				for( byte ix : inst.getAllIndexes() )
					ixRed.add(ix);
		}
		
		if( aggInst!=null && aggInst.length()>0 ) {
			aggInst = replaceInstructionPatch(aggInst);
			MRInstruction[] ins = MRInstructionParser.parseAggregateInstructions(aggInst);
			for( MRInstruction inst : ins )
				for( byte ix : inst.getAllIndexes() )
					ixRed.add(ix);
		}
		
		if( otherInst!=null && otherInst.length()>0 ) {
			otherInst = replaceInstructionPatch(otherInst);
			MRInstruction[] ins = MRInstructionParser.parseMixedInstructions(otherInst);
			for( MRInstruction inst : ins )
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
		if( jobtype == JobType.DATAGEN )
			return maxPMap;
			
		//input size, num blocks
		double mapInputSize = 0;
		int numBlocks = 0;
		for( int i=0; i<inputIx.length; i++ )
		{
			//input size
			mapInputSize += ((double)MatrixBlock.estimateSizeOnDisk((long)vs[inputIx[i]]._rlen, (long)vs[inputIx[i]]._clen, (long)vs[inputIx[i]]._nnz)) / (1024*1024);	
		
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

		switch( jobtype )
		{
			case REBLOCK:
			case CSV_REBLOCK: {
				for( int i=0; i<mapOutIx.length; i++ )
				{
					int lret =  (int) Math.ceil((double)vs[mapOutIx[i]]._rlen/vs[mapOutIx[i]]._brlen)
					           *(int) Math.ceil((double)vs[mapOutIx[i]]._clen/vs[mapOutIx[i]]._bclen);
					ret = Math.max(lret, ret);
				}		
				break;
			}
			
			default: {
				for( int i=0; i<mapOutIx.length; i++ )
				{
					int lret =  (int) Math.ceil((double)vs[mapOutIx[i]]._rlen/DMLTranslator.DMLBlockSize)
					           *(int) Math.ceil((double)vs[mapOutIx[i]]._clen/DMLTranslator.DMLBlockSize);
					ret = Math.max(lret, ret);
				}
				break;
			}
		}
		
		return Math.max(1, ret);
	}

	/**
	 * 
	 * @param inst
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	private int getDistcacheIndex(String inst) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		ArrayList<Byte> indexes = new ArrayList<Byte>();
		
		if( InstructionUtils.isDistributedCacheUsed(inst) ) {
			MRInstruction mrinst = MRInstructionParser.parseSingleInstruction(inst);
			if( mrinst instanceof IDistributedCacheConsumer )
				((IDistributedCacheConsumer)mrinst).addDistCacheIndex(inst, indexes);
		}
		
		if( !indexes.isEmpty() )
			return indexes.get(0);
		else
			return -1;
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
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		double ret = ((double)MatrixBlock.estimateSizeOnDisk((long)dm, (long)dn, (long)(ds*dm*dn))) / (1024*1024);  		
		
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
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		
		double bytes = (double)MatrixBlock.estimateSizeOnDisk((long)dm, (long)dn, (long)(ds*dm*dn));
		double mbytes = bytes / (1024*1024);  		
		
		double ret = -1;
		if( sparse )
			ret = mbytes / DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_SPARSE;
		else //dense
			ret = mbytes / DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_DENSE;
		
		//if( LOG.isDebugEnabled() )
		//	LOG.debug("Costs[export] = "+ret+"s, "+mbytes+" MB ("+dm+","+dn+","+ds+").");
		
		
		return ret;
	}
	
	private double getHDFSWriteTime( long dm, long dn, double ds, String format )
	{
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		
		double bytes = (double)MatrixBlock.estimateSizeOnDisk((long)dm, (long)dn, (long)(ds*dm*dn));
		double mbytes = bytes / (1024*1024);  		
		
		double ret = -1;
		
		if( format.equals("textcell") || format.equals("csv") )
		{
			if( sparse )
				ret = mbytes / DEFAULT_MBS_HDFSWRITE_TEXT_SPARSE;
			else //dense
				ret = mbytes / DEFAULT_MBS_HDFSWRITE_TEXT_DENSE;	
			ret *= 2.75; //text commonly 2x-3.5x larger than binary
		}
		else
		{
			if( sparse )
				ret = mbytes / DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_SPARSE;
			else //dense
				ret = mbytes / DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_DENSE;
		}
		//if( LOG.isDebugEnabled() )
		//	LOG.debug("Costs[export] = "+ret+"s, "+mbytes+" MB ("+dm+","+dn+","+ds+").");
		
		
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
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		
		double ret = ((double)MatrixBlock.estimateSizeOnDisk((long)dm, (long)dn, (long)(ds*dm*dn))) / (1024*1024);  		
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
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		
		double ret = ((double)MatrixBlock.estimateSizeOnDisk((long)dm, (long)dn, (long)(ds*dm*dn))) / (1024*1024);  		
		
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
	private double getInstTimeEstimate(String opcode, VarStats[] vs, String[] args, ExecType et) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		boolean inMR = (et == ExecType.MR);
		return getInstTimeEstimate(opcode, inMR,  
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
	private double getInstTimeEstimate( String opcode, boolean inMR, long d1m, long d1n, double d1s, long d2m, long d2n, double d2s, long d3m, long d3n, double d3s, String[] args ) throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		double nflops = getNFLOP(opcode, inMR, d1m, d1n, d1s, d2m, d2n, d2s, d3m, d3n, d3s, args);
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
	private double getNFLOP( String optype, boolean inMR, long d1m, long d1n, double d1s, long d2m, long d2n, double d2s, long d3m, long d3n, double d3s, String[] args ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//operation costs in FLOP on matrix block level (for CP and MR instructions)
		//(excludes IO and parallelism; assumes known dims for all inputs, outputs )
	
		boolean leftSparse = MatrixBlock.evalSparseFormatInMemory(d1m, d1n, (long)(d1s*d1m*d1n));
		boolean rightSparse = MatrixBlock.evalSparseFormatInMemory(d2m, d2n, (long)(d2s*d2m*d2n));
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
						//reduction by factor 2 because matrix mult better than
						//average flop count
						if( !leftSparse && !rightSparse )
							return 2 * (d1m * d1n * ((d2n>1)?d1s:1.0) * d2n) /2;
						else if( !leftSparse && rightSparse )
							return 2 * (d1m * d1n * d1s * d2n * d2s) /2;
						else if( leftSparse && !rightSparse )
							return 2 * (d1m * d1n * d1s * d2n) /2;
						else //leftSparse && rightSparse
							return 2 * (d1m * d1n * d1s * d2n * d2s) /2;
					}
					else if( optype.equals("cov") ) {
						//note: output always scalar, d3 used as weights block
						//if( allExists ), same runtime for 2 and 3 inputs
						return 23 * d1m; //(11+3*k+)
					}
					
					return 0;
				
				case MMChain:
					//reduction by factor 2 because matrix mult better than average flop count
					//(mmchain essentially two matrix-vector muliplications)
					if( !leftSparse  )
						return (2+2) * (d1m * d1n) /2;
					else 
						return (2+2) * (d1m * d1n * d1s) /2;
					
				case AggregateTernary: //opcodes: tak+*
					return 6 * d1m * d1n; //2*1(*) + 4 (k+)
					
				case AggregateUnary: //opcodes: uak+, uark+, uack+, uamean, uarmean, uacmean, 
									 //         uamax, uarmax, uarimax, uacmax, uamin, uarmin, uacmin, 
									 //         ua+, uar+, uac+, ua*, uatrace, uaktrace, 
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
				    
				case ArithmeticBinary: //opcodes: +, -, *, /, ^ (incl. ^2, *2)
					//note: covers scalar-scalar, scalar-matrix, matrix-matrix
					if( optype.equals("+") || optype.equals("-") //sparse safe
						&& ( leftSparse || rightSparse ) )
						return d1m*d1n*d1s + d2m*d2n*d2s;
					else
						return d3m*d3n;
					
				case Ternary: //opcodes: ctable
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
					
				case BuiltinBinary: //opcodes: max, min, solve
					//note: covers scalar-scalar, scalar-matrix, matrix-matrix
					if( optype.equals("solve") ) //see also MultiReturnBuiltin
						return d1m * d1n * d1n; //for 1kx1k ~ 1GFLOP -> 0.5s
					else //default
						return d3m * d3n;

					
				case BuiltinUnary: //opcodes: exp, abs, sin, cos, tan, sqrt, plogp, print, round, sprop, sigmoid
					if( optype.equals("print") ) //scalar only
						return 1;
					else
					{
						double xbu = 1; //default for all ops
						if( optype.equals("plogp") ) xbu = 2;
						else if( optype.equals("round") ) xbu = 4;
						
						if( optype.equals("sin") || optype.equals("tan") || optype.equals("round")
							|| optype.equals("abs") || optype.equals("sqrt") || optype.equals("sprop")
							|| optype.equals("sigmoid") ) //sparse-safe
						{
							if( leftSparse ) //sparse
								return xbu * d1m * d1n * d1s;	
							else //dense
								return xbu * d1m * d1n;
						}
						else
							return xbu * d1m * d1n;
					}
										
				case Reorg: //opcodes: r', rdiag
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
						boolean text = args[0].equals("textcell") || args[0].equals("csv");
						double xwrite =  text ? DEFAULT_NFLOP_TEXT_IO : DEFAULT_NFLOP_CP;
						
						if( !leftSparse )
							return d1m * d1n * xwrite; 
						else
							return d1m * d1n * d1s * xwrite;
					}
					else if ( optype.equals("inmem-iqm") )
						//note: assumes uniform distribution
						return 2 * d1m + //sum of weights
						       5 + 0.25d * d1m + //scan to lower quantile
						       8 * 0.5 * d1m; //scan from lower to upper quantile
					else
						return DEFAULT_NFLOP_NOOP;
			
				case Rand: //opcodes: rand, seq
					if( optype.equals(DataGen.RAND_OPCODE) ){
						int nflopRand = 32; //per random number
						switch(Integer.parseInt(args[0])) {
							case 0: return DEFAULT_NFLOP_NOOP; //empty matrix
							case 1: return d3m * d3n * 8; //allocate, arrayfill
							case 2: //full rand
							{
								if( d3s==1.0 )
									return d3m * d3n * nflopRand + d3m * d3n * 8; //DENSE gen (incl allocate)    
								else 
									return (d3s>=MatrixBlock.SPARSITY_TURN_POINT)? 
										    2 * d3m * d3n * nflopRand + d3m * d3n * 8: //DENSE gen (incl allocate)    
									        3 * d3m * d3n * d3s * nflopRand + d3m * d3n * d3s * 24; //SPARSE gen (incl allocate)
							}
						}
					}
					else //seq
						return d3m * d3n * DEFAULT_NFLOP_CP;
				
				case StringInit: //sinit
					return d3m * d3n * DEFAULT_NFLOP_CP;
					
				case External: //opcodes: extfunct
					//note: should be invoked independently for multiple outputs
					return d1m * d1n * d1s * DEFAULT_NFLOP_UNKNOWN;
				
				case MultiReturnBuiltin: //opcodes: qr, lu, eigen
					//note: they all have cubic complexity, the scaling factor refers to commons.math
					double xf = 2; //default e.g, qr
					if( optype.equals("eigen") ) 
						xf = 32;
					else if ( optype.equals("lu") )
						xf = 16;
					return xf * d1m * d1n * d1n; //for 1kx1k ~ 2GFLOP -> 1s
					
				case ParameterizedBuiltin: //opcodes: cdf, invcdf, groupedagg, rmempty
					if( optype.equals("cdf") || optype.equals("invcdf"))
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
					
				case QSort: //opcodes: sort
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
					//reduction by factor 2 because matrix mult better than
					//average flop count
					if( MMTSJType.valueOf(args[0]).isLeft() ) { //lefttranspose
						if( !rightSparse ) //dense						
							return d1m * d1n * d1s * d1n /2;
						else //sparse
							return d1m * d1n * d1s * d1n * d1s /2; 
					}
					else if(onlyLeft) { //righttranspose
						if( !leftSparse ) //dense
							return (double)d1m * d1n * d1m /2;
						else //sparse
							return   d1m * d1n * d1s //reorg sparse
							       + d1m * d1n * d1s * d1n * d1s /2; //core tsmm
					}					
					return 0;
				
				case Partition:
					return d1m * d1n * d1s + //partitioning costs
						   (inMR ? 0 : //include write cost if in CP  	
							getHDFSWriteTime(d1m, d1n, d1s)* DEFAULT_FLOPS);
					
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
					
				case AggregateBinary: //opcodes: cpmm, rmm, mapmult
					//note: copy from CP costs
					if(    optype.equals("cpmm") || optype.equals("rmm") 
						|| optype.equals(MapMult.OPCODE) ) //matrix mult
					{
						//reduction by factor 2 because matrix mult better than
						//average flop count
						if( !leftSparse && !rightSparse )
							return 2 * (d1m * d1n * ((d2n>1)?d1s:1.0) * d2n) /2;
						else if( !leftSparse && rightSparse )
							return 2 * (d1m * d1n * d1s * d2n * d2s) /2;
						else if( leftSparse && !rightSparse )
							return 2 * (d1m * d1n * d1s * d2n) /2;
						else //leftSparse && rightSparse
							return 2 * (d1m * d1n * d1s * d2n * d2s) /2;
					}
					return 0;
					
				case MapMultChain: //opcodes: mapmultchain	
					//assume dense input2 and input3
					return   2 * d1m * d2n * d1n * ((d2n>1)?d1s:1.0) //ba(+*) 
						   + d1m * d2n //cellwise b(*) 
					       + d1m * d2n //r(t)
					       + 2 * d2n * d1n * d1m * (leftSparse?d1s:1.0) //ba(+*)
					       + d2n * d1n; //r(t)
					
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
					
				case CombineTernary: //opcodes: combinetertiary
					return   d1m * d1n * d1s
				           + d2m * d2n * d2s
				           + d3m * d3n * d3s;
					
				case Unary: //opcodes: log, slog, pow 			
					//TODO requires opcode consolidation (builtin, arithmic)
					//note: covers scalar, matrix, matrix-scalar
					return d3m * d3n;
					
				case Ternary: //opcodes: ctabletransform, ctabletransformscalarweight, ctabletransformhistogram, ctabletransformweightedhistogram
					//note: copy from cp
					if( leftSparse )
						return d1m * d1n * d1s; //add
					else 
						return d1m * d1n;
			
				case Quaternary:
					//TODO pattern specific and all 4 inputs requires
					return d1m * d1n * d1s *4;
					
				case Reblock: //opcodes: rblk
					return DEFAULT_NFLOP_CP * ((leftSparse)? d1m*d1n*d1s : d1m*d1n); 
					
				case Replicate: //opcodes: rblk
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
					
				default:
					return 0;
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
