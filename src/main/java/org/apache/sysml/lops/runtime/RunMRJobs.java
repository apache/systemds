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

package org.apache.sysml.lops.runtime;

import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.wink.json4j.JSONException;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.mqo.RuntimePiggybacking;
import org.apache.sysml.runtime.instructions.MRInstructionParser;
import org.apache.sysml.runtime.instructions.MRJobInstruction;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.instructions.mr.DataGenMRInstruction;
import org.apache.sysml.runtime.instructions.mr.RandInstruction;
import org.apache.sysml.runtime.instructions.mr.ReblockInstruction;
import org.apache.sysml.runtime.instructions.mr.SeqInstruction;
import org.apache.sysml.runtime.io.MatrixWriter;
import org.apache.sysml.runtime.io.MatrixWriterFactory;
import org.apache.sysml.runtime.matrix.CMCOVMR;
import org.apache.sysml.runtime.matrix.CSVReblockMR;
import org.apache.sysml.runtime.matrix.CombineMR;
import org.apache.sysml.runtime.matrix.DataGenMR;
import org.apache.sysml.runtime.matrix.DataPartitionMR;
import org.apache.sysml.runtime.matrix.GMR;
import org.apache.sysml.runtime.matrix.GroupedAggMR;
import org.apache.sysml.runtime.matrix.JobReturn;
import org.apache.sysml.runtime.matrix.MMCJMR;
import org.apache.sysml.runtime.matrix.MMRJMR;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixDimensionsMetaData;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.ReblockMR;
import org.apache.sysml.runtime.matrix.SortMR;
import org.apache.sysml.runtime.matrix.WriteCSVMR;
import org.apache.sysml.runtime.matrix.data.LibMatrixDatagen;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.data.RandomMatrixGenerator;
import org.apache.sysml.runtime.transform.DataTransform;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.utils.Statistics;


public class RunMRJobs 
{

	/**
	 * Wrapper for submitting MR job instructions incl preparation and actual submission.
	 * The preparation includes (1) pulling stats out of symbol table and populating the
	 * instruction, (2) instruction patching, and (3) export of in-memory matrices if 
	 * required. 
	 * 
	 * Furthermore, this wrapper also provides a hook for runtime piggybacking to intercept
	 * concurrent job submissions in order to collect and merge instructions.   
	 * 
	 * @param inst
	 * @param ec
	 * @return
	 */

	public static JobReturn prepareAndSubmitJob( MRJobInstruction inst, ExecutionContext ec )
		throws DMLRuntimeException 
	{
		// Obtain references to all input matrices 
		MatrixObject[] inputMatrices = inst.extractInputMatrices(ec);
		
		// export dirty matrices to HDFS
		// note: for REBLOCK postponed until we know if necessary
		if( !(inst.getJobType() == JobType.REBLOCK) )
		{
			//export matrices
			for(MatrixObject m : inputMatrices) {
				if ( m.isDirty() || m.getRDDHandle()!=null )
					m.exportData();
			}

			//check input files
			checkEmptyInputs( inst, inputMatrices );	
		}
		
		// Obtain references to all output matrices
		inst.extractOutputMatrices(ec);
	
		// obtain original state
		String rdInst = inst.getIv_randInstructions();
		String rrInst = inst.getIv_recordReaderInstructions();
		String mapInst = inst.getIv_instructionsInMapper();
		String shuffleInst = inst.getIv_shuffleInstructions();
		String aggInst = inst.getIv_aggInstructions();
		String otherInst = inst.getIv_otherInstructions();			
					
		// variable patching (replace placeholders with variables)
		inst.setIv_randInstructions(updateLabels(rdInst, ec.getVariables()));
		inst.setIv_recordReaderInstructions(updateLabels(rrInst, ec.getVariables()));
		inst.setIv_instructionsInMapper(updateLabels(mapInst, ec.getVariables()));
		inst.setIv_shuffleInstructions(updateLabels(shuffleInst, ec.getVariables()));
		inst.setIv_aggInstructions(updateLabels(aggInst, ec.getVariables()));
		inst.setIv_otherInstructions(updateLabels(otherInst, ec.getVariables()));
		
		// runtime piggybacking if applicable
		JobReturn ret = null;
		if(   OptimizerUtils.ALLOW_RUNTIME_PIGGYBACKING
			&& RuntimePiggybacking.isActive()
			&& RuntimePiggybacking.isSupportedJobType(inst.getJobType()) )
		{
			ret = RuntimePiggybacking.submitJob( inst );
		}
		else
			ret = submitJob( inst );
		
		// reset original state
		inst.setIv_randInstructions(rdInst);
		inst.setIv_recordReaderInstructions(rrInst);
		inst.setIv_instructionsInMapper(mapInst);
		inst.setIv_shuffleInstructions(shuffleInst);
		inst.setIv_aggInstructions(aggInst);
		inst.setIv_otherInstructions(otherInst);
		
		return ret;
	}
	
	/**
	 * Submits an MR job instruction, without modifying any state of that instruction.
	 * 
	 * @param inst
	 * @param ec
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static JobReturn submitJob(MRJobInstruction inst ) 
		throws DMLRuntimeException 
	{
		JobReturn ret = new JobReturn();		
		MatrixObject[] inputMatrices = inst.getInputMatrices();
		MatrixObject[] outputMatrices = inst.getOutputMatrices();
		boolean execCP = false;
		
		// Spawn MapReduce Jobs
		try {
			// replace all placeholders in all instructions with appropriate values
			String rdInst = inst.getIv_randInstructions();
			String rrInst = inst.getIv_recordReaderInstructions();
			String mapInst = inst.getIv_instructionsInMapper();
			String shuffleInst = inst.getIv_shuffleInstructions();
			String aggInst = inst.getIv_aggInstructions();
			String otherInst = inst.getIv_otherInstructions();
			boolean jvmReuse = ConfigurationManager.getConfig().getBooleanValue(DMLConfig.JVM_REUSE);
			
			switch(inst.getJobType()) {
			
			case GMR: 
			case GMRCELL:
				ret = GMR.runJob(inst, inst.getInputs(), inst.getInputInfos(), 
						inst.getRlens(), inst.getClens(), inst.getBrlens(), inst.getBclens(),
						inst.getPartitioned(), inst.getPformats(), inst.getPsizes(),
						rrInst, mapInst, aggInst, otherInst,
						inst.getIv_numReducers(), inst.getIv_replication(), jvmReuse, inst.getIv_resultIndices(), inst.getDimsUnknownFilePrefix(),
						inst.getOutputs(), inst.getOutputInfos() );
				 break;

			case DATAGEN:
				if(    OptimizerUtils.ALLOW_DYN_RECOMPILATION
					&& OptimizerUtils.ALLOW_RAND_JOB_RECOMPILE
					&& DMLScript.rtplatform != RUNTIME_PLATFORM.HADOOP 
					&& Recompiler.checkCPDataGen( inst, rdInst ) ) 
				{
					ret = executeInMemoryDataGenOperations(inst, rdInst, outputMatrices);
					Statistics.decrementNoOfExecutedMRJobs();
					execCP = true;
				}
				else 
				{
					ret = DataGenMR.runJob(inst, 
							rdInst.split(Lop.INSTRUCTION_DELIMITOR), mapInst, aggInst, otherInst, 
							inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(), inst.getDimsUnknownFilePrefix(),
							inst.getOutputs(), inst.getOutputInfos());
				}
				break;
			
			case CM_COV:
				ret = CMCOVMR.runJob(inst, inst.getInputs(),  inst.getInputInfos(), 
						inst.getRlens(), inst.getClens(), inst.getBrlens(), inst.getBclens(),
						mapInst, shuffleInst, 
						inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(), 
						inst.getOutputs(), inst.getOutputInfos() );
				break;
			
			case GROUPED_AGG:
				ret = GroupedAggMR.runJob(inst, inst.getInputs(),  inst.getInputInfos(), 
						inst.getRlens(), inst.getClens(), inst.getBrlens(), inst.getBclens(),
						shuffleInst, otherInst, 
						inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(), inst.getDimsUnknownFilePrefix(),  
						inst.getOutputs(), inst.getOutputInfos() );
				break;
			
			case REBLOCK:
			case CSV_REBLOCK:
				if(    OptimizerUtils.ALLOW_DYN_RECOMPILATION 
					&& DMLScript.rtplatform != RUNTIME_PLATFORM.HADOOP 
					&& Recompiler.checkCPReblock( inst, inputMatrices ) ) 
				{
					ret = executeInMemoryReblockOperations(inst, shuffleInst, inputMatrices, outputMatrices);
					Statistics.decrementNoOfExecutedMRJobs();
					execCP = true;
				}
				else 
				{
					// export dirty matrices to HDFS (initially deferred)
					for(MatrixObject m : inputMatrices) {
						if ( m.isDirty() )
							m.exportData();
					}
					checkEmptyInputs( inst, inputMatrices );
					
					if ( inst.getJobType() == JobType.REBLOCK ) {
						ret = ReblockMR.runJob(inst, inst.getInputs(),  inst.getInputInfos(), 
								inst.getRlens(), inst.getClens(), inst.getBrlens(), inst.getBclens(), getNNZ(inputMatrices),
								mapInst, shuffleInst, otherInst,
								inst.getIv_numReducers(), inst.getIv_replication(), jvmReuse, inst.getIv_resultIndices(),   
								inst.getOutputs(), inst.getOutputInfos() );
					}
					else if( inst.getJobType() == JobType.CSV_REBLOCK ) {
						ret = CSVReblockMR.runJob(inst, inst.getInputs(), inst.getInputInfos(), 
								inst.getRlens(), inst.getClens(), inst.getBrlens(), inst.getBclens(),
								shuffleInst, otherInst,
								inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(),   
								inst.getOutputs(), inst.getOutputInfos() );
					}
				}
				break;

			case CSV_WRITE:
				ret = WriteCSVMR.runJob(inst, inst.getInputs(), inst.getInputInfos(), 
						inst.getRlens(), inst.getClens(), inst.getBclens(), inst.getBclens(), shuffleInst,
						inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(), inst.getOutputs());
				break;
				
			case MMCJ:
				ret = MMCJMR.runJob(inst, inst.getInputs(),  inst.getInputInfos(), 
						inst.getRlens(), inst.getClens(), inst.getBrlens(), inst.getBclens(),
						mapInst, aggInst, shuffleInst,
						inst.getIv_numReducers(), inst.getIv_replication(),    
						inst.getOutputs()[0], inst.getOutputInfos()[0] );
				break;

			case MMRJ:
				ret = MMRJMR.runJob(inst, inst.getInputs(),  inst.getInputInfos(), 
						inst.getRlens(), inst.getClens(), inst.getBrlens(), inst.getBclens(),
						mapInst, aggInst, shuffleInst, otherInst,
						inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(),    
						inst.getOutputs(), inst.getOutputInfos() );
				break;

			case SORT:
				boolean weightsflag = true;
				if ( !mapInst.equalsIgnoreCase("") )
					weightsflag = false;
				ret = SortMR.runJob(inst, inst.getInputs()[0],  inst.getInputInfos()[0], 
						inst.getRlens()[0], inst.getClens()[0], inst.getBrlens()[0], inst.getBclens()[0],
						mapInst, shuffleInst,
						inst.getIv_numReducers(), inst.getIv_replication(),    
						inst.getOutputs()[0], inst.getOutputInfos()[0], weightsflag );
				break;

			case COMBINE:
				ret = CombineMR.runJob(inst, inst.getInputs(),  inst.getInputInfos(), 
						inst.getRlens(), inst.getClens(), inst.getBrlens(), inst.getBclens(),
						shuffleInst, 
						inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(),    
						inst.getOutputs(), inst.getOutputInfos() );
				break;
			
			case DATA_PARTITION:
				ret = DataPartitionMR.runJob(inst, inputMatrices, shuffleInst, inst.getIv_resultIndices(), outputMatrices, inst.getIv_numReducers(), inst.getIv_replication());
				break;
				
			case TRANSFORM:
				
				if(    OptimizerUtils.ALLOW_DYN_RECOMPILATION
						&& OptimizerUtils.ALLOW_TRANSFORM_RECOMPILE
						&& DMLScript.rtplatform != RUNTIME_PLATFORM.HADOOP 
						&& Recompiler.checkCPTransform( inst, inputMatrices ) ) 
					{
						// transform the data and generate output in CSV format
						ret = executeInMemoryTransform(inst, inputMatrices, outputMatrices);
						Statistics.decrementNoOfExecutedMRJobs();
						execCP = true;
					}
					else 
					{
						ret = DataTransform.mrDataTransform(inst, inputMatrices, shuffleInst, otherInst, inst.getIv_resultIndices(), outputMatrices, inst.getIv_numReducers(), inst.getIv_replication());
					}
				break;
				
			default:
				throw new DMLRuntimeException("Invalid jobtype: " + inst.getJobType());
			}
			
		} // end of try block
		catch (Exception e) {
			throw new DMLRuntimeException( e );
		}

		if (ret.checkReturnStatus()) {
			/*
			 * Check if any output is empty. If yes, create a dummy file. Needs
			 * to be done only in case of (1) CellOutputInfo and if not CP, or 
			 * (2) BinaryBlockOutputInfo if not CP and output empty blocks disabled.
			 */
			try {
				if( !execCP )
				{
					for (int i = 0; i < outputMatrices.length; i++) {
						//get output meta data
						MatrixFormatMetaData meta = (MatrixFormatMetaData)outputMatrices[i].getMetaData();
						MatrixCharacteristics mc = meta.getMatrixCharacteristics();
						OutputInfo outinfo = meta.getOutputInfo();
						String fname = outputMatrices[i].getFileName();
						
						if (MapReduceTool.isHDFSFileEmpty(fname)) 
						{
							//prepare output file
							Path filepath = new Path(fname, "0-m-00000");
							MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(outinfo);
							writer.writeEmptyMatrixToHDFS(filepath.toString(), mc.getRows(), mc.getCols(), 
									              mc.getRowsPerBlock(), mc.getColsPerBlock());
						}
						
						outputMatrices[i].setFileExists(true);
						
						if ( inst.getJobType() != JobType.CSV_WRITE && inst.getJobType() != JobType.TRANSFORM) {
							// write out metadata file
							// Currently, valueType information in not stored in MR instruction, 
							// since only DOUBLE matrices are supported ==> hard coded the value type information for now
							MapReduceTool.writeMetaDataFile(fname + ".mtd", ValueType.DOUBLE,  ((MatrixDimensionsMetaData)ret.getMetaData(i)).getMatrixCharacteristics(), outinfo);
						}
					}
				}
				return ret;
			} catch (IOException e) {
				throw new DMLRuntimeException(e);
			}
		}

		// should not come here!
		throw new DMLRuntimeException("Unexpected Job Type: " + inst.getJobType());
	}

	/**
	 * 
	 * @param inst
	 * @param inputMatrices
	 * @param pb
	 * @throws DMLRuntimeException
	 */
	private static void checkEmptyInputs( MRJobInstruction inst, MatrixObject[] inputMatrices ) 
		throws DMLRuntimeException
	{
		// Check if any of the input files are empty.. only for those job types
		// for which empty inputs are NOT allowed
		if (!inst.getJobType().areEmptyInputsAllowed()) {
			for ( int i=0; i < inputMatrices.length; i++ ) {
				try {
					if (MapReduceTool.isHDFSFileEmpty(inputMatrices[i].getFileName())) {
						throw new DMLRuntimeException( "Can not operate on an empty file: " + inputMatrices[i].getFileName());
					}
				} catch (IOException e) {
					throw new DMLRuntimeException( "runtime error occurred -- " , e);
				}
			}
		}
	}
	
	/**
	 * Computes the replacement string for a given variable name placeholder string 
	 * (e.g., ##mVar2## or ##Var5##). The replacement is a HDFS filename for matrix 
	 * variables, and is the actual value (stored in symbol table) for scalar variables.
	 * 
	 * @param inst
	 * @param varName
	 * @param map
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static String getVarNameReplacement(String inst, String varName, LocalVariableMap map) throws DMLRuntimeException {
		Data val = map.get(varName);
		if ( val != null ) {
			String replacement = null;
			if (val.getDataType() == DataType.MATRIX) {
				replacement = ((MatrixObject)val).getFileName();
			}

			if (val.getDataType() == DataType.SCALAR)
				replacement = "" + ((ScalarObject) val).getStringValue();
			return replacement;
		}
		else {
			throw new DMLRuntimeException("Variable ("+varName+") in Instruction ("+inst+") is not found in the variablemap.");
		}
	}
	
	/** 
	 * Replaces ALL placeholder strings (such as ##mVar2## and ##Var5##) in a single instruction.
	 *  
	 * @param inst
	 * @param map
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static String updateInstLabels(String inst, LocalVariableMap map) throws DMLRuntimeException {
		if ( inst.contains(Lop.VARIABLE_NAME_PLACEHOLDER) ) {
			int skip = Lop.VARIABLE_NAME_PLACEHOLDER.toString().length();
			while ( inst.contains(Lop.VARIABLE_NAME_PLACEHOLDER) ) {
				int startLoc = inst.indexOf(Lop.VARIABLE_NAME_PLACEHOLDER)+skip;
				String varName = inst.substring(startLoc, inst.indexOf(Lop.VARIABLE_NAME_PLACEHOLDER, startLoc));
				String replacement = getVarNameReplacement(inst, varName, map);
				inst = inst.replaceAll(Lop.VARIABLE_NAME_PLACEHOLDER + varName + Lop.VARIABLE_NAME_PLACEHOLDER, replacement);
			}
		}
		return inst;
	}
	
	/**
	 * Takes a delimited string of instructions, and replaces ALL placeholder labels 
	 * (such as ##mVar2## and ##Var5##) in ALL instructions.
	 *  
	 * @param instList
	 * @param labelValueMapping
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static String updateLabels (String instList, LocalVariableMap labelValueMapping) throws DMLRuntimeException {

		if ( !instList.contains(Lop.VARIABLE_NAME_PLACEHOLDER) )
			return instList;
		
		StringBuilder updateInstList = new StringBuilder();
		String[] ilist = instList.split(Lop.INSTRUCTION_DELIMITOR); 
		
		for ( int i=0; i < ilist.length; i++ ) {
			if ( i > 0 )
				updateInstList.append(Lop.INSTRUCTION_DELIMITOR);
			
			updateInstList.append( updateInstLabels(ilist[i], labelValueMapping));
		}
		return updateInstList.toString();
	}

	
	/**
	 * 
	 * @param inputMatrices
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static long[] getNNZ( MatrixObject[] inputMatrices ) 
		throws DMLRuntimeException
	{
		int len = inputMatrices.length;
		long[] ret = new long[len];
		for( int i=0; i<len; i++ )
		{
			MatrixObject mo = inputMatrices[i];
			if( mo != null )
				ret[i] = mo.getNnz();
			else
				ret[i] = -1;
		}
			
		return ret;
	}
	
	/**
	 * 
	 * @param inst
	 * @param shuffleInst
	 * @param inputMatrices
	 * @param outputMatrices
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	private static JobReturn executeInMemoryReblockOperations( MRJobInstruction inst, String shuffleInst, MatrixObject[] inputMatrices, MatrixObject[] outputMatrices ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixCharacteristics[] mc = new MatrixCharacteristics[outputMatrices.length];
		ReblockInstruction[] rblkSet = MRInstructionParser.parseReblockInstructions(shuffleInst);
		byte[] results = inst.getIv_resultIndices();
		for( ReblockInstruction rblk : rblkSet )
		{
			//CP Reblock through caching framework (no copy required: same data, next op copies) 
			MatrixBlock mb = inputMatrices[rblk.input].acquireRead();
			for( int i=0; i<results.length; i++ )
				if( rblk.output == results[i] )
				{
					outputMatrices[i].acquireModify( mb );
					outputMatrices[i].release();
					mc[i] = new MatrixCharacteristics(mb.getNumRows(),mb.getNumColumns(), rblk.brlen, rblk.bclen, mb.getNonZeros());
				}
			inputMatrices[rblk.input].release();
		}
		
		return  new JobReturn( mc, inst.getOutputInfos(), true);
	}
	
	private static JobReturn executeInMemoryDataGenOperations( MRJobInstruction inst, String randInst, MatrixObject[] outputMatrices ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		MatrixCharacteristics[] mc = new MatrixCharacteristics[outputMatrices.length];
		DataGenMRInstruction[] dgSet = MRInstructionParser.parseDataGenInstructions(randInst);
		byte[] results = inst.getIv_resultIndices();
		for( DataGenMRInstruction ldgInst : dgSet )
		{
			if( ldgInst instanceof RandInstruction )
			{
				//CP Rand block operation 
				RandInstruction lrand = (RandInstruction)ldgInst; 
				RandomMatrixGenerator rgen = LibMatrixDatagen.createRandomMatrixGenerator(
																	lrand.getProbabilityDensityFunction(), 
																	(int)lrand.getRows(), (int)lrand.getCols(), 
																	lrand.getRowsInBlock(), lrand.getColsInBlock(), 
																	lrand.getSparsity(), lrand.getMinValue(), lrand.getMaxValue(), 
																	lrand.getPdfParams());
				
				
				MatrixBlock mb = MatrixBlock.randOperations(rgen, lrand.getSeed());
				
				for( int i=0; i<results.length; i++ )
					if( lrand.output == results[i] )
					{
						outputMatrices[i].acquireModify( mb );
						outputMatrices[i].release();
						mc[i] = new MatrixCharacteristics(mb.getNumRows(),mb.getNumColumns(), lrand.getRowsInBlock(), lrand.getColsInBlock(), mb.getNonZeros());
					}
			}
			else if( ldgInst instanceof SeqInstruction )
			{
				SeqInstruction lseq = (SeqInstruction) ldgInst;
				MatrixBlock mb = MatrixBlock.seqOperations(lseq.fromValue, lseq.toValue, lseq.incrValue);
				for( int i=0; i<results.length; i++ )
					if( lseq.output == results[i] )
					{
						outputMatrices[i].acquireModify( mb );
						outputMatrices[i].release();
						mc[i] = new MatrixCharacteristics(mb.getNumRows(),mb.getNumColumns(), lseq.getRowsInBlock(), lseq.getColsInBlock(), mb.getNonZeros());
					}
			}
		}
		
		return  new JobReturn( mc, inst.getOutputInfos(), true);
	}
	
	private static JobReturn executeInMemoryTransform( MRJobInstruction inst, MatrixObject[] inputMatrices, MatrixObject[] outputMatrices) throws IOException, DMLRuntimeException, IllegalArgumentException, JSONException {
		return DataTransform.cpDataTransform(
				inst.getIv_shuffleInstructions(), 
				inputMatrices, 
				outputMatrices);
	}
}
