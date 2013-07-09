package com.ibm.bi.dml.lops.runtime;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.lops.compile.Recompiler;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.MRInstructionParser;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReblockInstruction;
import com.ibm.bi.dml.runtime.matrix.CMCOVMR;
import com.ibm.bi.dml.runtime.matrix.CombineMR;
import com.ibm.bi.dml.runtime.matrix.DataPartitionMR;
import com.ibm.bi.dml.runtime.matrix.GMR;
import com.ibm.bi.dml.runtime.matrix.GroupedAggMR;
import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.MMCJMR;
import com.ibm.bi.dml.runtime.matrix.MMRJMR;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.RandMR;
import com.ibm.bi.dml.runtime.matrix.ReblockMR;
import com.ibm.bi.dml.runtime.matrix.SortMR;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.Statistics;


public class RunMRJobs {
	public static boolean flagLocalModeOpt = false;
	public enum ExecMode { LOCAL, CLUSTER, INVALID }; 
	//private static final Log LOG = LogFactory.getLog(RunMRJobs.class.getName());

	public static JobReturn submitJob(MRJobInstruction inst, ExecutionContext ec ) 
		throws DMLRuntimeException 
	{
		
		JobReturn ret = new JobReturn();

		// Obtain references to all input matrices 
		MatrixObject[] inputMatrices = inst.extractInputMatrices(ec);
		
		// export dirty matrices to HDFS
		// note: for REBLOCK postponed until we know if necessary
		if( !(inst.getJobType() == JobType.REBLOCK_TEXT || inst.getJobType() == JobType.REBLOCK_BINARY ) )
		{
			//export matrices
			for(MatrixObject m : inputMatrices) {
				if ( m.isDirty() )
					m.exportData();
			}

			//check input files
			checkEmptyInputs( inst, inputMatrices );	
		}
		
		// Obtain references to all output matrices
		MatrixObject[] outputMatrices = inst.extractOutputMatrices(ec);
		
		boolean execCP = false;
		
		// Spawn MapReduce Jobs
		try {
			// replace all placeholders in all instructions with appropriate values
			String rdInst = updateLabels(inst.getIv_randInstructions(), ec.getVariables());
			String rrInst = updateLabels(inst.getIv_recordReaderInstructions(), ec.getVariables());
			String mapInst = updateLabels(inst.getIv_instructionsInMapper(), ec.getVariables());
			String shuffleInst = updateLabels(inst.getIv_shuffleInstructions(), ec.getVariables());
			String aggInst = updateLabels(inst.getIv_aggInstructions(), ec.getVariables());
			String otherInst = updateLabels(inst.getIv_otherInstructions(), ec.getVariables());
			
			switch(inst.getJobType()) {
			
			case GMR: 
				ret = GMR.runJob(inst, inst.getInputs(), inst.getInputInfos(), 
						inst.getRlens(), inst.getClens(), inst.getBrlens(), inst.getBclens(),
						inst.getPartitioned(), inst.getPformats(), inst.getPsizes(),
						rrInst, mapInst, aggInst, otherInst,
						inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(), inst.getDimsUnknownFilePrefix(),
						inst.getOutputs(), inst.getOutputInfos() );
				 break;

			case RAND:
				ret = RandMR.runJob(inst, 
						rdInst.split(Lops.INSTRUCTION_DELIMITOR), mapInst, aggInst, otherInst, 
						inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(), inst.getDimsUnknownFilePrefix(),
						inst.getOutputs(), inst.getOutputInfos());
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
			
			case REBLOCK_TEXT:
			case REBLOCK_BINARY:
				if(    OptimizerUtils.ALLOW_DYN_RECOMPILATION 
					&& DMLScript.rtplatform != RUNTIME_PLATFORM.HADOOP 
					&& Recompiler.checkCPReblock( inst, inputMatrices ) ) 
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
					ret = new JobReturn( mc, inst.getOutputInfos(), true);
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
					
					ret = ReblockMR.runJob(inst, inst.getInputs(),  inst.getInputInfos(), 
							inst.getRlens(), inst.getClens(), inst.getBrlens(), inst.getBclens(),
							mapInst, shuffleInst, otherInst,
							inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(),   
							inst.getOutputs(), inst.getOutputInfos() );
				}
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
						mapInst, 
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
				
			default:
				throw new DMLRuntimeException("Invalid jobtype: " + inst.getJobType());
			}
			
			/*if (inst.getJobType() == JobType.PARTITION) {
				boolean blocked_rep = true;

				String input = updateLabels(inst.getIv_inputs(), inst.getInputLabelValueMapping())[0];	//the hdfs filepath! not the varbl name!
				InputInfo inputInfo = inst.getIv_inputInfos()[0];
				int numReducers = inst.getIv_numReducers();  
				int replication = inst.getIv_replication();
				long nr = inst.getIv_rows()[0];
				long nc = inst.getIv_cols()[0];
				int bnr = inst.getIv_num_rows_per_block()[0];
				int bnc = inst.getIv_num_cols_per_block()[0];
				PartitionParams pp = inst.getPartitionParams();

				// pp.setScratchPrefix(scratch);
				//TODO: fix output filepathnames (preprend "./data/" to outputs in submatrix and cell!) 
				if (pp.isEL == false && pp.pt == PartitionParams.PartitionType.submatrix) {
					if (inputInfo == InputInfo.BinaryBlockInputInfo && nr > bnr && nc > bnc) {
						ret = PartitionBlockMR.runJob(input, inputInfo, numReducers, replication, nr, nc, bnr, bnc, pp);
					} else {
						if (inputInfo != InputInfo.BinaryBlockInputInfo) {
							bnr =  DMLConfig.DEFAULT_BLOCK_SIZE; //config.getTextValue("defaultblocksize"); //pp.get_rows_in_block();
							bnc = DMLConfig.DEFAULT_BLOCK_SIZE; //config.getTextValue("defaultblocksize"); //pp.get_columns_in_block();
						}
						ret = PartitionSubMatrixMR.runJob(blocked_rep, input, inputInfo, numReducers, replication, nr,
								nr, bnr, bnc, pp);
					}
				}
				else if (pp.isEL == false && pp.pt == PartitionParams.PartitionType.cell) {
					PartitionCellMR.runJob(numReducers, nr, nc, replication,pp) ;
				}
				
				else if (pp.isEL == true || (pp.isEL == false && pp.pt == PartitionParams.PartitionType.row)) {
					if(pp.apt == PartitionParams.AccessPath.HM) {	//we can simply use the hashmap MR job!
						ret = PartitionBlockHashMapMR.runJob(input, InputInfo.BinaryBlockInputInfo, numReducers, replication,
								nr, nc, bnr, bnc, pp);
					}
					else if(pp.apt == PartitionParams.AccessPath.JR) {	//JR method
							ret = PartitionBlockJoinMR.runJob(input, InputInfo.BinaryBlockInputInfo, numReducers, 
																replication, nr, nc, bnr, bnc, pp);
							//TODO: need to reblock all the output fold matrices later from subrowblk format to blk format
					} 
					else {	//RB method	- usually this path is never chosen!
							//TODO: DRB: THIS IS BROKEN Call reblock here since it is required for this; need to reblock bef partng
							String inputre = input + "re";	//again, the hdfs file path - not the varbl name!
							if(pp.isColumn == false) {	//row-wise, so output as rowblk matrix
								ret = ReblockMR.runJob(new String[] { input }, new InputInfo[] { inputInfo },
										new long[] { nr }, new long[] { nc }, new int[] { bnr }, new int[] { bnc }, "",
										"rblk:::0:DOUBLE:::1:DOUBLE:::1:::"+nc,		//TODO #### can cause underflow!! from long to int in cpb!
										"", numReducers, replication, new byte[] { 1 }, new byte[] {0}, 
										new String[] { inputre }, new OutputInfo[] { OutputInfo.BinaryBlockOutputInfo });
								ret.checkReturnStatus();
								System.out.println("$$$$ Finished first reblocking in RB method $$$$$");
								ret = PartitionBlockMR.runJob(inputre, InputInfo.BinaryBlockInputInfo, numReducers, replication,
										nr, nc, 1, (int) nc, pp);	//the inputre is in rowblock format
							}
							else {	//col-wise, so output as colblk matrix
								ret = ReblockMR.runJob(new String[] { input }, new InputInfo[] { inputInfo },
										new long[] { nr }, new long[] { nc }, new int[] { bnr }, new int[] { bnc }, "",
										"rblk:::0:DOUBLE:::1:DOUBLE:::"+nr+":::1", 
										"", numReducers, replication, new byte[] { 1 }, new byte[] {0}, 
										new String[] { inputre }, new OutputInfo[] { OutputInfo.BinaryBlockOutputInfo });
								ret.checkReturnStatus();
								ret = PartitionBlockMR.runJob(inputre, InputInfo.BinaryBlockInputInfo, numReducers, replication,
										nr, nc, (int) nr, 1, pp);	//the inputre is in colblk format
							}
							//TODO: need to reblock all the output fold matrices later from subrowblk format to blk format
					}//end if on access paths
				}//end if on row type
			}//end partition stmt
			*/
			
		} // end of try block
		catch (Exception e) {
			throw new DMLRuntimeException( e );
		}

		if (ret.checkReturnStatus()) {
			/*
			 * Check if any output is empty. If yes, create a dummy file. Needs
			 * to be done only in case of CellOutputInfo and if not CP.
			 */
			try {
				if( !execCP )
				{
					for (int i = 0; i < outputMatrices.length; i++) {
						OutputInfo outinfo = ((MatrixFormatMetaData)outputMatrices[i].getMetaData()).getOutputInfo();
						String fname = outputMatrices[i].getFileName();
						if (outinfo == OutputInfo.TextCellOutputInfo || outinfo == OutputInfo.BinaryCellOutputInfo) {
							if (MapReduceTool.isHDFSFileEmpty(fname)) {
								// createNewFile(Path f)
								if (outinfo == OutputInfo.TextCellOutputInfo) {
									FileSystem fs = FileSystem
											.get(new Configuration());
									Path filepath = new Path(fname, "0-m-00000");
									FSDataOutputStream writer = fs.create(filepath);
									writer.writeBytes("1 1 0");
									writer.sync();
									writer.flush();
									writer.close();
								} else if (outinfo == OutputInfo.BinaryCellOutputInfo) {
									Configuration conf = new Configuration();
									FileSystem fs = FileSystem.get(conf);
									Path filepath = new Path(fname, "0-r-00000");
									SequenceFile.Writer writer = new SequenceFile.Writer(
											fs, conf, filepath,
											MatrixIndexes.class, MatrixCell.class);
									MatrixIndexes index = new MatrixIndexes(1, 1);
									MatrixCell cell = new MatrixCell(0);
									writer.append(index, cell);
									writer.close();
								}
							}
						}
						outputMatrices[i].setFileExists(true);
						
						// write out metadata file
						// Currently, valueType information in not stored in MR instruction, 
						// since only DOUBLE matrices are supported ==> hard coded the value type information for now
						MapReduceTool.writeMetaDataFile(fname + ".mtd", ValueType.DOUBLE,  ((MatrixDimensionsMetaData)ret.getMetaData(i)).getMatrixCharacteristics(), outinfo);
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
		if ( inst.contains(Lops.VARIABLE_NAME_PLACEHOLDER) ) {
			int skip = Lops.VARIABLE_NAME_PLACEHOLDER.toString().length();
			while ( inst.contains(Lops.VARIABLE_NAME_PLACEHOLDER) ) {
				int startLoc = inst.indexOf(Lops.VARIABLE_NAME_PLACEHOLDER)+skip;
				String varName = inst.substring(startLoc, inst.indexOf(Lops.VARIABLE_NAME_PLACEHOLDER, startLoc));
				String replacement = getVarNameReplacement(inst, varName, map);
				inst = inst.replaceAll(Lops.VARIABLE_NAME_PLACEHOLDER + varName + Lops.VARIABLE_NAME_PLACEHOLDER, replacement);
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

		if ( !instList.contains(Lops.VARIABLE_NAME_PLACEHOLDER) )
			return instList;
		
		StringBuilder updateInstList = new StringBuilder();
		String[] ilist = instList.split(Lops.INSTRUCTION_DELIMITOR); 
		
		for ( int i=0; i < ilist.length; i++ ) {
			if ( i > 0 )
				updateInstList.append(Lops.INSTRUCTION_DELIMITOR);
			
			updateInstList.append( updateInstLabels(ilist[i], labelValueMapping));
		}
		return updateInstList.toString();
	}

	/*private static String[] updateLabels(String[] inst, LocalVariableMap labelValueMapping) throws DMLRuntimeException {
		String[] str_array = new String[inst.length];
		for (int i = 0; i < inst.length; i++) {
			str_array[i] = updateInstLabels(inst[i], labelValueMapping);
		}
		return str_array;
	}*/
	
	/**
	 * Method to determine whether to execute a particular job in local-mode or 
	 * cluster-mode. This decision is take based on the "jobType" 
	 * (GMR, Reblock, etc.) as well as the input matrix dimensions, 
	 * given by "stats". Currently, the thresholds are chosen empirically.
	 * 
	 * @param jobType
	 * @param stats
	 * @return
	 * @throws DMLRuntimeException 
	 */
	@Deprecated
	public static ExecMode getExecMode(JobType jt, MatrixCharacteristics[] stats) throws DMLRuntimeException {
		
		//if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE )
		//	return ExecMode.LOCAL;
		
		if ( !flagLocalModeOpt )
			return ExecMode.CLUSTER;
		
		switch ( jt ) {
		case GMR:
		case RAND:
			if ( compareInputDimensions(stats, 3) )
				return ExecMode.LOCAL;
			break;
		case REBLOCK_BINARY:
		case REBLOCK_TEXT:
			if ( compareInputDimensions(stats, 1) )
				return ExecMode.LOCAL;
			break;
		case MMCJ:
			if ( compareInputDimensions(stats, 2) )
				return ExecMode.LOCAL;
			break;
		case MMRJ:
			if ( compareInputDimensions(stats, 1) ) // this needs to be verified (empirically)
				return ExecMode.LOCAL;
			break;
		case CM_COV:
		case PARTITION:
		case SORT:
		case COMBINE:
			if ( compareInputDimensions(stats, 1) ) // this needs to be verified (empirically)
				return ExecMode.LOCAL;
			break;
		default:
			throw new DMLRuntimeException("Unknown job type (" + jt.getName() + ") while determining the execution mode.");
		}
		
		return ExecMode.CLUSTER;
	}
	
	private static boolean compareInputDimensions(MatrixCharacteristics[] stats, int numBlocks ) {
		for(int i=0; i < stats.length; i++ ) {
			// check if the input dimensions are smaller than the specified number of blocks
			if ( stats[i].numRows != -1
					&& stats[i].numRows <= numBlocks * DMLTranslator.DMLBlockSize
					&& stats[i].numColumns != -1
					&& stats[i].numColumns <= numBlocks * DMLTranslator.DMLBlockSize) {
				continue;
			}
			else 
				return false;
		}
		return true;
	}
}
