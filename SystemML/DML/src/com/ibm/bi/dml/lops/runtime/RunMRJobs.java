package com.ibm.bi.dml.lops.runtime;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.meta.PartitionBlockHashMapMR;
import com.ibm.bi.dml.meta.PartitionBlockJoinMR;
import com.ibm.bi.dml.meta.PartitionBlockMR;
import com.ibm.bi.dml.meta.PartitionCellMR;
import com.ibm.bi.dml.meta.PartitionParams;
import com.ibm.bi.dml.meta.PartitionSubMatrixMR;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.matrix.CMCOVMR;
import com.ibm.bi.dml.runtime.matrix.CombineMR;
import com.ibm.bi.dml.runtime.matrix.GMR;
import com.ibm.bi.dml.runtime.matrix.GroupedAggMR;
import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.MMCJMR;
import com.ibm.bi.dml.runtime.matrix.MMRJMR;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;
import com.ibm.bi.dml.runtime.matrix.RandMR;
import com.ibm.bi.dml.runtime.matrix.ReblockMR;
import com.ibm.bi.dml.runtime.matrix.SortMR;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;


public class RunMRJobs {
	public static boolean flagLocalModeOpt = true;
	public enum ExecMode { LOCAL, CLUSTER, INVALID }; 

	public static JobReturn submitJob(MRJobInstruction inst, ProgramBlock pb ) throws DMLRuntimeException {
		JobReturn ret = new JobReturn();

		if ( DMLScript.DEBUG  )
			System.out.println(inst.toString());

		// Check if any of the input files are empty.. only for those job types
		// for which empty inputs are NOT allowed
		if (!inst.getJobType().areEmptyInputsAllowed()) {
			for (String f : updateLabels(inst.getIv_inputs(), inst.getInputLabelValueMapping())) {
				try {
					if (MapReduceTool.isHDFSFileEmpty(f)) {
						throw new DMLRuntimeException("Can not operate on an empty file: " + f);
					}
				} catch (IOException e) {
					throw new DMLRuntimeException(e);
				}
			}
		}

		try {
			String[] updatedInputLabels = null;
			String[] updatedOutputLabels = null;
			long[] updatedRows = null;
			long[] updatedCols = null;
			int[] updatedRowsPerBlock = null;
			int[] updatedColsPerBlock = null;
			
			/*
			 * Update placeholder labels i.e., the ones of the form ##<label>##
			 * Update rows and columns that are set in piggybacking by 
			 *   looking at the metadata structure (_matrices) in program block 
			 */
			if ( inst.getIv_inputs() != null )
				updatedInputLabels = updateLabels(inst.getIv_inputs(), inst.getInputLabelValueMapping());
			if ( inst.getIv_outputs() != null )
				updatedOutputLabels = updateLabels(inst.getIv_outputs(), inst.getOutputLabelValueMapping());
			if ( inst.getIv_rows() != null && updatedInputLabels != null )
				updatedRows = updateRows(inst.getIv_rows(), updatedInputLabels, inst.getInputLabels(), pb);
			if ( inst.getIv_cols() != null && updatedInputLabels != null )
				updatedCols = updateCols(inst.getIv_cols(), updatedInputLabels, inst.getInputLabels(), pb);
			if ( inst.getIv_num_rows_per_block() != null && updatedInputLabels != null )
				updatedRowsPerBlock = updateRowsPerBlock(inst.getIv_num_rows_per_block(), updatedInputLabels, inst.getInputLabels(), pb);
			if ( inst.getIv_num_cols_per_block() != null && updatedInputLabels != null )
				updatedColsPerBlock = updateColsPerBlock(inst.getIv_num_cols_per_block(), updatedInputLabels, inst.getInputLabels(), pb);
			
			if (inst.getJobType() == JobType.GMR) {
				boolean blocked_rep = true;
				
				// If any input value class is Cell then the whole job should run in "Cell" mode
				// Remaining blocked input is implicitly converted into cell format
				for ( InputInfo ii : inst.getIv_inputInfos()) {
					if ( ii == InputInfo.TextCellInputInfo || ii == InputInfo.BinaryCellInputInfo ) {
						blocked_rep = false;
						break;
					}
				}
				
				// TODO: statiko -- how to remove this check?
				if ( blocked_rep == false ) {
					// Check if any outputs are of type BinaryBlock!
					// output info must not be blocked
					for ( int oi=0; oi < inst.getIv_outputs().length; oi++ ) {
						if ( inst.getIv_outputInfos()[oi] == OutputInfo.BinaryBlockOutputInfo) {
							inst.getIv_outputInfos()[oi] = OutputInfo.BinaryCellOutputInfo;
						}
					}
				}
				
				if ( !inst.getIv_recordReaderInstructions().equals("") ) {
					// if there are record reader instructions, we need to update MetaData
					
					// In the presence of recordReader instructions (valuepick/rangepick), the job must operate in cell mode
					blocked_rep = false;
					
					// output info must not be blocked
					for ( int oi=0; oi < inst.getIv_outputs().length; oi++ ) {
						if ( inst.getIv_outputInfos()[oi] == OutputInfo.BinaryBlockOutputInfo) {
							inst.getIv_outputInfos()[oi] = OutputInfo.BinaryCellOutputInfo;
						}
					}
					
					// get the index of the matrix whose metadata needs to be fetched
					String[] ins = inst.getIv_recordReaderInstructions().split(Lops.INSTRUCTION_DELIMITOR);
					
					if ( ins.length > 1 ) 
						throw new DMLRuntimeException("There can not be more than one recordreader instructions");
					
					// look at the first instruction
					String[] parts = ins[0].split(Lops.OPERAND_DELIMITOR);
					
					if ( parts[1].equalsIgnoreCase("valuepick") || parts[1].equalsIgnoreCase("rangepick")) {
						String [] fields = parts[2].split(Lops.VALUETYPE_PREFIX);
						int input_index = Integer.parseInt(fields[0]); 
						
						String fname = inst.getIv_inputs()[input_index];
						String varname = inst.getInputLabels().get(input_index);
						MatrixObject mobj = (MatrixObject)pb.getVariable(varname);
						inst.getIv_inputInfos()[input_index].metadata = mobj.getMetaData();
					}
					else 
						throw new DMLRuntimeException("Recordreader instructions for opcode=" + parts[0] + " are not supported yet.");
						
				}
				
				
				ret = GMR.runJob(blocked_rep, updatedInputLabels, inst.getIv_inputInfos(), updatedRows, updatedCols, updatedRowsPerBlock,
						updatedColsPerBlock, updateLabels(inst.getIv_recordReaderInstructions(), inst
								.getInputLabelValueMapping()), updateLabels(inst.getIv_instructionsInMapper(), inst
										.getInputLabelValueMapping()), inst.getIv_aggInstructions(), updateLabels(inst
								.getIv_otherInstructions(), inst.getInputLabelValueMapping()),
						inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(), inst.getIv_resultDimsUnknown(), 
						updatedOutputLabels, inst.getIv_outputInfos());
			}

			if (inst.getJobType() == JobType.CM_COV) {
				ret = CMCOVMR.runJob(updateLabels(inst.getIv_inputs(), inst.getInputLabelValueMapping()), inst
						.getIv_inputInfos(), updatedRows, updatedCols, updatedRowsPerBlock, updatedColsPerBlock, 
						updateLabels(inst.getIv_instructionsInMapper(), inst
								.getInputLabelValueMapping()), updateLabels(inst.getIv_shuffleInstructions(),inst.getInputLabelValueMapping()), 
						inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(), inst.getIv_outputs(), inst.getIv_outputInfos());
			}
			
			if (inst.getJobType() == JobType.GROUPED_AGG) {
				// GroupedAgg job must always run in cell mode
				// hence, no output info must be blocked
				for ( int oi=0; oi < inst.getIv_outputs().length; oi++ ) {
					if ( inst.getIv_outputInfos()[oi] == OutputInfo.BinaryBlockOutputInfo) {
						inst.getIv_outputInfos()[oi] = OutputInfo.BinaryCellOutputInfo;
					}
				}
				
				ret = GroupedAggMR.runJob(updateLabels(inst.getIv_inputs(), inst.getInputLabelValueMapping()), inst
						.getIv_inputInfos(), updatedRows, updatedCols, updatedRowsPerBlock, updatedColsPerBlock, 
						updateLabels(inst.getIv_shuffleInstructions(),inst.getInputLabelValueMapping()), 
						updateLabels(inst.getIv_otherInstructions(),inst.getInputLabelValueMapping()),
						inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(), inst.getIv_outputs(), inst.getIv_outputInfos());
			}
			
			if (inst.getJobType() == JobType.REBLOCK_TEXT || inst.getJobType() == JobType.REBLOCK_BINARY) {
				ret = ReblockMR.runJob(updateLabels(inst.getIv_inputs(), inst.getInputLabelValueMapping()), inst
						.getIv_inputInfos(), updatedRows, updatedCols, updatedRowsPerBlock, updatedColsPerBlock, 
						updateLabels(inst.getIv_instructionsInMapper(), inst
								.getInputLabelValueMapping()), inst.getIv_shuffleInstructions(), updateLabels(inst
								.getIv_otherInstructions(), inst.getInputLabelValueMapping()),
						inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(), inst.getIv_resultDimsUnknown(), 
						updateLabels(inst.getIv_outputs(), inst.getOutputLabelValueMapping()), inst.getIv_outputInfos());
			}

			if (inst.getJobType() == JobType.MMCJ) {
				boolean blocked_rep = true;

				ret = MMCJMR.runJob(blocked_rep, updateLabels(inst.getIv_inputs(), inst.getInputLabelValueMapping()),
						inst.getIv_inputInfos(), updatedRows, updatedCols, 
						updatedRowsPerBlock, updatedColsPerBlock, updateLabels(inst.getIv_instructionsInMapper(), inst
								.getInputLabelValueMapping()), inst.getIv_aggInstructions(), inst
								.getIv_shuffleInstructions(), inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultDimsUnknown()[0], inst
								.getIv_outputs()[0], inst.getIv_outputInfos()[0]);
			}

			if (inst.getJobType() == JobType.MMRJ) {
				boolean blocked_rep = true;

				ret = MMRJMR.runJob(blocked_rep, updateLabels(inst.getIv_inputs(), inst.getInputLabelValueMapping()),
						inst.getIv_inputInfos(), updatedRows, updatedCols, updatedRowsPerBlock, updatedColsPerBlock, 
						updateLabels(inst.getIv_instructionsInMapper(), inst.getInputLabelValueMapping()), inst.getIv_aggInstructions(), 
						inst.getIv_shuffleInstructions(), updateLabels(inst.getIv_otherInstructions(), inst.getInputLabelValueMapping()), 
						inst.getIv_numReducers(), inst.getIv_replication(), inst.getIv_resultIndices(), inst.getIv_resultDimsUnknown(), 
						updateLabels(inst.getIv_outputs(),inst.getOutputLabelValueMapping() ), inst.getIv_outputInfos());
			}

			if (inst.getJobType() == JobType.SORT) {
				// TODO: statiko -- fix the flag for weights
				boolean weightsflag = true;
				if ( !inst.getIv_instructionsInMapper().equalsIgnoreCase("") )
					weightsflag = false;
				ret = SortMR.runJob(updateLabels(inst.getIv_inputs(), inst.getInputLabelValueMapping())[0], inst.getIv_inputInfos()[0], 
									updatedRows[0], updatedCols[0], updatedRowsPerBlock[0], updatedColsPerBlock[0], 
									inst.getIv_instructionsInMapper(), inst.getIv_numReducers(), inst.getIv_replication(), (byte)0,
									inst.getIv_outputs()[0], inst.getIv_outputInfos()[0], weightsflag );
			}

			if (inst.getJobType() == JobType.COMBINE) {
				boolean blocked_rep = true;
				
				// If any input value class is Cell then the whole job should run in "Cell" mode
				// Remaining blocked input is implicitly converted into cell format
				for ( InputInfo ii : inst.getIv_inputInfos()) {
					if ( ii == InputInfo.TextCellInputInfo || ii == InputInfo.BinaryCellInputInfo ) {
						blocked_rep = false;
					}
				}
				ret = CombineMR.runJob(blocked_rep, updateLabels(inst.getIv_inputs(), inst.getInputLabelValueMapping()), inst.getIv_inputInfos(), 
									updatedRows, updatedCols, updatedRowsPerBlock, updatedColsPerBlock, 
									inst.getIv_shuffleInstructions(), inst.getIv_numReducers(), inst.getIv_replication(),  
									inst.getIv_resultIndices(), updateLabels(inst.getIv_outputs(),inst.getOutputLabelValueMapping() ),inst.getIv_outputInfos() );

			}
			
			if (inst.getJobType() == JobType.PARTITION) {
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
			
			if (inst.getJobType() == JobType.RAND) {

				String[] randJobs = inst.getIv_randInstructions().split(",");
				ret = RandMR.runJob(randJobs, inst.getIv_num_rows_per_block(), inst.getIv_num_cols_per_block(),
						updateLabels(inst.getIv_instructionsInMapper(), inst.getInputLabelValueMapping()), inst
								.getIv_aggInstructions(), updateLabels(inst.getIv_otherInstructions(), inst
								.getInputLabelValueMapping()), inst.getIv_numReducers(), inst.getIv_replication(), inst
								.getIv_resultIndices(), inst.getIv_resultDimsUnknown(), inst.getIv_outputs(), inst.getIv_outputInfos());

			
			}
		} // end of try block
		catch (Exception e) {
			throw new DMLRuntimeException(e);
		}

		if (ret.checkReturnStatus()) {
			/*
			 * Check if any output is empty. If yes, create a dummy file. Needs
			 * to be done only in case of CellOutputInfo.
			 */
			try {
				for (int i = 0; i < inst.getIv_outputs().length; i++) {
					OutputInfo outinfo = inst.getIv_outputInfos()[i];
					if (outinfo == OutputInfo.TextCellOutputInfo || outinfo == OutputInfo.BinaryCellOutputInfo) {
						String fname = inst.getIv_outputs()[i];
							if (MapReduceTool.isHDFSFileEmpty(fname)) {
								// createNewFile(Path f)
								if (outinfo == OutputInfo.TextCellOutputInfo) {
									FileSystem fs = FileSystem.get(new Configuration());
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
									SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, filepath,
											MatrixIndexes.class, MatrixCell.class);
									MatrixIndexes index = new MatrixIndexes(1, 1);
									MatrixCell cell = new MatrixCell(0);
									writer.append(index, cell);
									writer.close();
								}
							}
					}
					
					// write out metadata file
					// TODO: we currently don't carry valueType information in MR instruction, 
					// as we only support DOUBLE type, hard coded the value type information for now
					MapReduceTool.writeMetaDataFile(inst.getIv_outputs()[i] + ".mtd", ValueType.DOUBLE,  ((MatrixDimensionsMetaData)ret.getMetaData(i)).getMatrixCharacteristics(), outinfo);
				}
				return ret;
			} catch (IOException e) {
				throw new DMLRuntimeException(e);
			}
		}

		// should not come here!
		throw new DMLRuntimeException("Unexpected Job Type: " + inst.getJobType());
	}

	public static String updateLabels (String inst, LocalVariableMap labelValueMapping) {
		Iterator<String> it = labelValueMapping.keySet().iterator();
		String updateInst = new String(inst);

		while (it.hasNext()) {
			String label = it.next();
			Data val = labelValueMapping.get(label);

			if ( val != null ) {
				String replacement = null;
				if (val.getDataType() == DataType.MATRIX) {
					replacement = ((MatrixObject)val).getFileName();
				}
	
				if (val.getDataType() == DataType.SCALAR)
					replacement = "" + ((ScalarObject) val).getStringValue();
	
				// System.out.println("Replacement string = " + replacement);
				// System.out.println("Original string = " + updateInst + " label ="
				// + label);
				updateInst = updateInst.replaceAll(Lops.VARIABLE_NAME_PLACEHOLDER + label + Lops.VARIABLE_NAME_PLACEHOLDER, replacement);
	
				//System.out.println("New string = " + updateInst);
			}
		}

		return updateInst;
	}

	private static String[] updateLabels(String[] inst, LocalVariableMap labelValueMapping) {
		String[] str_array = new String[inst.length];
		for (int i = 0; i < inst.length; i++) {
			str_array[i] = updateLabels(inst[i], labelValueMapping);
		}

		return str_array;
	}
	
	private static long[] updateRows(long[] rows, String[] inputlabels, ArrayList<String> inputVars, ProgramBlock pb) throws DMLRuntimeException {
		
		long[] newrows = new long[rows.length];
		
		MatrixCharacteristics matchar;
		for ( int i=0; i < rows.length && inputVars.size() > 0; i++) {
			matchar = ((MatrixDimensionsMetaData)pb.getMetaData(inputVars.get(i))).getMatrixCharacteristics();
			//matchar = ((MatrixDimensionsMetaData) mdmap.get(inputlabels[i])).getMatrixCharacteristics(); 
			
			// matchar represents the metadata that is computed by runtime for the file inputlabels[i]
			// it can not be NULL at this point
			if ( matchar == null || matchar.numRows == -1 ) { 
				throw new DMLRuntimeException("Unexpeced error in populating the metadata for intermediate matrix: " + inputlabels[i]);
			}
			else {
			//	if ( rows[i] == -1 )
					newrows[i] = matchar.numRows;
			//	else {
			//		if ( rows[i] != matchar.numRows ) 
			//			throw new DMLRuntimeException("Mismatch in dimenstions: " + rows[i] + " != " + matchar.numRows);
			//		else
			//			newrows[i] = matchar.numRows;
			//	}
			}
		}
		return newrows;
	}

	private static long[] updateCols(long[] cols, String[] inputlabels, ArrayList<String> inputVars, ProgramBlock pb) throws DMLRuntimeException {

		long[] newcols = new long[cols.length];
		
		MatrixCharacteristics matchar;
		for ( int i=0; i < cols.length&& inputVars.size() > 0; i++) {
			matchar = ((MatrixDimensionsMetaData)pb.getMetaData(inputVars.get(i))).getMatrixCharacteristics();
			//matchar = ((MatrixDimensionsMetaData) mdmap.get(inputlabels[i])).getMatrixCharacteristics(); 
			
			// matchar represents the metadata that is computed by runtime for the file inputlabels[i]
			// it can not be NULL at this point
			if ( matchar == null || matchar.numColumns == -1 ) { 
				throw new DMLRuntimeException("Unexpeced error in populating the metadata for intermediate matrix: " + inputlabels[i]);
			}
			else {
			//	if ( cols[i] == -1 )
					newcols[i] = matchar.numColumns;
			//	else {
			//		if ( cols[i] != matchar.numColumns ) 
			//			throw new DMLRuntimeException("Mismatch in dimenstions: " + cols[i] + " != " + matchar.numColumns);
			//		else
			//			newcols[i] = matchar.numColumns;
			//	}
			}
		}
		return newcols;
	}
	
	private static int[] updateRowsPerBlock(int[] rpb, String[] inputlabels, ArrayList<String> inputVars, ProgramBlock pb) throws DMLRuntimeException {
		
		int[] newrpb = new int[rpb.length];
		
		MatrixCharacteristics matchar;
		for ( int i=0; i < rpb.length&& inputVars.size() > 0; i++) {
			matchar = ((MatrixDimensionsMetaData)pb.getMetaData(inputVars.get(i))).getMatrixCharacteristics();
			//matchar = ((MatrixDimensionsMetaData) mdmap.get(inputlabels[i])).getMatrixCharacteristics(); 
			
			// matchar represents the metadata that is computed by runtime for the file inputlabels[i]
			// it can not be NULL at this point
			if ( matchar == null) { 
				throw new DMLRuntimeException("Unexpeced error in populating the metadata for intermediate matrix: " + inputlabels[i]);
			}
			else {
				//if ( rpb[i] == -1 )
					newrpb[i] = matchar.numRowsPerBlock;
				//else {
				//	if ( rpb[i] != matchar.numRowsPerBlock ) 
				//		throw new DMLRuntimeException("Mismatch in dimenstions: " + rpb[i] + " != " + matchar.numRowsPerBlock);
				//	else
				//		newrpb[i] = matchar.numRowsPerBlock;
				//}
			}
		}
		return newrpb;
	}

	private static int[] updateColsPerBlock(int[] cpb, String[] inputlabels, ArrayList<String> inputVars, ProgramBlock pb) throws DMLRuntimeException {
		
		int[] newcpb = new int[cpb.length];
		
		MatrixCharacteristics matchar;
		for ( int i=0; i < cpb.length&& inputVars.size() > 0; i++) {
			matchar = ((MatrixDimensionsMetaData)pb.getMetaData(inputVars.get(i))).getMatrixCharacteristics();
			//matchar = ((MatrixDimensionsMetaData) mdmap.get(inputlabels[i])).getMatrixCharacteristics(); 
			
			// matchar represents the metadata that is computed by runtime for the file inputlabels[i]
			// it can not be NULL at this point
			if ( matchar == null) { 
				throw new DMLRuntimeException("Unexpeced error in populating the metadata for intermediate matrix: " + inputlabels[i]);
			}
			else {
				//if ( cpb[i] == -1 )
					newcpb[i] = matchar.numColumnsPerBlock;
				//else {
				//	if ( cpb[i] != matchar.numColumnsPerBlock ) 
				//		throw new DMLRuntimeException("Mismatch in dimenstions: " + cpb[i] + " != " + matchar.numColumnsPerBlock);
				//	else
				//		newcpb[i] = matchar.numColumnsPerBlock;
				//}
			}
		}
		return newcpb;
	}


/*	private static long[] updateRows(long[] rows, String[] inputlabels, HashMap<String, MetaData> mdmap) throws DMLRuntimeException {
		
		long[] newrows = new long[rows.length];
		
		MatrixCharacteristics matchar;
		for ( int i=0; i < rows.length; i++) {
			matchar = ((MatrixDimensionsMetaData) mdmap.get(inputlabels[i])).getMatrixCharacteristics(); 
			
			// matchar represents the metadata that is computed by runtime for the file inputlabels[i]
			// it can not be NULL at this point
			if ( matchar == null || matchar.numRows == -1 ) { 
				throw new DMLRuntimeException("Unexpeced error in populating the metadata for intermediate matrix: " + inputlabels[i]);
			}
			else {
				if ( rows[i] == -1 )
					newrows[i] = matchar.numRows;
				else {
					if ( rows[i] != matchar.numRows ) 
						throw new DMLRuntimeException("Mismatch in dimenstions: " + rows[i] + " != " + matchar.numRows);
					else
						newrows[i] = matchar.numRows;
				}
			}
		}
		return newrows;
	}

	private static long[] updateCols(long[] cols, String[] inputlabels, HashMap<String, MetaData> mdmap) throws DMLRuntimeException {

		long[] newcols = new long[cols.length];
		
		MatrixCharacteristics matchar;
		for ( int i=0; i < cols.length; i++) {
			matchar = ((MatrixDimensionsMetaData) mdmap.get(inputlabels[i])).getMatrixCharacteristics(); 
			
			// matchar represents the metadata that is computed by runtime for the file inputlabels[i]
			// it can not be NULL at this point
			if ( matchar == null || matchar.numColumns == -1 ) { 
				throw new DMLRuntimeException("Unexpeced error in populating the metadata for intermediate matrix: " + inputlabels[i]);
			}
			else {
				if ( cols[i] == -1 )
					newcols[i] = matchar.numColumns;
				else {
					if ( cols[i] != matchar.numColumns ) 
						throw new DMLRuntimeException("Mismatch in dimenstions: " + cols[i] + " != " + matchar.numColumns);
					else
						newcols[i] = matchar.numColumns;
				}
			}
		}
		return newcols;
	}
	
	private static int[] updateRowsPerBlock(int[] rpb, String[] inputlabels, HashMap<String, MetaData> mdmap) throws DMLRuntimeException {
		
		int[] newrpb = new int[rpb.length];
		
		MatrixCharacteristics matchar;
		for ( int i=0; i < rpb.length; i++) {
			matchar = ((MatrixDimensionsMetaData) mdmap.get(inputlabels[i])).getMatrixCharacteristics(); 
			
			// matchar represents the metadata that is computed by runtime for the file inputlabels[i]
			// it can not be NULL at this point
			if ( matchar == null) { 
				throw new DMLRuntimeException("Unexpeced error in populating the metadata for intermediate matrix: " + inputlabels[i]);
			}
			else {
				//if ( rpb[i] == -1 )
					newrpb[i] = matchar.numRowsPerBlock;
				//else {
				//	if ( rpb[i] != matchar.numRowsPerBlock ) 
				//		throw new DMLRuntimeException("Mismatch in dimenstions: " + rpb[i] + " != " + matchar.numRowsPerBlock);
				//	else
				//		newrpb[i] = matchar.numRowsPerBlock;
				//}
			}
		}
		return newrpb;
	}

	private static int[] updateColsPerBlock(int[] cpb, String[] inputlabels, HashMap<String, MetaData> mdmap) throws DMLRuntimeException {
		
		int[] newcpb = new int[cpb.length];
		
		MatrixCharacteristics matchar;
		for ( int i=0; i < cpb.length; i++) {
			matchar = ((MatrixDimensionsMetaData) mdmap.get(inputlabels[i])).getMatrixCharacteristics(); 
			
			// matchar represents the metadata that is computed by runtime for the file inputlabels[i]
			// it can not be NULL at this point
			if ( matchar == null) { 
				throw new DMLRuntimeException("Unexpeced error in populating the metadata for intermediate matrix: " + inputlabels[i]);
			}
			else {
				//if ( cpb[i] == -1 )
					newcpb[i] = matchar.numColumnsPerBlock;
				//else {
				//	if ( cpb[i] != matchar.numColumnsPerBlock ) 
				//		throw new DMLRuntimeException("Mismatch in dimenstions: " + cpb[i] + " != " + matchar.numColumnsPerBlock);
				//	else
				//		newcpb[i] = matchar.numColumnsPerBlock;
				//}
			}
		}
		return newcpb;
	}

*/	/**
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
	public static ExecMode getExecMode(JobType jt, MatrixCharacteristics[] stats) throws DMLRuntimeException {
		
		if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE )
			return ExecMode.LOCAL;
		
		if ( flagLocalModeOpt == false )
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
