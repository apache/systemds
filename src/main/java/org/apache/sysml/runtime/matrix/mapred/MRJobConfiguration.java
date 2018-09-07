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


package org.apache.sysml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.io.BinaryBlockSerialization;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.AddDummyWeightConverter;
import org.apache.sysml.runtime.matrix.data.BinaryBlockToBinaryCellConverter;
import org.apache.sysml.runtime.matrix.data.BinaryBlockToRowBlockConverter;
import org.apache.sysml.runtime.matrix.data.BinaryBlockToTextCellConverter;
import org.apache.sysml.runtime.matrix.data.BinaryCellToRowBlockConverter;
import org.apache.sysml.runtime.matrix.data.BinaryCellToTextConverter;
import org.apache.sysml.runtime.matrix.data.Converter;
import org.apache.sysml.runtime.matrix.data.IdenticalConverter;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.data.TextCellToRowBlockConverter;
import org.apache.sysml.runtime.matrix.data.TextToBinaryCellConverter;
import org.apache.sysml.runtime.matrix.data.WeightedCellToSortInputConverter;
import org.apache.sysml.runtime.matrix.data.WeightedPair;

@SuppressWarnings({ "rawtypes", "deprecation" })
public class MRJobConfiguration 
{
		
	 //internal param: custom deserializer/serializer (usually 30% faster than WritableSerialization)
	public static final boolean USE_BINARYBLOCK_SERIALIZATION = true;
	
	//Job configurations
	
	public static IDSequence seq = new IDSequence();
	
	//input matrices
	private static final String INPUT_MATRICIES_DIRS_CONFIG="input.matrices.dirs";
	
	//this is here to handle record reader instructions
	private static final String MAPFUNC_INPUT_MATRICIES_INDEXES_CONFIG="mapfuc.input.matrices.indexes";
	
	//about the formats of inputs
	private static final String BLOCK_REPRESENTATION_CONFIG="in.block.representation";
	private static final String WEIGHTEDCELL_REPRESENTATION_CONFIG="in.weighted.cell.representation";
	private static final String INPUT_CONVERTER_CLASS_PREFIX_CONFIG="input.converter.class.for.";
	private static final String INPUT_KEY_CLASS_PREFIX_CONFIG="input.key.class.for.";
	private static final String INPUT_VALUE_CLASS_PREFIX_CONFIG="input.value.class.for.";
	
	//characteristics about input matrices
	private static final String INPUT_MATRIX_NUM_ROW_PREFIX_CONFIG="input.matrix.num.row.";
	private static final String INPUT_MATRIX_NUM_COLUMN_PREFIX_CONFIG="input.matrix.num.column.";
	private static final String INPUT_BLOCK_NUM_ROW_PREFIX_CONFIG="input.block.num.row.";
	private static final String INPUT_BLOCK_NUM_COLUMN_PREFIX_CONFIG="input.block.num.column.";
	private static final String INPUT_MATRIX_NUM_NNZ_PREFIX_CONFIG="input.matrix.num.nnz.";
	
	//characteristics about the matrices to map outputs
	private static final String MAPOUTPUT_MATRIX_NUM_ROW_PREFIX_CONFIG="map.output.matrix.num.row.";
	private static final String MAPOUTPUT_MATRIX_NUM_COLUMN_PREFIX_CONFIG="map.output.matrix.num.column.";
	private static final String MAPOUTPUT_BLOCK_NUM_ROW_PREFIX_CONFIG="map.output.block.num.row.";
	private static final String MAPOUTPUT_BLOCK_NUM_COLUMN_PREFIX_CONFIG="map.output.block.num.column.";
	
	//operations performed in the mapper
	private static final String INSTRUCTIONS_IN_MAPPER_CONFIG="instructions.in.mapper";
	private static final String RAND_INSTRUCTIONS_CONFIG="rand.instructions";
	//matrix indexes to be outputted to reducer
	private static final String OUTPUT_INDEXES_IN_MAPPER_CONFIG="output.indexes.in.mapper";
	
	//parfor serialized program
	private static final String PARFOR_PROGRAMBLOCKS_CONFIG = "parfor.programblocks.in.mr";
	private static final String PARFOR_CACHING_CONFIG = "parfor.cp.caching";
	
	//partitioning input/output info
	private static final String PARTITIONING_INPUT_MATRIX_NUM_ROW_CONFIG="partitioning.input.matrix.num.row";
	private static final String PARTITIONING_INPUT_MATRIX_NUM_COLUMN_CONFIG="partitioning.input.matrix.num.column";
	private static final String PARTITIONING_INPUT_BLOCK_NUM_ROW_CONFIG="partitioning.input.block.num.row";
	private static final String PARTITIONING_INPUT_BLOCK_NUM_COLUMN_CONFIG="partitioning.input.block.num.column";
	private static final String PARTITIONING_INPUT_INFO_CONFIG="partitioning.input.inputinfo";
	private static final String PARTITIONING_OUTPUT_INFO_CONFIG="partitioning.output.outputinfo";
	private static final String PARTITIONING_OUTPUT_FORMAT_CONFIG="partitioning.output.format";
	private static final String PARTITIONING_OUTPUT_N_CONFIG="partitioning.output.n";
	private static final String PARTITIONING_OUTPUT_FILENAME_CONFIG="partitioning.output.filename";
	private static final String PARTITIONING_ITERVAR_CONFIG="partitioning.itervar";
	private static final String PARTITIONING_MATRIXVAR_CONFIG="partitioning.matrixvar";
	private static final String PARTITIONING_TRANSPOSE_COL_CONFIG="partitioning.transposed.col";
	private static final String PARTITIONING_OUTPUT_KEEP_INDEXES_CONFIG="partitioning.output.keep.indexes";
	
	//result merge info
	private static final String RESULTMERGE_INPUT_INFO_CONFIG="resultmerge.input.inputinfo";
	private static final String RESULTMERGE_COMPARE_FILENAME_CONFIG="resultmerge.compare.filename";
	private static final String RESULTMERGE_ACCUMULATOR_CONFIG="resultmerge.accumulator";
	private static final String RESULTMERGE_STAGING_DIR_CONFIG="resultmerge.staging.dir";
	private static final String RESULTMERGE_MATRIX_NUM_ROW_CONFIG="resultmerge.matrix.num.row";
	private static final String RESULTMERGE_MATRIX_NUM_COLUMN_CONFIG="resultmerge.matrix.num.column";
	private static final String RESULTMERGE_BLOCK_NUM_ROW_CONFIG="resultmerge.block.num.row";
	private static final String RESULTMERGE_BLOCK_NUM_COLUMN_CONFIG="resultmerge.block.num.column";
	
	//operations performed in the reduer
	private static final String AGGREGATE_INSTRUCTIONS_CONFIG="aggregate.instructions.after.groupby.at";
	private static final String INSTRUCTIONS_IN_REDUCER_CONFIG="instructions.in.reducer";
	private static final String AGGREGATE_BINARY_INSTRUCTIONS_CONFIG="aggregate.binary.instructions";
	private static final String REBLOCK_INSTRUCTIONS_CONFIG="reblock.instructions";
	private static final String CSV_REBLOCK_INSTRUCTIONS_CONFIG="csv.reblock.instructions";
	private static final String CSV_WRITE_INSTRUCTIONS_CONFIG="csv.write.instructions";
	private static final String COMBINE_INSTRUCTIONS_CONFIG="combine.instructions";
	private static final String CM_N_COV_INSTRUCTIONS_CONFIG="cm_n_com.instructions";
	private static final String GROUPEDAGG_INSTRUCTIONS_CONFIG="groupedagg.instructions";
	
	//characteristics about the matrices to aggregate binary instructions
	private static final String AGGBIN_MATRIX_NUM_ROW_PREFIX_CONFIG="aggbin.matrix.num.row.";
	private static final String AGGBIN_MATRIX_NUM_COLUMN_PREFIX_CONFIG="aggbin.matrix.num.column.";
	private static final String AGGBIN_BLOCK_NUM_ROW_PREFIX_CONFIG="aggbin.block.num.row.";
	private static final String AGGBIN_BLOCK_NUM_COLUMN_PREFIX_CONFIG="aggbin.block.num.column.";
	
	//characteristics about the matrices to outputs
	private static final String OUTPUT_MATRIX_NUM_ROW_PREFIX_CONFIG="output.matrix.num.row.";
	private static final String OUTPUT_MATRIX_NUM_COLUMN_PREFIX_CONFIG="output.matrix.num.column.";
	private static final String OUTPUT_BLOCK_NUM_ROW_PREFIX_CONFIG="output.block.num.row.";
	private static final String OUTPUT_BLOCK_NUM_COLUMN_PREFIX_CONFIG="output.block.num.column.";
	
	//characteristics about the matrices to reblock instructions
	private static final String REBLOCK_MATRIX_NUM_ROW_PREFIX_CONFIG="reblock.matrix.num.row.";
	private static final String REBLOCK_MATRIX_NUM_COLUMN_PREFIX_CONFIG="reblock.matrix.num.column.";
	private static final String REBLOCK_BLOCK_NUM_ROW_PREFIX_CONFIG="reblock.block.num.row.";
	private static final String REBLOCK_BLOCK_NUM_COLUMN_PREFIX_CONFIG="reblock.block.num.column.";
	private static final String REBLOCK_MATRIX_NUM_NNZ_PREFIX_CONFIG="reblock.matrix.num.nnz.";
	
	
	//characteristics about the matrices to matrixdiag instructions
	private static final String INTERMEDIATE_MATRIX_NUM_ROW_PREFIX_CONFIG="rdiag.matrix.num.row.";
	private static final String INTERMEDIATE_MATRIX_NUM_COLUMN_PREFIX_CONFIG="rdiag.matrix.num.column.";
	private static final String INTERMEDIATE_BLOCK_NUM_ROW_PREFIX_CONFIG="rdiag.block.num.row.";
	private static final String INTERMEDIATE_BLOCK_NUM_COLUMN_PREFIX_CONFIG="rdiag.block.num.column.";
	
	//matrix indexes to be outputted as final results
	private static final String RESULT_INDEXES_CONFIG="results.indexes";
	private static final String RESULT_DIMS_UNKNOWN_CONFIG="results.dims.unknown";
	
	private static final String INTERMEDIATE_INDEXES_CONFIG="rdiag.indexes";
	
	//output matrices and their formats
	public static final String OUTPUT_MATRICES_DIRS_CONFIG="output.matrices.dirs";
	private static final String OUTPUT_CONVERTER_CLASS_PREFIX_CONFIG="output.converter.class.for.";
	
	private static final String DIMS_UNKNOWN_FILE_PREFIX = "dims.unknown.file.prefix";
	
	private static final String MMCJ_CACHE_SIZE="mmcj.cache.size";
	
	private static final String DISTCACHE_INPUT_INDICES="distcache.input.indices";
	private static final String DISTCACHE_INPUT_PATHS = "distcache.input.paths";
	
	private static final String SYSTEMML_LOCAL_TMP_DIR = "systemml.local.tmp.dir";
	
	/*
	 * SystemML Counter Group names
	 * 
	 * group name for the counters on number of output nonZeros
	 */
	public static final String NUM_NONZERO_CELLS="nonzeros";

	public static final int getMiscMemRequired(JobConf job)
	{
		return job.getInt(MRConfigurationNames.IO_FILE_BUFFER_SIZE, 4096);
	}

	public static void setMMCJCacheSize(JobConf job, long size)
	{
		job.setLong(MMCJ_CACHE_SIZE, size);
	}
	
	public static long getMMCJCacheSize(JobConf job)
	{
		return job.getLong(MMCJ_CACHE_SIZE, 0);
	}
	
	public static void setMatrixValueClass(JobConf job, boolean blockRepresentation)
	{
		job.setBoolean(BLOCK_REPRESENTATION_CONFIG, blockRepresentation);
	}
	
	public static void setMatrixValueClassForCM_N_COM(JobConf job, boolean weightedCellRepresentation)
	{
		job.setBoolean(WEIGHTEDCELL_REPRESENTATION_CONFIG, weightedCellRepresentation);
	}
	
	public static Class<? extends MatrixValue> getMatrixValueClass(JobConf job)
	{
		if(job.getBoolean(WEIGHTEDCELL_REPRESENTATION_CONFIG, false))
			return WeightedPair.class;
		
		if(job.getBoolean(BLOCK_REPRESENTATION_CONFIG, true))
			return MatrixBlock.class;
		else
			return MatrixCell.class;
	}

	public static enum ConvertTarget{CELL, BLOCK, WEIGHTEDCELL, CSVWRITE}
	
	public static Class<? extends Converter> getConverterClass(InputInfo inputinfo, int brlen, int bclen, ConvertTarget target)
	{

		Class<? extends Converter> converterClass=IdenticalConverter.class;
		if(inputinfo.inputValueClass.equals(MatrixCell.class))
		{
			switch (target)
			{
			case CELL:
				converterClass=IdenticalConverter.class;
				break;
			case BLOCK:
				throw new RuntimeException("cannot convert binary cell to binary block representation implicitly");
			case WEIGHTEDCELL:
				converterClass=AddDummyWeightConverter.class;
				break;
			case CSVWRITE:
				converterClass=BinaryCellToRowBlockConverter.class;
				break;
			}
			
		}else if(inputinfo.inputValueClass.equals(MatrixBlock.class))
		{
			switch (target)
			{
			case CELL:
				converterClass=BinaryBlockToBinaryCellConverter.class;
				break;
			case BLOCK:
				converterClass=IdenticalConverter.class;
				break;
			case WEIGHTEDCELL:
				converterClass=AddDummyWeightConverter.class;
				break;
			case CSVWRITE:
				converterClass=BinaryBlockToRowBlockConverter.class;
				break;
			}
		}else if(inputinfo.inputValueClass.equals(Text.class))
		{
			switch (target)
			{
			case CELL:
				converterClass=TextToBinaryCellConverter.class;
				break;
			case BLOCK:
				if(brlen>1 || bclen>1)
					throw new RuntimeException("cannot convert text cell to binary block representation implicitly");
				else
					converterClass=TextToBinaryCellConverter.class;
				break;
			case WEIGHTEDCELL:
				converterClass=AddDummyWeightConverter.class;
				break;
			case CSVWRITE:
				converterClass=TextCellToRowBlockConverter.class;
				break;
			}
		}
	
		return converterClass;
	}
	
	/**
	 * Unique working dirs required for thread-safe submission of parallel jobs;
	 * otherwise job.xml and other files might be overridden (in local mode).
	 * 
	 * @param job job configuration
	 */
	public static void setUniqueWorkingDir( JobConf job )
	{
		if( InfrastructureAnalyzer.isLocalMode(job) )
		{
			StringBuilder tmp = new StringBuilder();
			tmp.append( Lop.FILE_SEPARATOR );
			tmp.append( Lop.PROCESS_PREFIX );
			tmp.append( DMLScript.getUUID() );
			tmp.append( Lop.FILE_SEPARATOR );
			tmp.append( seq.getNextID() );
			String uniqueSubdir = tmp.toString();
			
			//unique local dir
			String[] dirlist = job.get(MRConfigurationNames.MR_CLUSTER_LOCAL_DIR,"/tmp").split(",");
			StringBuilder sb2 = new StringBuilder();
			for( String dir : dirlist ) {
				if( sb2.length()>0 )
					sb2.append(",");
				sb2.append(dir);
				sb2.append( uniqueSubdir );
			}
			job.set(MRConfigurationNames.MR_CLUSTER_LOCAL_DIR, sb2.toString() );
			
			//unique system dir 
			job.set(MRConfigurationNames.MR_JOBTRACKER_SYSTEM_DIR, job.get(MRConfigurationNames.MR_JOBTRACKER_SYSTEM_DIR) + uniqueSubdir);
			
			//unique staging dir 
			job.set( MRConfigurationNames.MR_JOBTRACKER_STAGING_ROOT_DIR,  job.get(MRConfigurationNames.MR_JOBTRACKER_STAGING_ROOT_DIR) + uniqueSubdir );
		}
	}
	
	public static String getLocalWorkingDirPrefix(JobConf job)
	{
		return job.get(MRConfigurationNames.MR_CLUSTER_LOCAL_DIR);
	}
	
	public static String getSystemWorkingDirPrefix(JobConf job)
	{
		return job.get(MRConfigurationNames.MR_JOBTRACKER_SYSTEM_DIR);
	}
	
	public static String getStagingWorkingDirPrefix(JobConf job)
	{
		return job.get(MRConfigurationNames.MR_JOBTRACKER_STAGING_ROOT_DIR);
	}

	public static void setInputInfo(JobConf job, byte input, InputInfo inputinfo, 
			int brlen, int bclen, ConvertTarget target)
	{
		Class<? extends Converter> converterClass=getConverterClass(inputinfo, brlen, bclen, target);
		job.setClass(INPUT_CONVERTER_CLASS_PREFIX_CONFIG+input, converterClass, Converter.class);
		job.setClass(INPUT_KEY_CLASS_PREFIX_CONFIG+input, inputinfo.inputKeyClass, Writable.class);
		job.setClass(INPUT_VALUE_CLASS_PREFIX_CONFIG+input, inputinfo.inputValueClass, Writable.class);
	}
	
	
	public static void setOutputInfo(JobConf job, int i, OutputInfo outputinfo, boolean sourceInBlock) {
		Class<? extends Converter> converterClass;
		if(sourceInBlock)
		{
			if(outputinfo.outputValueClass.equals(MatrixCell.class))
				converterClass=BinaryBlockToBinaryCellConverter.class;
			else if(outputinfo.outputValueClass.equals(Text.class))
				converterClass=BinaryBlockToTextCellConverter.class;
			else if(outputinfo.outputValueClass.equals(MatrixBlock.class))
				converterClass=IdenticalConverter.class;
			else if(outputinfo.outputValueClass.equals(IntWritable.class))
				converterClass=WeightedCellToSortInputConverter.class;
			else if(outputinfo.outputValueClass.equals(WeightedPair.class))
				converterClass=IdenticalConverter.class;
			else
				converterClass=IdenticalConverter.class;
		}else
		{
			if(outputinfo.outputValueClass.equals(MatrixCell.class))
				converterClass=IdenticalConverter.class;
			else if(outputinfo.outputValueClass.equals(Text.class))
				converterClass=BinaryCellToTextConverter.class;
			else if(outputinfo.outputValueClass.equals(IntWritable.class))
				converterClass=WeightedCellToSortInputConverter.class;
			else if(outputinfo.outputValueClass.equals(WeightedPair.class))
				converterClass=IdenticalConverter.class;
			else
				throw new DMLRuntimeException("unsupported conversion: " + outputinfo.outputValueClass);
				// converterClass=IdenticalConverter.class; 
		}
		job.setClass(OUTPUT_CONVERTER_CLASS_PREFIX_CONFIG+i, converterClass, Converter.class);
	}
	
	public static Converter getInputConverter(JobConf job, byte input)
	{
		Converter inputConverter;
		try {
			inputConverter=(Converter) job.getClass(INPUT_CONVERTER_CLASS_PREFIX_CONFIG+input, 
					IdenticalConverter.class).newInstance();
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 
		return inputConverter;
	}
	
	public static Converter getOuputConverter(JobConf job, int i)
	{
		Converter outputConverter;
		try {
			outputConverter=(Converter) job.getClass(OUTPUT_CONVERTER_CLASS_PREFIX_CONFIG+i, 
					IdenticalConverter.class).newInstance();
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 
		return outputConverter;
	}
	
	//parfor configurations
	public static void setProgramBlocks(JobConf job, String sProgramBlocks) 
	{
		job.set(PARFOR_PROGRAMBLOCKS_CONFIG, sProgramBlocks);
	}
	
	public static String getProgramBlocks(JobConf job) 
	{
		String str = job.get(PARFOR_PROGRAMBLOCKS_CONFIG);
		return str;
	}
	
	public static void setParforCachingConfig(JobConf job, boolean flag)
	{
		job.setBoolean(PARFOR_CACHING_CONFIG, flag);
	}
	
	public static boolean getParforCachingConfig(JobConf job) 
	{
		return job.getBoolean(PARFOR_CACHING_CONFIG, true);
	}
	
	//partitioning configurations
	public static void setPartitioningInfo( JobConf job, long rlen, long clen, int brlen, int bclen, InputInfo ii, OutputInfo oi, PDataPartitionFormat dpf, int n, String fnameNew ){
		job.set(PARTITIONING_INPUT_MATRIX_NUM_ROW_CONFIG, String.valueOf(rlen));
		job.set(PARTITIONING_INPUT_MATRIX_NUM_COLUMN_CONFIG, String.valueOf(clen));
		job.set(PARTITIONING_INPUT_BLOCK_NUM_ROW_CONFIG, String.valueOf(brlen));
		job.set(PARTITIONING_INPUT_BLOCK_NUM_COLUMN_CONFIG, String.valueOf(bclen));
		job.set(PARTITIONING_INPUT_INFO_CONFIG, InputInfo.inputInfoToString(ii));
		job.set(PARTITIONING_OUTPUT_INFO_CONFIG, OutputInfo.outputInfoToString(oi));
		job.set(PARTITIONING_OUTPUT_FORMAT_CONFIG, dpf.toString());
		job.set(PARTITIONING_OUTPUT_N_CONFIG, String.valueOf(n));
		job.set(PARTITIONING_OUTPUT_FILENAME_CONFIG, fnameNew);
	}
	
	public static void setPartitioningInfo( JobConf job, long rlen, long clen, int brlen, int bclen, InputInfo ii, OutputInfo oi, PDataPartitionFormat dpf, int n, String fnameNew, String itervar, String matrixvar, boolean tSparseCol ) {
		//set basic partitioning information
		setPartitioningInfo(job, rlen, clen, brlen, bclen, ii, oi, dpf, n, fnameNew);
		
		//set iteration variable name (used for ParFor-DPE)
		job.set(PARTITIONING_ITERVAR_CONFIG, itervar);
		
		//set iteration variable name (used for ParFor-DPE)
		job.set(PARTITIONING_MATRIXVAR_CONFIG, matrixvar);
		
		//set transpose sparse column vector
		job.setBoolean(PARTITIONING_TRANSPOSE_COL_CONFIG, tSparseCol);		
	}
	
	public static void setPartitioningInfo( JobConf job, long rlen, long clen, int brlen, int bclen, InputInfo ii, OutputInfo oi, PDataPartitionFormat dpf, int n, String fnameNew, boolean keepIndexes ) {
		//set basic partitioning information
		setPartitioningInfo(job, rlen, clen, brlen, bclen, ii, oi, dpf, n, fnameNew);
		//set transpose sparse column vector
		job.setBoolean(PARTITIONING_OUTPUT_KEEP_INDEXES_CONFIG, keepIndexes);
	}
	
	public static MatrixCharacteristics getPartitionedMatrixSize(JobConf job) {
		return new MatrixCharacteristics(
			Long.parseLong(job.get(PARTITIONING_INPUT_MATRIX_NUM_ROW_CONFIG)),
			Long.parseLong(job.get(PARTITIONING_INPUT_MATRIX_NUM_COLUMN_CONFIG)),
			Integer.parseInt(job.get(PARTITIONING_INPUT_BLOCK_NUM_ROW_CONFIG)),
			Integer.parseInt(job.get(PARTITIONING_INPUT_BLOCK_NUM_COLUMN_CONFIG)));
	}
	
	
	public static void setPartitioningBlockNumRows( JobConf job, int brlen ) {
		job.set(PARTITIONING_INPUT_BLOCK_NUM_ROW_CONFIG, String.valueOf(brlen));
	}
	
	public static void setPartitioningBlockNumCols( JobConf job, int bclen ) {
		job.set(PARTITIONING_INPUT_BLOCK_NUM_COLUMN_CONFIG,String.valueOf(bclen));
	}
	
	public static InputInfo getPartitioningInputInfo( JobConf job ) {
		return InputInfo.stringToInputInfo(job.get(PARTITIONING_INPUT_INFO_CONFIG));
	}
	
	public static OutputInfo getPartitioningOutputInfo( JobConf job ) {
		return OutputInfo.stringToOutputInfo(job.get(PARTITIONING_OUTPUT_INFO_CONFIG));
	}
	
	public static void setPartitioningFormat( JobConf job, PDataPartitionFormat dpf ) {
		job.set(PARTITIONING_OUTPUT_FORMAT_CONFIG, dpf.toString());
	}
	
	public static PDataPartitionFormat getPartitioningFormat( JobConf job )	{
		return PDataPartitionFormat.valueOf(job.get(PARTITIONING_OUTPUT_FORMAT_CONFIG));
	}
	
	public static int getPartitioningSizeN( JobConf job ) {
		return Integer.parseInt(job.get(PARTITIONING_OUTPUT_N_CONFIG));
	}
	
	public static boolean getPartitioningIndexFlag( JobConf job )
	{
		return Boolean.parseBoolean(job.get(PARTITIONING_OUTPUT_KEEP_INDEXES_CONFIG));
	}
	
	public static void setPartitioningFilename( JobConf job, String fname )
	{
		job.set(PARTITIONING_OUTPUT_FILENAME_CONFIG, fname);
	}
	
	public static String getPartitioningFilename( JobConf job )
	{
		return job.get(PARTITIONING_OUTPUT_FILENAME_CONFIG);
	}
	
	public static String getPartitioningItervar( JobConf job )
	{
		return job.get(PARTITIONING_ITERVAR_CONFIG);
	}
	
	public static String getPartitioningMatrixvar( JobConf job )
	{
		return job.get(PARTITIONING_MATRIXVAR_CONFIG);
	}
	
	public static boolean getPartitioningTransposedCol( JobConf job )
	{
		return job.getBoolean(PARTITIONING_TRANSPOSE_COL_CONFIG, false);
	}
	
	public static void setResultMergeInfo( JobConf job, String fnameNew, boolean accum, InputInfo ii, String stagingDir, long rlen, long clen, int brlen, int bclen ) {
		job.set(RESULTMERGE_COMPARE_FILENAME_CONFIG, fnameNew);
		job.set(RESULTMERGE_ACCUMULATOR_CONFIG, String.valueOf(accum));
		job.set(RESULTMERGE_INPUT_INFO_CONFIG, InputInfo.inputInfoToString(ii));
		job.set(RESULTMERGE_STAGING_DIR_CONFIG, stagingDir);
		job.set(RESULTMERGE_MATRIX_NUM_ROW_CONFIG, String.valueOf(rlen));
		job.set(RESULTMERGE_MATRIX_NUM_COLUMN_CONFIG, String.valueOf(clen));
		job.set(RESULTMERGE_BLOCK_NUM_ROW_CONFIG, String.valueOf(brlen));
		job.set(RESULTMERGE_BLOCK_NUM_COLUMN_CONFIG, String.valueOf(bclen));
	}
	
	public static String getResultMergeInfoCompareFilename( JobConf job ) {
		return job.get(RESULTMERGE_COMPARE_FILENAME_CONFIG);
	}
	
	public static boolean getResultMergeInfoAccumulator( JobConf job ) {
		return Boolean.parseBoolean(job.get(RESULTMERGE_ACCUMULATOR_CONFIG));
	}
	
	public static InputInfo getResultMergeInputInfo( JobConf job )
	{
		return InputInfo.stringToInputInfo( job.get(RESULTMERGE_INPUT_INFO_CONFIG) );
	}

	public static long[] getResultMergeMatrixCharacteristics( JobConf job )
	{
		long[] ret = new long[4];
		ret[0] = Long.parseLong(job.get(RESULTMERGE_MATRIX_NUM_ROW_CONFIG));
		ret[1] = Long.parseLong(job.get(RESULTMERGE_MATRIX_NUM_COLUMN_CONFIG));
		ret[2] = Long.parseLong(job.get(RESULTMERGE_BLOCK_NUM_ROW_CONFIG));
		ret[3] = Long.parseLong(job.get(RESULTMERGE_BLOCK_NUM_COLUMN_CONFIG));
		
		return ret;
	}
	
	public static byte[] getInputIndexesInMapper(JobConf job)
	{
		String[] istrs=job.get(MAPFUNC_INPUT_MATRICIES_INDEXES_CONFIG).split(Instruction.INSTRUCTION_DELIM);
		return stringArrayToByteArray(istrs);
	}
	
	public static byte[] getOutputIndexesInMapper(JobConf job)
	{
		String[] istrs=job.get(OUTPUT_INDEXES_IN_MAPPER_CONFIG).split(Instruction.INSTRUCTION_DELIM);
		return stringArrayToByteArray(istrs);
	}
	
	//get the indexes that this matrix file represents, 
	//since one matrix file can occur multiple times in a statement
	public static ArrayList<Byte> getInputMatrixIndexesInMapper(JobConf job) throws IOException
	{
		String[] matrices=job.getStrings(INPUT_MATRICIES_DIRS_CONFIG);
		String str=job.get(MAPFUNC_INPUT_MATRICIES_INDEXES_CONFIG);
		byte[] indexes;
		if(str==null || str.isEmpty())
		{
			indexes=new byte[matrices.length];
			for(int i=0; i<indexes.length; i++)
				indexes[i]=(byte)i;
		}else
		{
			String[] strs=str.split(Instruction.INSTRUCTION_DELIM);
			indexes=new byte[strs.length];
			for(int i=0; i<strs.length; i++)
				indexes[i]=Byte.parseByte(strs[i]);
		}
		
		int numMatrices=matrices.length;
		if(numMatrices>Byte.MAX_VALUE)
			throw new RuntimeException("number of matrices is too large > "+Byte.MAX_VALUE);
		for(int i=0; i<matrices.length; i++)
			matrices[i]=new Path(matrices[i]).toString();
		
		Path thisFile=new Path(job.get(MRConfigurationNames.MR_MAP_INPUT_FILE));
		FileSystem fs = IOUtilFunctions.getFileSystem(thisFile, job);
		thisFile = thisFile.makeQualified(fs);
		
		Path thisDir=thisFile.getParent().makeQualified(fs);
		ArrayList<Byte> representativeMatrixes=new ArrayList<>();
		for(int i=0; i<matrices.length; i++)
		{
			Path p = new Path(matrices[i]).makeQualified(fs);
			if(thisFile.toUri().equals(p.toUri()) || thisDir.toUri().equals(p.toUri()))
				representativeMatrixes.add(indexes[i]);
		}
		return representativeMatrixes;
	}
	
	/*public static void setMatrixToCacheInMMCJ(JobConf job, boolean left)
	{
		job.setBoolean(CACHE_LEFT_MATRIX_FOR_MMCJ_CONFIG, left);
	}
	
	public static boolean getMatrixToCacheInMMCJ(JobConf job)
	{
		return job.getBoolean(CACHE_LEFT_MATRIX_FOR_MMCJ_CONFIG, true);
	}*/
	
	public static void setInstructionsInMapper(JobConf job, String instructionsInMapper)
	{
		job.set(INSTRUCTIONS_IN_MAPPER_CONFIG, instructionsInMapper);
	}
	
	public static void setAggregateInstructions(JobConf job, String aggInstructionsInReducer)
	{
		job.set(AGGREGATE_INSTRUCTIONS_CONFIG, aggInstructionsInReducer);
	}
	
	public static void setReblockInstructions(JobConf job, String reblockInstructions)
	{
		job.set(REBLOCK_INSTRUCTIONS_CONFIG, reblockInstructions);
	}
	
	public static void setCSVReblockInstructions(JobConf job, String reblockInstructions)
	{
		job.set(CSV_REBLOCK_INSTRUCTIONS_CONFIG, reblockInstructions);
	}
	
	public static void setCSVWriteInstructions(JobConf job, String csvWriteInstructions)
	{
		job.set(CSV_WRITE_INSTRUCTIONS_CONFIG, csvWriteInstructions);
	}
	
	public static void setCombineInstructions(JobConf job, String combineInstructions)
	{
		job.set(COMBINE_INSTRUCTIONS_CONFIG, combineInstructions);
	}
	
	public static void setInstructionsInReducer(JobConf job, String instructionsInReducer)
	{
		if(instructionsInReducer!=null)
			job.set(INSTRUCTIONS_IN_REDUCER_CONFIG, instructionsInReducer);
	}
	
	public static void setAggregateBinaryInstructions(JobConf job, String aggBinInstrctions)
	{
		job.set(AGGREGATE_BINARY_INSTRUCTIONS_CONFIG, aggBinInstrctions);
	}
	public static void setCM_N_COMInstructions(JobConf job, String cmInstrctions)
	{
		job.set(CM_N_COV_INSTRUCTIONS_CONFIG, cmInstrctions);
	}
	public static void setGroupedAggInstructions(JobConf job, String grpaggInstructions)
	{
		job.set(GROUPEDAGG_INSTRUCTIONS_CONFIG, grpaggInstructions);
	}

	public static void setRandInstructions(JobConf job, String randInstrctions)
	{
		job.set(RAND_INSTRUCTIONS_CONFIG, randInstrctions);
	}

	public static String[] getOutputs(JobConf job) {
		return job.getStrings(OUTPUT_MATRICES_DIRS_CONFIG);
	}
	
	private static byte[] stringArrayToByteArray(String[] istrs) {
		byte[] ret=new byte[istrs.length];
		for(int i=0; i<istrs.length; i++)
			ret[i]=Byte.parseByte(istrs[i]);
		return ret;
	}
	
	public static byte[] getResultIndexes(JobConf job) {
		String[] istrs=job.get(RESULT_INDEXES_CONFIG).split(Instruction.INSTRUCTION_DELIM);
		return stringArrayToByteArray(istrs);
	}
	
	public static byte[] getResultDimsUnknown(JobConf job) {
		String str=job.get(RESULT_DIMS_UNKNOWN_CONFIG);
		if (str==null || str.isEmpty())
			return null;
		String[] istrs=str.split(Instruction.INSTRUCTION_DELIM);
		return stringArrayToByteArray(istrs);
	}
	
	public static byte[] getIntermediateMatrixIndexes(JobConf job) {
		String str=job.get(INTERMEDIATE_INDEXES_CONFIG);
		if(str==null || str.isEmpty())
			return null;
		String[] istrs=str.split(Instruction.INSTRUCTION_DELIM);
		return stringArrayToByteArray(istrs);
	}

	public static void setDimsUnknownFilePrefix(JobConf job, String prefix) {
		job.setStrings(DIMS_UNKNOWN_FILE_PREFIX, prefix);
	}
	
	public static void setMatricesDimensions(JobConf job, byte[] inputIndexes, long[] rlens, long[] clens) {
		if(rlens.length!=clens.length)
			throw new RuntimeException("rlens.length should be clens.length");
		for(int i=0; i<rlens.length; i++)
			setMatrixDimension(job, inputIndexes[i], rlens[i], clens[i]);
	}
	
	public static void setMatricesDimensions(JobConf job, byte[] inputIndexes, long[] rlens, long[] clens, long[] nnz) {
		if(rlens.length!=clens.length)
			throw new RuntimeException("rlens.length should be clens.length");
		for(int i=0; i<rlens.length; i++)
			setMatrixDimension(job, inputIndexes[i], rlens[i], clens[i], nnz[i]);
	}
	
	public static void setMatrixDimension(JobConf job, byte matrixIndex, long rlen, long clen)
	{
		job.setLong(INPUT_MATRIX_NUM_ROW_PREFIX_CONFIG+matrixIndex, rlen);
		job.setLong(INPUT_MATRIX_NUM_COLUMN_PREFIX_CONFIG+matrixIndex, clen);
	}
	
	public static void setMatrixDimension(JobConf job, byte matrixIndex, long rlen, long clen, long nnz)
	{
		job.setLong(INPUT_MATRIX_NUM_ROW_PREFIX_CONFIG+matrixIndex, rlen);
		job.setLong(INPUT_MATRIX_NUM_COLUMN_PREFIX_CONFIG+matrixIndex, clen);
		job.setLong(INPUT_MATRIX_NUM_NNZ_PREFIX_CONFIG+matrixIndex, nnz);
	}
	
	public static String[] getInputPaths(JobConf job)
	{
		return job.getStrings(INPUT_MATRICIES_DIRS_CONFIG);
	}
	
	public static long getNumRows(JobConf job, byte matrixIndex)
	{
		return job.getLong(INPUT_MATRIX_NUM_ROW_PREFIX_CONFIG+matrixIndex, 0);
	}
	
	public static long getNumColumns(JobConf job, byte matrixIndex)
	{
		return job.getLong(INPUT_MATRIX_NUM_COLUMN_PREFIX_CONFIG+matrixIndex, 0);
	}	
	
	public static void setBlocksSizes(JobConf job, byte[] inputIndexes, int[] brlens, int[] bclens) {
		if(brlens.length!=bclens.length)
			throw new RuntimeException("brlens.length should be bclens.length");
		for(int i=0; i<brlens.length; i++)
			setBlockSize(job, inputIndexes[i], brlens[i], bclens[i]);
	}
	
	public static void setBlockSize(JobConf job, byte matrixIndex, int brlen, int bclen)
	{
		job.setInt(INPUT_BLOCK_NUM_ROW_PREFIX_CONFIG+matrixIndex, brlen);
		job.setInt(INPUT_BLOCK_NUM_COLUMN_PREFIX_CONFIG+matrixIndex, bclen);
	}
	
	public static int getNumRowsPerBlock(JobConf job, byte matrixIndex)
	{
		return job.getInt(INPUT_BLOCK_NUM_ROW_PREFIX_CONFIG+matrixIndex, 1);
	}
	
	public static int getNumColumnsPerBlock(JobConf job, byte matrixIndex)
	{
		return job.getInt(INPUT_BLOCK_NUM_COLUMN_PREFIX_CONFIG+matrixIndex, 1);
	}
	
	public static long getNumNonZero(JobConf job, byte matrixIndex)
	{
		return job.getLong(INPUT_MATRIX_NUM_NNZ_PREFIX_CONFIG+matrixIndex, 1);
	}

	public static void setupDistCacheInputs(JobConf job, String indices, String pathsString, ArrayList<String> paths) {
		job.set(DISTCACHE_INPUT_INDICES, indices);
		job.set(DISTCACHE_INPUT_PATHS, pathsString);
		Path p = null;
		
		if( !InfrastructureAnalyzer.isLocalMode(job) ) {
			for(String spath : paths) {
				p = new Path(spath);
				
				DistributedCache.addCacheFile(p.toUri(), job);
				DistributedCache.createSymlink(job);
			}
		}
	}
	
	public static String getDistCacheInputIndices(JobConf job) {
		return job.get(DISTCACHE_INPUT_INDICES);
	}

	private static String getCSVString(PDataPartitionFormat[] formats) {
		if ( formats == null || formats.length == 0 )
			return "";
		
		StringBuilder s = new StringBuilder();
		s.append(formats[0]);
		for(int i=1; i < formats.length; i++) { 
			s.append(",");
			s.append(formats[i]);
		}
		
		return s.toString();
	}
	
	public static void setInputPartitioningInfo(JobConf job, PDataPartitionFormat[] pformats) {
		job.set(PARTITIONING_OUTPUT_FORMAT_CONFIG, MRJobConfiguration.getCSVString(pformats));
	}

	private static PDataPartitionFormat[] csv2PFormat(String s) {
		String[] parts = s.split(",");
		PDataPartitionFormat[] pformats = new PDataPartitionFormat[parts.length];
		for(int i=0; i < parts.length; i++) {
			pformats[i] = PDataPartitionFormat.parsePDataPartitionFormat(parts[i]);
		}
		return pformats;
	}

	public static PDataPartitionFormat[] getInputPartitionFormats(JobConf job) {
		return MRJobConfiguration.csv2PFormat(job.get(PARTITIONING_OUTPUT_FORMAT_CONFIG));
	}
	
	public static class MatrixChar_N_ReducerGroups
	{
		public MatrixCharacteristics[] stats;
		public long numReducerGroups=0;
		
		public MatrixChar_N_ReducerGroups(MatrixCharacteristics[] sts, long ng)
		{
			stats=sts;
			numReducerGroups=ng;
		}
	}
	
	
	public static void setIntermediateMatrixCharactristics(JobConf job,
			byte tag, MatrixCharacteristics dim) {
		
		job.setLong(INTERMEDIATE_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, dim.getRows());
		job.setLong(INTERMEDIATE_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, dim.getCols());
		job.setInt(INTERMEDIATE_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, dim.getRowsPerBlock());
		job.setInt(INTERMEDIATE_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, dim.getColsPerBlock());
	}
	
	public static MatrixCharacteristics getIntermediateMatrixCharactristics(JobConf job, byte tag)
	{
		MatrixCharacteristics dim=new MatrixCharacteristics();
		dim.setDimension( job.getLong(INTERMEDIATE_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, 0),
		                  job.getLong(INTERMEDIATE_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, 0) );
		dim.setBlockSize( job.getInt(INTERMEDIATE_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, 1),
		                  job.getInt(INTERMEDIATE_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, 1) );
		return dim;
	}

	public static void setMatrixCharactristicsForOutput(JobConf job,
			byte tag, MatrixCharacteristics dim)
	{
		job.setLong(OUTPUT_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, dim.getRows());
		job.setLong(OUTPUT_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, dim.getCols());
		job.setInt(OUTPUT_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, dim.getRowsPerBlock());
		job.setInt(OUTPUT_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, dim.getColsPerBlock());
	}
	
	public static MatrixCharacteristics getMatrixCharacteristicsForOutput(JobConf job, byte tag)
	{
		MatrixCharacteristics dim=new MatrixCharacteristics();
		dim.setDimension( job.getLong(OUTPUT_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, 0),
		                  job.getLong(OUTPUT_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, 0) );
		dim.setBlockSize( job.getInt(OUTPUT_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, 1), 
		                  job.getInt(OUTPUT_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, 1) );
		return dim;
	}
	
	public static MatrixCharacteristics getMatrixCharacteristicsForInput(JobConf job, byte tag)
	{
		MatrixCharacteristics dim=new MatrixCharacteristics();
		dim.setDimension( job.getLong(INPUT_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, 0),
		                  job.getLong(INPUT_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, 0) );
		dim.setBlockSize( job.getInt(INPUT_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, 1),
		                  job.getInt(INPUT_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, 1) );		
		return dim;
	}
	
	public static void setMatrixCharactristicsForMapperOutput(JobConf job,
		byte tag, MatrixCharacteristics dim)
	{
		job.setLong(MAPOUTPUT_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, dim.getRows());
		job.setLong(MAPOUTPUT_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, dim.getCols());
		job.setInt(MAPOUTPUT_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, dim.getRowsPerBlock());
		job.setInt(MAPOUTPUT_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, dim.getColsPerBlock());
	}
	 	
	public static MatrixCharacteristics getMatrixCharacteristicsForMapOutput(JobConf job, byte tag)
	{
		MatrixCharacteristics dim=new MatrixCharacteristics();
		dim.setDimension( job.getLong(MAPOUTPUT_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, 0),
		job.getLong(MAPOUTPUT_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, 0) );
		dim.setBlockSize( job.getInt(MAPOUTPUT_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, 1), 
		job.getInt(MAPOUTPUT_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, 1) );
		return dim;
	}

	public static void setMatrixCharactristicsForReblock(JobConf job,
			byte tag, MatrixCharacteristics dim)
	{
		job.setLong(REBLOCK_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, dim.getRows());
		job.setLong(REBLOCK_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, dim.getCols());
		job.setInt(REBLOCK_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, dim.getRowsPerBlock());
		job.setInt(REBLOCK_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, dim.getColsPerBlock());
		job.setLong(REBLOCK_MATRIX_NUM_NNZ_PREFIX_CONFIG+tag, dim.getNonZeros());
	}
	
	public static MatrixCharacteristics getMatrixCharactristicsForReblock(JobConf job, byte tag)
	{
		MatrixCharacteristics dim=new MatrixCharacteristics();
		dim.setDimension( job.getLong(REBLOCK_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, 0),
		                 job.getLong(REBLOCK_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, 0) );
		dim.setBlockSize( job.getInt(REBLOCK_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, 1),
						 job.getInt(REBLOCK_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, 1));
		
		long nnz = job.getLong(REBLOCK_MATRIX_NUM_NNZ_PREFIX_CONFIG+tag, -1);
		if( nnz>=0 )
			dim.setNonZeros( nnz );
		
		return dim;
	}
	
	public static void setMatrixCharactristicsForBinAgg(JobConf job,
			byte tag, MatrixCharacteristics dim) {
		job.setLong(AGGBIN_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, dim.getRows());
		job.setLong(AGGBIN_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, dim.getCols());
		job.setInt(AGGBIN_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, dim.getRowsPerBlock());
		job.setInt(AGGBIN_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, dim.getColsPerBlock());
	}
	
	public static MatrixCharacteristics getMatrixCharactristicsForBinAgg(JobConf job, byte tag)
	{
		MatrixCharacteristics dim=new MatrixCharacteristics();
		dim.setDimension( job.getLong(AGGBIN_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, 0),
		                  job.getLong(AGGBIN_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, 0) );
		dim.setBlockSize( job.getInt(AGGBIN_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, 1),
		                  job.getInt(AGGBIN_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, 1) );
		return dim;
	}

	public static boolean deriveRepresentation(InputInfo[] inputInfos) {
		for(InputInfo input: inputInfos)
		{
			if(!(input.inputValueClass==MatrixBlock.class))
			{
				return false;
			}	
		}
		return true;
	}
	
	public static String constructTempOutputFilename() 
	{
		StringBuilder sb = new StringBuilder();
		sb.append(ConfigurationManager.getScratchSpace());
		sb.append(Lop.FILE_SEPARATOR);
		sb.append(Lop.PROCESS_PREFIX);
		sb.append(DMLScript.getUUID());
		sb.append(Lop.FILE_SEPARATOR);
		
		sb.append("TmpOutput"+seq.getNextID());
		
		//old unique dir (no guarantees): 
		//sb.append(Integer.toHexString(new Random().nextInt(Integer.MAX_VALUE))); 
		
		return sb.toString(); 
	}
	
	public static void setSystemMLLocalTmpDir(JobConf job, String dir)
	{
		job.set(SYSTEMML_LOCAL_TMP_DIR, dir);
	}
	
	public static String getSystemMLLocalTmpDir(JobConf job)
	{
		return job.get(SYSTEMML_LOCAL_TMP_DIR);
	}
	
	public static void addBinaryBlockSerializationFramework( Configuration job )
	{
		String frameworkList = job.get(MRConfigurationNames.IO_SERIALIZATIONS);
		String frameworkClassBB = BinaryBlockSerialization.class.getCanonicalName();
		job.set(MRConfigurationNames.IO_SERIALIZATIONS, frameworkClassBB+","+frameworkList);
	}
	
	/**
	 * Set all configurations with prefix mapred or mapreduce that exist in the given
	 * DMLConfig into the given JobConf.
	 * 
	 * @param job job configuration
	 * @param config dml configuration
	 */
	public static void setupCustomMRConfigurations( JobConf job, DMLConfig config ) {
		Map<String,String> map = config.getCustomMRConfig();
		for( Entry<String,String> e : map.entrySet() ) {
			job.set(e.getKey(), e.getValue());
		}
	}
}