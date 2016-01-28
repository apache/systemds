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
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.lib.CombineSequenceFileInputFormat;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.hadoop.mapred.lib.NullOutputFormat;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.MRInstructionParser;
import org.apache.sysml.runtime.instructions.mr.AggregateBinaryInstruction;
import org.apache.sysml.runtime.instructions.mr.AggregateInstruction;
import org.apache.sysml.runtime.instructions.mr.AppendGInstruction;
import org.apache.sysml.runtime.instructions.mr.AppendMInstruction;
import org.apache.sysml.runtime.instructions.mr.BinaryMInstruction;
import org.apache.sysml.runtime.instructions.mr.CM_N_COVInstruction;
import org.apache.sysml.runtime.instructions.mr.CSVReblockInstruction;
import org.apache.sysml.runtime.instructions.mr.CSVWriteInstruction;
import org.apache.sysml.runtime.instructions.mr.DataGenMRInstruction;
import org.apache.sysml.runtime.instructions.mr.GroupedAggregateInstruction;
import org.apache.sysml.runtime.instructions.mr.MRInstruction;
import org.apache.sysml.runtime.instructions.mr.MapMultChainInstruction;
import org.apache.sysml.runtime.instructions.mr.PMMJMRInstruction;
import org.apache.sysml.runtime.instructions.mr.ReblockInstruction;
import org.apache.sysml.runtime.instructions.mr.RemoveEmptyMRInstruction;
import org.apache.sysml.runtime.instructions.mr.UnaryMRInstructionBase;
import org.apache.sysml.runtime.io.BinaryBlockSerialization;
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
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.MultipleOutputCommitter;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.data.TextCellToRowBlockConverter;
import org.apache.sysml.runtime.matrix.data.TextToBinaryCellConverter;
import org.apache.sysml.runtime.matrix.data.WeightedCellToSortInputConverter;
import org.apache.sysml.runtime.matrix.data.WeightedPair;
import org.apache.sysml.runtime.matrix.data.hadoopfix.MultipleInputs;
import org.apache.sysml.runtime.matrix.sort.SamplingSortMRInputFormat;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.yarn.ropt.YarnClusterAnalyzer;

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
	private static final String RESULTMERGE_STAGING_DIR_CONFIG="resultmerge.staging.dir";
	private static final String RESULTMERGE_MATRIX_NUM_ROW_CONFIG="resultmerge.matrix.num.row";
	private static final String RESULTMERGE_MATRIX_NUM_COLUMN_CONFIG="resultmerge.matrix.num.column";
	private static final String RESULTMERGE_BLOCK_NUM_ROW_CONFIG="resultmerge.block.num.row";
	private static final String RESULTMERGE_BLOCK_NUM_COLUMN_CONFIG="resultmerge.block.num.column";
	
	private static final String SORT_PARTITION_FILENAME = "sort.partition.filename";
	
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
	
	/*
	 * Counter group for determining the dimensions of result matrix. It is 
	 * useful in operations like ctable and groupedAgg, in which the dimensions 
	 * of result matrix are known only after computing the matrix.
	 */
	public static final String MAX_ROW_DIMENSION = "maxrows";
	public static final String MAX_COL_DIMENSION = "maxcols";
	
	public static final String PARFOR_NUMTASKS="numtasks";
	public static final String PARFOR_NUMITERATOINS="numiterations";
	
	public static final String TF_NUM_COLS 		= "transform.num.columns";
	public static final String TF_HAS_HEADER 	= "transform.has.header";
	public static final String TF_DELIM 		= "transform.field.delimiter";
	public static final String TF_NA_STRINGS 	= "transform.na.strings";
	public static final String TF_HEADER		= "transform.header.line";
	public static final String TF_SPEC_FILE 	= "transform.specification.file";
	public static final String TF_TMP_LOC    	= "transform.temp.location";
	public static final String TF_TRANSFORM     = "transform.omit.na.rows";
	
	public static final String TF_SMALLEST_FILE= "transform.smallest.file";
	public static final String TF_OFFSETS_FILE = "transform.offsets.file";
	public static final String TF_TXMTD_PATH   = "transform.txmtd.path";
	
	/*public static enum DataTransformJobProperty 
	{
		RCD_NUM_COLS("recode.num.columns");
		
		private final String name;
		private DataTransformJobProperty(String n) {
			name = n;
		}
	}*/
	
	public static enum DataTransformCounters { 
		TRANSFORMED_NUM_ROWS
	};
	
	public static final int getMiscMemRequired(JobConf job)
	{
		return job.getInt(MRConfigurationNames.IO_FILE_BUFFER_SIZE, 4096);
	}
	
	public static final int getJVMMaxMemSize(JobConf job)
	{
		String str=job.get(MRConfigurationNames.MR_CHILD_JAVA_OPTS);
		int start=str.indexOf("-Xmx");
		if(start<0)
			return 209715200; //default 200MB
		str=str.substring(start+4);
		int i=0;
		for(; i<str.length() && str.charAt(i)<='9' && str.charAt(i)>='0'; i++);
		int ret=Integer.parseInt(str.substring(0, i));
		if(i>=str.length())
			return ret;
		
		switch(str.charAt(i))
		{
		case 'k':
		case 'K':
			ret=ret*1024;
			break;
		case 'm':
		case 'M':
			ret=ret*1048576;
			break;
		case 'g':
		case 'G':
			ret=ret*1073741824;
			break;
			default:
		}
		return ret;
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
	 * @param job
	 * @param mode
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
	
	/**
	 * 
	 * @param job
	 */
	public static void setStagingDir( JobConf job )
	{
		String dir = DMLConfig.LOCAL_MR_MODE_STAGING_DIR + 
		             Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID() + Lop.FILE_SEPARATOR;
		job.set( MRConfigurationNames.MR_JOBTRACKER_STAGING_ROOT_DIR, dir );
	}

	public static void setInputInfo(JobConf job, byte input, InputInfo inputinfo, 
			int brlen, int bclen, ConvertTarget target)
	{
		Class<? extends Converter> converterClass=getConverterClass(inputinfo, brlen, bclen, target);
		job.setClass(INPUT_CONVERTER_CLASS_PREFIX_CONFIG+input, converterClass, Converter.class);
		job.setClass(INPUT_KEY_CLASS_PREFIX_CONFIG+input, inputinfo.inputKeyClass, Writable.class);
		job.setClass(INPUT_VALUE_CLASS_PREFIX_CONFIG+input, inputinfo.inputValueClass, Writable.class);
	}
	
	
	public static void setOutputInfo(JobConf job, int i, OutputInfo outputinfo, boolean sourceInBlock) 
		throws DMLRuntimeException
	{
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
	
	@SuppressWarnings("unchecked")
	public static Class<Writable> getInputKeyClass(JobConf job, byte input)
	{
		return (Class<Writable>) job.getClass(INPUT_KEY_CLASS_PREFIX_CONFIG+input, 
				MatrixIndexes.class);
	}
	
	@SuppressWarnings("unchecked")
	public static Class<Writable> getInputValueClass(JobConf job, byte input)
	{
		return (Class<Writable>) job.getClass(INPUT_VALUE_CLASS_PREFIX_CONFIG+input, 
				DoubleWritable.class);
	}
	
	public static MRInstruction[] getInstructionsInReducer(JobConf job) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		String str=job.get(INSTRUCTIONS_IN_REDUCER_CONFIG);
		MRInstruction[] mixed_ops = MRInstructionParser.parseMixedInstructions(str);
		return mixed_ops;
	}
	
	public static ReblockInstruction[] getReblockInstructions(JobConf job) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		String str=job.get(REBLOCK_INSTRUCTIONS_CONFIG);
		ReblockInstruction[] reblock_instructions = MRInstructionParser.parseReblockInstructions(str);
		return reblock_instructions;
	}
	
	public static CSVReblockInstruction[] getCSVReblockInstructions(JobConf job) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		String str=job.get(CSV_REBLOCK_INSTRUCTIONS_CONFIG);
		CSVReblockInstruction[] reblock_instructions = MRInstructionParser.parseCSVReblockInstructions(str);
		return reblock_instructions;
	}
	
	public static CSVWriteInstruction[] getCSVWriteInstructions(JobConf job) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		String str=job.get(CSV_WRITE_INSTRUCTIONS_CONFIG);
		CSVWriteInstruction[] reblock_instructions = MRInstructionParser.parseCSVWriteInstructions(str);
		return reblock_instructions;
	}
	
	public static AggregateInstruction[] getAggregateInstructions(JobConf job) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		String str=job.get(AGGREGATE_INSTRUCTIONS_CONFIG);
		AggregateInstruction[] agg_instructions = MRInstructionParser.parseAggregateInstructions(str);
		return agg_instructions;
	}
	
	public static MRInstruction[] getCombineInstruction(JobConf job) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		String str=job.get(COMBINE_INSTRUCTIONS_CONFIG);
		MRInstruction[] comb_instructions = MRInstructionParser.parseCombineInstructions(str);
		return comb_instructions;
	}
	
	public static MRInstruction[] getInstructionsInMapper(JobConf job) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		String str=job.get(INSTRUCTIONS_IN_MAPPER_CONFIG);
		MRInstruction[] instructions = MRInstructionParser.parseMixedInstructions(str);
		return instructions;
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
	public static void setPartitioningInfo( JobConf job, long rlen, long clen, int brlen, int bclen, InputInfo ii, OutputInfo oi, PDataPartitionFormat dpf, int n, String fnameNew )
		throws DMLRuntimeException
	{
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
	
	public static void setPartitioningInfo( JobConf job, long rlen, long clen, int brlen, int bclen, InputInfo ii, OutputInfo oi, PDataPartitionFormat dpf, int n, String fnameNew, String itervar, String matrixvar, boolean tSparseCol )
			throws DMLRuntimeException
	{
		//set basic partitioning information
		setPartitioningInfo(job, rlen, clen, brlen, bclen, ii, oi, dpf, n, fnameNew);
		
		//set iteration variable name (used for ParFor-DPE)
		job.set(PARTITIONING_ITERVAR_CONFIG, itervar);
		
		//set iteration variable name (used for ParFor-DPE)
		job.set(PARTITIONING_MATRIXVAR_CONFIG, matrixvar);
		
		//set transpose sparse column vector
		job.setBoolean(PARTITIONING_TRANSPOSE_COL_CONFIG, tSparseCol);		
	}
	
	public static void setPartitioningInfo( JobConf job, long rlen, long clen, int brlen, int bclen, InputInfo ii, OutputInfo oi, PDataPartitionFormat dpf, int n, String fnameNew, boolean keepIndexes )
			throws DMLRuntimeException
	{
		//set basic partitioning information
		setPartitioningInfo(job, rlen, clen, brlen, bclen, ii, oi, dpf, n, fnameNew);
		
		//set transpose sparse column vector
		job.setBoolean(PARTITIONING_OUTPUT_KEEP_INDEXES_CONFIG, keepIndexes);
				
	}
	
	public static long getPartitioningNumRows( JobConf job )
	{
		return Long.parseLong(job.get(PARTITIONING_INPUT_MATRIX_NUM_ROW_CONFIG));
	}
	
	public static long getPartitioningNumCols( JobConf job )
	{
		return Long.parseLong(job.get(PARTITIONING_INPUT_MATRIX_NUM_COLUMN_CONFIG));
	}
	
	public static void setPartitioningBlockNumRows( JobConf job, int brlen )
	{
		job.set(PARTITIONING_INPUT_BLOCK_NUM_ROW_CONFIG, String.valueOf(brlen));
	}
	
	public static int getPartitioningBlockNumRows( JobConf job )
	{
		return Integer.parseInt(job.get(PARTITIONING_INPUT_BLOCK_NUM_ROW_CONFIG));
	}

	public static void setPartitioningBlockNumCols( JobConf job, int bclen )
	{
		job.set(PARTITIONING_INPUT_BLOCK_NUM_COLUMN_CONFIG,String.valueOf(bclen));
	}
	
	public static int getPartitioningBlockNumCols( JobConf job )
	{
		return Integer.parseInt(job.get(PARTITIONING_INPUT_BLOCK_NUM_COLUMN_CONFIG));
	}
	
	public static InputInfo getPartitioningInputInfo( JobConf job )
	{
		return InputInfo.stringToInputInfo(job.get(PARTITIONING_INPUT_INFO_CONFIG));
	}
	
	public static OutputInfo getPartitioningOutputInfo( JobConf job )
	{
		return OutputInfo.stringToOutputInfo(job.get(PARTITIONING_OUTPUT_INFO_CONFIG));
	}

	public static void setPartitioningFormat( JobConf job, PDataPartitionFormat dpf )
	{
		job.set(PARTITIONING_OUTPUT_FORMAT_CONFIG, dpf.toString());
	}
	
	public static PDataPartitionFormat getPartitioningFormat( JobConf job )
	{
		return PDataPartitionFormat.valueOf(job.get(PARTITIONING_OUTPUT_FORMAT_CONFIG));
	}
	
	public static int getPartitioningSizeN( JobConf job )
	{
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
	
	public static void setResultMergeInfo( JobConf job, String fnameNew, InputInfo ii, String stagingDir, long rlen, long clen, int brlen, int bclen )
		throws DMLRuntimeException
	{
		job.set(RESULTMERGE_COMPARE_FILENAME_CONFIG, fnameNew);
		job.set(RESULTMERGE_INPUT_INFO_CONFIG, InputInfo.inputInfoToString(ii));
		job.set(RESULTMERGE_STAGING_DIR_CONFIG, stagingDir);
		job.set(RESULTMERGE_MATRIX_NUM_ROW_CONFIG, String.valueOf(rlen));
		job.set(RESULTMERGE_MATRIX_NUM_COLUMN_CONFIG, String.valueOf(clen));
		job.set(RESULTMERGE_BLOCK_NUM_ROW_CONFIG, String.valueOf(brlen));
		job.set(RESULTMERGE_BLOCK_NUM_COLUMN_CONFIG, String.valueOf(bclen));
	}
	
	public static String getResultMergeInfoCompareFilename( JobConf job )
	{
		return job.get(RESULTMERGE_COMPARE_FILENAME_CONFIG);
	}
	
	public static InputInfo getResultMergeInputInfo( JobConf job )
	{
		return InputInfo.stringToInputInfo( job.get(RESULTMERGE_INPUT_INFO_CONFIG) );
	}
	
	public static String getResultMergeStagingDir( JobConf job )
	{
		return job.get(RESULTMERGE_STAGING_DIR_CONFIG) + job.get("mapred.tip.id");
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
		
		FileSystem fs=FileSystem.get(job);
		Path thisFile=new Path(job.get("map.input.file")).makeQualified(fs);
		
		//Path p=new Path(thisFileName);
		
		Path thisDir=thisFile.getParent().makeQualified(fs);
		ArrayList<Byte> representativeMatrixes=new ArrayList<Byte>();
		for(int i=0; i<matrices.length; i++)
		{
			Path p = new Path(matrices[i]).makeQualified(fs);
			if(thisFile.toUri().compareTo(p.toUri())==0 || thisDir.toUri().compareTo(p.toUri())==0)
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
	
	// TODO: check Rand
	public static DataGenMRInstruction[] getDataGenInstructions(JobConf job) throws DMLUnsupportedOperationException, DMLRuntimeException {
		String str=job.get(RAND_INSTRUCTIONS_CONFIG);
		return MRInstructionParser.parseDataGenInstructions(str);
	}
	
	public static AggregateBinaryInstruction[] getAggregateBinaryInstructions(JobConf job) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		String str=job.get(AGGREGATE_BINARY_INSTRUCTIONS_CONFIG);
		return MRInstructionParser.parseAggregateBinaryInstructions(str);
	}
	
	public static CM_N_COVInstruction[] getCM_N_COVInstructions(JobConf job) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		String str=job.get(CM_N_COV_INSTRUCTIONS_CONFIG);
		return MRInstructionParser.parseCM_N_COVInstructions(str);
	}
	
	/**
	 * 
	 * @param job
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public static GroupedAggregateInstruction[] getGroupedAggregateInstructions(JobConf job) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//parse all grouped aggregate instructions
		String str=job.get(GROUPEDAGG_INSTRUCTIONS_CONFIG);
		GroupedAggregateInstruction[] tmp = MRInstructionParser.parseGroupedAggInstructions(str);
		
		//obtain bclen for all instructions
		for( int i=0; i< tmp.length; i++ ) {
			byte tag = tmp[i].input;
			tmp[i].setBclen(getMatrixCharacteristicsForInput(job, tag).getColsPerBlock());
		}
		
		return tmp;
	}
	
	public static String[] getOutputs(JobConf job)
	{
		return job.getStrings(OUTPUT_MATRICES_DIRS_CONFIG);
	}
	
	private static byte[] stringArrayToByteArray(String[] istrs)
	{
		byte[] ret=new byte[istrs.length];
		for(int i=0; i<istrs.length; i++)
			ret[i]=Byte.parseByte(istrs[i]);
		return ret;
	}
	
	public static byte[] getResultIndexes(JobConf job)
	{
		String[] istrs=job.get(RESULT_INDEXES_CONFIG).split(Instruction.INSTRUCTION_DELIM);
		return stringArrayToByteArray(istrs);
	}
	public static byte[] getResultDimsUnknown(JobConf job)
	{
		String str=job.get(RESULT_DIMS_UNKNOWN_CONFIG);
		if (str==null || str.isEmpty())
			return null;
		String[] istrs=str.split(Instruction.INSTRUCTION_DELIM);
		return stringArrayToByteArray(istrs);
	}
		
	public static byte[] getIntermediateMatrixIndexes(JobConf job)
	{
		String str=job.get(INTERMEDIATE_INDEXES_CONFIG);
		if(str==null || str.isEmpty())
			return null;
		String[] istrs=str.split(Instruction.INSTRUCTION_DELIM);
		return stringArrayToByteArray(istrs);
	}
	
	public static void setIntermediateMatrixIndexes(JobConf job, HashSet<Byte> indexes)
	{
		job.set(INTERMEDIATE_INDEXES_CONFIG, getIndexesString(indexes));
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
	
	public static void handleRecordReaderInstrucion(JobConf job, String recordReaderInstruction, String[] inputs, InputInfo[] inputInfos)
	{
		//do nothing, not used currently
	}
	
	public static void setupDistCacheInputs(JobConf job, String indices, String pathsString, ArrayList<String> paths) {
		job.set(DISTCACHE_INPUT_INDICES, indices);
		job.set(DISTCACHE_INPUT_PATHS, pathsString);
		Path p = null;
		
		for(String spath : paths) {
			p = new Path(spath);
			
			DistributedCache.addCacheFile(p.toUri(), job);
			DistributedCache.createSymlink(job);
		}
	}
	
	public static String getDistCacheInputIndices(JobConf job) {
		return job.get(DISTCACHE_INPUT_INDICES);
	}
	
	public static String getDistCacheInputPaths(JobConf job) {
		return job.get(DISTCACHE_INPUT_PATHS);
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
	
	public static void setUpMultipleInputs(JobConf job, byte[] inputIndexes, String[] inputs, InputInfo[] inputInfos, 
			int[] brlens, int[] bclens, boolean setConverter, ConvertTarget target) 
	throws Exception
	{
		//conservative initialize (all jobs except GMR)
		boolean[] distCacheOnly = new boolean[inputIndexes.length];
		Arrays.fill(distCacheOnly, false);
		
		setUpMultipleInputs(job, inputIndexes, inputs, inputInfos, brlens, bclens, distCacheOnly, setConverter, target);
	}
	
	/**
	 * 
	 * @param job
	 * @param inputIndexes
	 * @param inputs
	 * @param inputInfos
	 * @param brlens
	 * @param bclens
	 * @param distCacheOnly
	 * @param setConverter
	 * @param target
	 * @throws Exception
	 */
	public static void setUpMultipleInputs(JobConf job, byte[] inputIndexes, String[] inputs, InputInfo[] inputInfos, 
			int[] brlens, int[] bclens, boolean[] distCacheOnly, boolean setConverter, ConvertTarget target) 
		throws Exception
	{
		if(inputs.length!=inputInfos.length)
			throw new Exception("number of inputs and inputInfos does not match");
		
		//set up names of the input matrices and their inputformat information
		job.setStrings(INPUT_MATRICIES_DIRS_CONFIG, inputs);
		MRJobConfiguration.setMapFunctionInputMatrixIndexes(job, inputIndexes);
		
		//set up converter infos (converter determined implicitly)
		if(setConverter) {
			for(int i=0; i<inputs.length; i++)
				setInputInfo(job, inputIndexes[i], inputInfos[i], brlens[i], bclens[i], target);
		}
		
		//remove redundant inputs and pure broadcast variables
		ArrayList<Path> lpaths = new ArrayList<Path>();
		ArrayList<InputInfo> liinfos = new ArrayList<InputInfo>();
		for(int i=0; i<inputs.length; i++)
		{
			Path p = new Path(inputs[i]);
			
			//check and skip redundant inputs
			if(   lpaths.contains(p) //path already included
			   || distCacheOnly[i] ) //input only required in dist cache
			{
				continue;
			}
			
			lpaths.add(p);
			liinfos.add(inputInfos[i]);
		}
		
		boolean combineInputFormat = false;
		if( OptimizerUtils.ALLOW_COMBINE_FILE_INPUT_FORMAT ) 
		{
			//determine total input sizes
			double totalInputSize = 0;
			for(int i=0; i<inputs.length; i++)
				totalInputSize += MapReduceTool.getFilesizeOnHDFS(new Path(inputs[i]));
				
			//set max split size (default blocksize) to 2x blocksize if (1) sort buffer large enough, 
			//(2) degree of parallelism not hurt, and only a single input (except broadcasts)
			//(the sort buffer size is relevant for pass-through of, potentially modified, inputs to the reducers)
			//(the single input constraint stems from internal runtime assumptions used to relate meta data to inputs)
			long sizeSortBuff = InfrastructureAnalyzer.getRemoteMaxMemorySortBuffer();
			long sizeHDFSBlk = InfrastructureAnalyzer.getHDFSBlockSize();
			long newSplitSize = sizeHDFSBlk * 2;
			double spillPercent = job.getDouble("mapreduce.map.sort.spill.percent", 1.0);
			int numPMap = OptimizerUtils.getNumMappers();
			if( numPMap < totalInputSize/newSplitSize && sizeSortBuff*spillPercent >= newSplitSize && lpaths.size()==1 ) {
				job.setLong(MRConfigurationNames.MR_INPUT_FILEINPUTFORMAT_SPLIT_MAXSIZE, newSplitSize);
				combineInputFormat = true;
			}
		}
		
		//add inputs to jobs input (incl input format configuration)
		for(int i=0; i<lpaths.size(); i++)
		{
			//add input to job inputs (for binaryblock we use CombineSequenceFileInputFormat to reduce task latency)
			if( combineInputFormat && liinfos.get(i) == InputInfo.BinaryBlockInputInfo )
				MultipleInputs.addInputPath(job, lpaths.get(i), CombineSequenceFileInputFormat.class);
			else
				MultipleInputs.addInputPath(job, lpaths.get(i), liinfos.get(i).inputFormatClass);
		}
	}
	
	/**
	 * Specific method because we need to set the input converter class according to the 
	 * input infos. Note that any mapper instruction before reblock can work on binary block
	 * if it can work on binary cell as well.
	 * 
	 * @param job
	 * @param inputIndexes
	 * @param inputs
	 * @param inputInfos
	 * @param inBlockRepresentation
	 * @param brlens
	 * @param bclens
	 * @param setConverter
	 * @param forCMJob
	 * @throws Exception
	 */
	public static void setUpMultipleInputsReblock(JobConf job, byte[] inputIndexes, String[] inputs, InputInfo[] inputInfos, 
												  int[] brlens, int[] bclens) 
		throws Exception
	{
		if(inputs.length!=inputInfos.length)
			throw new Exception("number of inputs and inputInfos does not match");
		
		//set up names of the input matrices and their inputformat information
		job.setStrings(INPUT_MATRICIES_DIRS_CONFIG, inputs);
		MRJobConfiguration.setMapFunctionInputMatrixIndexes(job, inputIndexes);
		
		for(int i=0; i<inputs.length; i++)
		{
			ConvertTarget target=ConvertTarget.CELL;
			if(inputInfos[i]==InputInfo.BinaryBlockInputInfo)
				target=ConvertTarget.BLOCK;
			setInputInfo(job, inputIndexes[i], inputInfos[i], brlens[i], bclens[i], target);
		}
		
		//remove redundant input files
		ArrayList<Path> paths=new ArrayList<Path>();
		for(int i=0; i<inputs.length; i++)
		{
			String name=inputs[i];
			Path p=new Path(name);
			boolean redundant=false;
			for(Path ep: paths)
				if(ep.equals(p))
				{
					redundant=true;
					break;
				}
			if(redundant)
				continue;
			MultipleInputs.addInputPath(job, p, inputInfos[i].inputFormatClass);
			paths.add(p);
		}
	}
	
	
	public static void updateResultDimsUnknown (JobConf job, byte[] updDimsUnknown) {
		job.set(RESULT_DIMS_UNKNOWN_CONFIG, MRJobConfiguration.getIndexesString(updDimsUnknown));
	}
	
	public static void setUpMultipleOutputs(JobConf job, byte[] resultIndexes, byte[] resultDimsUnknown, String[] outputs, 
			OutputInfo[] outputInfos, boolean inBlockRepresentation, boolean mayContainCtable) 
	throws Exception
	{
		if(resultIndexes.length!=outputs.length)
			throw new Exception("number of outputs and result indexes does not match");
		if(outputs.length!=outputInfos.length)
			throw new Exception("number of outputs and outputInfos indexes does not match");
		
		job.set(RESULT_INDEXES_CONFIG, MRJobConfiguration.getIndexesString(resultIndexes));
		job.set(RESULT_DIMS_UNKNOWN_CONFIG, MRJobConfiguration.getIndexesString(resultDimsUnknown));
		job.setStrings(OUTPUT_MATRICES_DIRS_CONFIG, outputs);
		job.setOutputCommitter(MultipleOutputCommitter.class);
		
		for(int i=0; i<outputs.length; i++)
		{
			MapReduceTool.deleteFileIfExistOnHDFS(new Path(outputs[i]), job);
			if ( mayContainCtable && resultDimsUnknown[i] == (byte) 1 )  {
				setOutputInfo(job, i, outputInfos[i], false);
			}
			else {
				setOutputInfo(job, i, outputInfos[i], inBlockRepresentation);
			}
			MultipleOutputs.addNamedOutput(job, Integer.toString(i), 
					outputInfos[i].outputFormatClass, outputInfos[i].outputKeyClass, 
					outputInfos[i].outputValueClass);
		}
		job.setOutputFormat(NullOutputFormat.class);
		
		// configure temp output
		Path tempOutputPath = new Path( constructTempOutputFilename() );
		FileOutputFormat.setOutputPath(job, tempOutputPath);
		MapReduceTool.deleteFileIfExistOnHDFS(tempOutputPath, job);
	}
	
	public static void setUpMultipleOutputs(JobConf job, byte[] resultIndexes, byte[] resultDimsUnknwon, String[] outputs, 
			OutputInfo[] outputInfos, boolean inBlockRepresentation) 
	throws Exception
	{
		setUpMultipleOutputs(job, resultIndexes, resultDimsUnknwon, outputs, 
				outputInfos, inBlockRepresentation, false);
	}
	
	/**
	 * 
	 * @param job
	 * @return
	 */
	public static String setUpSortPartitionFilename( JobConf job ) 
	{
		String pfname = constructPartitionFilename();
		job.set( SORT_PARTITION_FILENAME, pfname );
		
		return pfname;
	}
	
	/**
	 * 
	 * @param job
	 * @return
	 */
	public static String getSortPartitionFilename( JobConf job )
	{
		return job.get( SORT_PARTITION_FILENAME );
	}
	
	public static MatrixChar_N_ReducerGroups computeMatrixCharacteristics(JobConf job, byte[] inputIndexes, 
			String instructionsInMapper, String aggInstructionsInReducer, String aggBinInstructions, 
			String otherInstructionsInReducer, byte[] resultIndexes, HashSet<Byte> mapOutputIndexes, boolean forMMCJ) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		return computeMatrixCharacteristics(job, inputIndexes, null, instructionsInMapper, null, aggInstructionsInReducer, 
				aggBinInstructions, otherInstructionsInReducer, resultIndexes, mapOutputIndexes, forMMCJ);
	}
	
	public static MatrixChar_N_ReducerGroups computeMatrixCharacteristics(JobConf job, byte[] inputIndexes, 
			String instructionsInMapper, String reblockInstructions, String aggInstructionsInReducer, String aggBinInstructions, 
			String otherInstructionsInReducer, byte[] resultIndexes, HashSet<Byte> mapOutputIndexes, boolean forMMCJ) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		return computeMatrixCharacteristics(job, inputIndexes, null, instructionsInMapper, reblockInstructions, aggInstructionsInReducer, 
				aggBinInstructions, otherInstructionsInReducer, resultIndexes, mapOutputIndexes, forMMCJ);
	}
	
	public static void setNumReducers(JobConf job, long numReducerGroups, int numFromCompiler) throws IOException
	{
		JobClient client=new JobClient(job);
		int n=client.getClusterStatus().getMaxReduceTasks();
		//correction max number of reducers on yarn clusters
		if( InfrastructureAnalyzer.isYarnEnabled() )
			n = (int)Math.max( n, YarnClusterAnalyzer.getNumCores()/2 );
		n=Math.min(n, ConfigurationManager.getConfig().getIntValue(DMLConfig.NUM_REDUCERS));
		n=Math.min(n, numFromCompiler);
		if(numReducerGroups>0)
			n=(int) Math.min(n, numReducerGroups);
		job.setNumReduceTasks(n); 
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
	
	/**
	 * NOTE: this method needs to be in-sync with MRBaseForCommonInstructions.processOneInstruction,
	 * otherwise, the latter will potentially fail with missing dimension information.
	 * 
	 * @param job
	 * @param inputIndexes
	 * @param dataGenInstructions
	 * @param instructionsInMapper
	 * @param reblockInstructions
	 * @param aggInstructionsInReducer
	 * @param aggBinInstructions
	 * @param otherInstructionsInReducer
	 * @param resultIndexes
	 * @param mapOutputIndexes
	 * @param forMMCJ
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public static MatrixChar_N_ReducerGroups computeMatrixCharacteristics(JobConf job, byte[] inputIndexes, String dataGenInstructions,
			String instructionsInMapper, String reblockInstructions, String aggInstructionsInReducer, String aggBinInstructions, 
			String otherInstructionsInReducer, byte[] resultIndexes, HashSet<Byte> mapOutputIndexes, boolean forMMCJ) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		HashSet<Byte> intermediateMatrixIndexes=new HashSet<Byte>();
		HashMap<Byte, MatrixCharacteristics> dims=new HashMap<Byte, MatrixCharacteristics>();
		for(byte i: inputIndexes){
			MatrixCharacteristics dim=new MatrixCharacteristics(getNumRows(job, i), getNumColumns(job, i), 
					getNumRowsPerBlock(job, i), getNumColumnsPerBlock(job, i), getNumNonZero(job, i));
			dims.put(i, dim);
		}
		DataGenMRInstruction[] dataGenIns = null;
		dataGenIns = MRInstructionParser.parseDataGenInstructions(dataGenInstructions);
		if(dataGenIns!=null)
		{
			for(DataGenMRInstruction ins: dataGenIns)
			{
				MatrixCharacteristics.computeDimension(dims, ins);
			}
		}
		
		MRInstruction[] insMapper = MRInstructionParser.parseMixedInstructions(instructionsInMapper);
		if(insMapper!=null)
		{
			for(MRInstruction ins: insMapper)
			{
				MatrixCharacteristics.computeDimension(dims, ins);
				
				if( ins instanceof UnaryMRInstructionBase )
				{
					UnaryMRInstructionBase tempIns=(UnaryMRInstructionBase) ins;
					setIntermediateMatrixCharactristics(job, tempIns.input, 
							dims.get(tempIns.input));
					intermediateMatrixIndexes.add(tempIns.input);
				}
				else if(ins instanceof AppendMInstruction)
				{
					AppendMInstruction tempIns=(AppendMInstruction) ins;
					setIntermediateMatrixCharactristics(job, tempIns.input1, 
							dims.get(tempIns.input1));
					intermediateMatrixIndexes.add(tempIns.input1);
				}
				else if(ins instanceof AppendGInstruction)
				{
					AppendGInstruction tempIns=(AppendGInstruction) ins;
					setIntermediateMatrixCharactristics(job, tempIns.input1, 
							dims.get(tempIns.input1));
					intermediateMatrixIndexes.add(tempIns.input1);
				}
				else if(ins instanceof BinaryMInstruction)
				{
					BinaryMInstruction tempIns=(BinaryMInstruction) ins;
					setIntermediateMatrixCharactristics(job, tempIns.input1, 
							dims.get(tempIns.input1));
					intermediateMatrixIndexes.add(tempIns.input1);
				}
				else if(ins instanceof AggregateBinaryInstruction)
				{
					AggregateBinaryInstruction tempIns=(AggregateBinaryInstruction) ins;
					setIntermediateMatrixCharactristics(job, tempIns.input1, dims.get(tempIns.input1));
					intermediateMatrixIndexes.add(tempIns.input1); //TODO
				}
				else if(ins instanceof MapMultChainInstruction)
				{
					MapMultChainInstruction tempIns=(MapMultChainInstruction) ins;
					setIntermediateMatrixCharactristics(job, tempIns.getInput1(), dims.get(tempIns.getInput2()));	
					intermediateMatrixIndexes.add(tempIns.getInput1());
				}
				else if(ins instanceof PMMJMRInstruction)
				{
					PMMJMRInstruction tempIns=(PMMJMRInstruction) ins;
					setIntermediateMatrixCharactristics(job, tempIns.input2, dims.get(tempIns.input2));	
					intermediateMatrixIndexes.add(tempIns.input2);
				}
			}
		}
		
		ReblockInstruction[] reblockIns = MRInstructionParser.parseReblockInstructions(reblockInstructions);
		if(reblockIns!=null)
		{
			for(ReblockInstruction ins: reblockIns)
			{
				MatrixCharacteristics.computeDimension(dims, ins);
				setMatrixCharactristicsForReblock(job, ins.output, dims.get(ins.output));
			}
		}
		
		Instruction[] aggIns = MRInstructionParser.parseAggregateInstructions(aggInstructionsInReducer);
		if(aggIns!=null)
		{
			for(Instruction ins: aggIns) {
				MatrixCharacteristics.computeDimension(dims, (MRInstruction) ins);
			
				// if instruction's output is not in resultIndexes, then add its dimensions to jobconf
				MRInstruction mrins = (MRInstruction)ins;
				boolean found = false;
				for(byte b : resultIndexes) {
					if(b==mrins.output) {
						found = true;
						break;
					}
				}
				if(!found) {
					setIntermediateMatrixCharactristics(job, mrins.output, dims.get(mrins.output));
					intermediateMatrixIndexes.add(mrins.output);	
				}
			}
		}
		
		long numReduceGroups=0;
		AggregateBinaryInstruction[] aggBinIns = getAggregateBinaryInstructions(job);
		if(aggBinIns!=null)
		{
			for(AggregateBinaryInstruction ins: aggBinIns)
			{
				MatrixCharacteristics dim1=dims.get(ins.input1);
				MatrixCharacteristics dim2=dims.get(ins.input2);
				setMatrixCharactristicsForBinAgg(job, ins.input1, dim1);
				setMatrixCharactristicsForBinAgg(job, ins.input2, dim2);
				MatrixCharacteristics.computeDimension(dims, ins);
				if(forMMCJ)//there will be only one aggbin operation for MMCJ
					numReduceGroups=(long) Math.ceil((double)dim1.getCols()/(double)dim1.getColsPerBlock());
			}
		}
		if(!forMMCJ)
		{
			//store the skylines
			ArrayList<Long> xs=new ArrayList<Long>(mapOutputIndexes.size());
			ArrayList<Long> ys=new ArrayList<Long>(mapOutputIndexes.size());
			for(byte idx: mapOutputIndexes)
			{
				MatrixCharacteristics dim=dims.get(idx);
				long x=(long)Math.ceil((double)dim.getRows()/(double)dim.getRowsPerBlock());
				long y=(long)Math.ceil((double)dim.getCols()/(double)dim.getColsPerBlock());
				
				int i=0; 
				boolean toadd=true;
				while(i<xs.size())
				{
					if( (x>=xs.get(i)&&y>ys.get(i)) || (x>xs.get(i)&&y>=ys.get(i)))
					{
						//remove any included x's and y's
						xs.remove(i);
						ys.remove(i);
					}else if(x<=xs.get(i) && y<=ys.get(i))//if included in others, stop
					{
						toadd=false;
						break;
					}
					else
						i++;
				}
				
				if(toadd)
				{
					xs.add(x);
					ys.add(y);
				}
			}
			//sort by x
			TreeMap<Long, Long> map=new TreeMap<Long, Long>();
			for(int i=0; i<xs.size(); i++)
				map.put(xs.get(i), ys.get(i));
			numReduceGroups=0;
			//compute area
			long prev=0;
			for(Entry<Long, Long> e: map.entrySet())
			{
				numReduceGroups+=(e.getKey()-prev)*e.getValue();
				prev=e.getKey();
			}
		}
		
		
		MRInstruction[] insReducer = MRInstructionParser.parseMixedInstructions(otherInstructionsInReducer);
		if(insReducer!=null)
		{
			for(MRInstruction ins: insReducer)
			{
				MatrixCharacteristics.computeDimension(dims, ins);
				if( ins instanceof UnaryMRInstructionBase )
				{
					UnaryMRInstructionBase tempIns=(UnaryMRInstructionBase) ins;
					setIntermediateMatrixCharactristics(job, tempIns.input, 
							dims.get(tempIns.input));
					intermediateMatrixIndexes.add(tempIns.input);
				}
				else if( ins instanceof RemoveEmptyMRInstruction )
				{
					RemoveEmptyMRInstruction tempIns = (RemoveEmptyMRInstruction) ins;
					setIntermediateMatrixCharactristics(job, tempIns.input1, 
							dims.get(tempIns.input1));
					intermediateMatrixIndexes.add(tempIns.input1);	
				}
				
				// if instruction's output is not in resultIndexes, then add its dimensions to jobconf
				boolean found = false;
				for(byte b : resultIndexes) {
					if(b==ins.output) {
						found = true;
						break;
					}
				}
				if(!found) {
					setIntermediateMatrixCharactristics(job, ins.output, 
							dims.get(ins.output));
					intermediateMatrixIndexes.add(ins.output);	
				}
			}
		}
		
		setIntermediateMatrixIndexes(job, intermediateMatrixIndexes);
		
		for (byte tag : mapOutputIndexes)
			setMatrixCharactristicsForMapperOutput(job, tag, dims.get(tag));
		
		MatrixCharacteristics[] stats=new MatrixCharacteristics[resultIndexes.length];
		MatrixCharacteristics resultDims;
		for(int i=0; i<resultIndexes.length; i++)
		{
			resultDims = dims.get(resultIndexes[i]);
			stats[i]=resultDims;
			setMatrixCharactristicsForOutput(job, resultIndexes[i], stats[i]);
		}
		
		return new MatrixChar_N_ReducerGroups(stats, numReduceGroups);
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
	
	public static HashSet<Byte> setUpOutputIndexesForMapper(JobConf job, byte[] inputIndexes, String instructionsInMapper, 
			String aggInstructionsInReducer, String otherInstructionsInReducer, byte[] resultIndexes) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		return setUpOutputIndexesForMapper(job, inputIndexes, null, instructionsInMapper, 
				null, aggInstructionsInReducer, otherInstructionsInReducer, resultIndexes);
	}
	
	public static HashSet<Byte> setUpOutputIndexesForMapper(JobConf job, byte[] inputIndexes, String instructionsInMapper, 
			String reblockInstructions, String aggInstructionsInReducer, String otherInstructionsInReducer, byte[] resultIndexes) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		return setUpOutputIndexesForMapper(job, inputIndexes, null, instructionsInMapper, 
				reblockInstructions, aggInstructionsInReducer, otherInstructionsInReducer, resultIndexes);
	}
	public static HashSet<Byte> setUpOutputIndexesForMapper(JobConf job, byte[] inputIndexes, String randInstructions, String instructionsInMapper, 
			String reblockInstructions, String aggInstructionsInReducer, String otherInstructionsInReducer, byte[] resultIndexes) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		//find out what results are needed to send to reducers
		
		HashSet<Byte> indexesInMapper=new HashSet<Byte>();
		for(byte b: inputIndexes)
			indexesInMapper.add(b);
		
		DataGenMRInstruction[] dataGenIns = null;
		dataGenIns = MRInstructionParser.parseDataGenInstructions(randInstructions);
		getIndexes(dataGenIns, indexesInMapper);
		
		MRInstruction[] insMapper = MRInstructionParser.parseMixedInstructions(instructionsInMapper);
		getIndexes(insMapper, indexesInMapper);
		
		ReblockInstruction[] reblockIns = null;
		reblockIns = MRInstructionParser.parseReblockInstructions(reblockInstructions);
		getIndexes(reblockIns, indexesInMapper);
		
		MRInstruction[] insReducer = MRInstructionParser.parseAggregateInstructions(aggInstructionsInReducer);
		HashSet<Byte> indexesInReducer=new HashSet<Byte>();
		getIndexes(insReducer, indexesInReducer);
		
		insReducer = MRInstructionParser.parseMixedInstructions(otherInstructionsInReducer);
		getIndexes(insReducer, indexesInReducer);
	
		for(byte ind: resultIndexes)
			indexesInReducer.add(ind);
		
		indexesInMapper.retainAll(indexesInReducer);
		
		job.set(OUTPUT_INDEXES_IN_MAPPER_CONFIG, getIndexesString(indexesInMapper));
		return indexesInMapper;
	}
	
	public static CollectMultipleConvertedOutputs getMultipleConvertedOutputs(JobConf job)
	{		
		byte[] resultIndexes=MRJobConfiguration.getResultIndexes(job);
		Converter[] outputConverters=new Converter[resultIndexes.length];
		MatrixCharacteristics[] stats=new MatrixCharacteristics[resultIndexes.length];
		HashMap<Byte, ArrayList<Integer>> tagMapping=new HashMap<Byte, ArrayList<Integer>>();
		for(int i=0; i<resultIndexes.length; i++)
		{
			byte output=resultIndexes[i];
			ArrayList<Integer> vec=tagMapping.get(output);
			if(vec==null)
			{
				vec=new ArrayList<Integer>();
				tagMapping.put(output, vec);
			}
			vec.add(i);
			
			outputConverters[i]=getOuputConverter(job, i);
			stats[i]=MRJobConfiguration.getMatrixCharacteristicsForOutput(job, output);
		}
		
		MultipleOutputs multipleOutputs=new MultipleOutputs(job);
		
		return new CollectMultipleConvertedOutputs(outputConverters, stats, multipleOutputs);
		
	}
	
	private static void getIndexes(MRInstruction[] instructions, HashSet<Byte> indexes) 
		throws DMLRuntimeException
	{
		if(instructions==null)
			return;
		for(MRInstruction ins: instructions)
		{
			for(byte i: ins.getAllIndexes())
				indexes.add(i);
		}
	}
	
	private static String getIndexesString(HashSet<Byte> indexes)
	{
		if(indexes==null || indexes.isEmpty())
			return "";
		
		StringBuilder sb = new StringBuilder();
		for(Byte ind: indexes) {
			sb.append(ind);
			sb.append(Instruction.INSTRUCTION_DELIM);
		}
		
		//return string without last character
		return sb.substring(0, sb.length()-1);
	}
	
	private static String getIndexesString(byte[] indexes)
	{
		if(indexes==null || indexes.length==0)
			return "";
		
		StringBuilder sb = new StringBuilder();
		for(Byte ind: indexes) {
			sb.append(ind);
			sb.append(Instruction.INSTRUCTION_DELIM);
		}
		
		//return string without last character
		return sb.substring(0, sb.length()-1);
	}

	public static void setMapFunctionInputMatrixIndexes(JobConf job, byte[] realIndexes) 
	{
		job.set(MAPFUNC_INPUT_MATRICIES_INDEXES_CONFIG, getIndexesString(realIndexes));	
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
		sb.append(ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE));
		sb.append(Lop.FILE_SEPARATOR);
		sb.append(Lop.PROCESS_PREFIX);
		sb.append(DMLScript.getUUID());
		sb.append(Lop.FILE_SEPARATOR);
		
		sb.append("TmpOutput"+seq.getNextID());
		
		//old unique dir (no guarantees): 
		//sb.append(Integer.toHexString(new Random().nextInt(Integer.MAX_VALUE))); 
		
		return sb.toString(); 
	}
	
	private static String constructPartitionFilename() 
	{
		StringBuilder sb = new StringBuilder();
		sb.append(ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE));
		sb.append(Lop.FILE_SEPARATOR);
		sb.append(Lop.PROCESS_PREFIX);
		sb.append(DMLScript.getUUID());
		sb.append(Lop.FILE_SEPARATOR);
		
		sb.append(SamplingSortMRInputFormat.PARTITION_FILENAME+seq.getNextID());
		
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
}