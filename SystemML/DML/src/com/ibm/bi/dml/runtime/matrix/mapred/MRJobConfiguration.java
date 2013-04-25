package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;
import java.util.Vector;
import java.util.Map.Entry;

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
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.hadoop.mapred.lib.NullOutputFormat;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.runtime.RunMRJobs.ExecMode;
import com.ibm.bi.dml.meta.PartitionParams;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.AddDummyWeightConverter;
import com.ibm.bi.dml.runtime.matrix.io.BinaryBlockToBinaryCellConverter;
import com.ibm.bi.dml.runtime.matrix.io.BinaryBlockToTextCellConverter;
import com.ibm.bi.dml.runtime.matrix.io.BinaryCellToTextConverter;
import com.ibm.bi.dml.runtime.matrix.io.Converter;
import com.ibm.bi.dml.runtime.matrix.io.IdenticalConverter;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock1D;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM;
import com.ibm.bi.dml.runtime.matrix.io.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.MultipleOutputCommitter;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.TextToBinaryCellConverter;
import com.ibm.bi.dml.runtime.matrix.io.WeightedCellToSortInputConverter;
import com.ibm.bi.dml.runtime.matrix.io.hadoopfix.MultipleInputs;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.*;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AggregateUnaryInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.AppendInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.CM_N_COVInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.GroupedAggregateInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.MRInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RandInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RangeBasedReIndexInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ReblockInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.ZeroOutInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.UnaryMRInstructionBase;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;
import com.ibm.bi.dml.runtime.matrix.io.WeightedPair;
import com.ibm.bi.dml.runtime.matrix.sort.SamplingSortMRInputFormat;

public class MRJobConfiguration {
	
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
	
	//operations performed in the mapper
	private static final String INSTRUCTIONS_IN_MAPPER_CONFIG="instructions.in.mapper";
	private static final String RAND_INSTRUCTIONS_CONFIG="rand.instructions";
	//matrix indexes to be outputted to reducer
	private static final String OUTPUT_INDEXES_IN_MAPPER_CONFIG="output.indexes.in.mapper";
	
	//parfor serialized program
	private static final String PARFOR_PROGRAMBLOCKS_IN_MAPPER_CONFIG="programblocks.in.mapper";
	
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
	
	//result merge info
	//private static final String RESULTMERGE_INPUT_BLOCK_NUM_ROW_CONFIG="partitioning.input.block.num.row";
	//private static final String RESULTMERGE_INPUT_BLOCK_NUM_COLUMN_CONFIG="partitioning.input.block.num.column";
	private static final String RESULTMERGE_INPUT_INFO_CONFIG="resultmerge.input.inputinfo";
	private static final String RESULTMERGE_COMPARE_FILENAME_CONFIG="resultmerge.compare.filename";
	private static final String RESULTMERGE_STAGING_DIR_CONFIG="resultmerge.staging.dir";
	
	private static final String SORT_PARTITION_FILENAME = "sort.partition.filename";
	
	//operations performed in the reduer
	private static final String AGGREGATE_INSTRUCTIONS_CONFIG="aggregate.instructions.after.groupby.at";
	private static final String INSTRUCTIONS_IN_REDUCER_CONFIG="instructions.in.reducer";
	private static final String AGGREGATE_BINARY_INSTRUCTIONS_CONFIG="aggregate.binary.instructions";
	private static final String REBLOCK_INSTRUCTIONS_CONFIG="reblock.instructions";
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
	
	//characteristics about the matrices to matrixdiag instructions
	private static final String INTERMEDIATE_MATRIX_NUM_ROW_PREFIX_CONFIG="diagm2v.matrix.num.row.";
	private static final String INTERMEDIATE_MATRIX_NUM_COLUMN_PREFIX_CONFIG="diagm2v.matrix.num.column.";
	private static final String INTERMEDIATE_BLOCK_NUM_ROW_PREFIX_CONFIG="diagm2v.block.num.row.";
	private static final String INTERMEDIATE_BLOCK_NUM_COLUMN_PREFIX_CONFIG="diagm2v.block.num.column.";
	
	//matrix indexes to be outputted as final results
	private static final String RESULT_INDEXES_CONFIG="results.indexes";
	private static final String RESULT_DIMS_UNKNOWN_CONFIG="results.dims.unknown";
	
	private static final String INTERMEDIATE_INDEXES_CONFIG="diagm2v.indexes";
	
	//output matrices and their formats
	private static final String OUTPUT_MATRICES_DIRS_CONFIG="output.matrices.dirs";
	private static final String OUTPUT_CONVERTER_CLASS_PREFIX_CONFIG="output.converter.class.for.";
	
	private static final String DIMS_UNKNOWN_FILE_PREFIX = "dims.unknown.file.prefix";
	
	private static final String PARTIAL_AGG_CACHE_SIZE="partial.aggregate.cache.size";
	
	private static final String PARTITION_FLAGS   = "partition.flags";
	private static final String PARTITION_FORMATS = "partition.formats";
	private static final String PARTITION_SIZES   = "partition.sizes";
	
	private static final String DISTCACHE_INPUT_INDICES="distcache.input.indices";
	private static final String DISTCACHE_INPUT_PATHS = "distcache.input.paths";
	
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
	
	public static final int getMiscMemRequired(JobConf job)
	{
		int ret=job.getInt("io.file.buffer.size", 4096);
	//	ret+=job.getInt("io.sort.mb", 0)*1048576;
	//	ret+=job.getInt("fs.inmemory.size.mb", 0)*1048576;
		return ret;
	}
	
	public static final int getJVMMaxMemSize(JobConf job)
	{
		String str=job.get("mapred.child.java.opts");
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
	
	public static void setPartialAggCacheSize(JobConf job, int size)
	{
		job.setInt(PARTIAL_AGG_CACHE_SIZE, size);
	}
	
	public static int getPartialAggCacheSize(JobConf job)
	{
		return job.getInt(PARTIAL_AGG_CACHE_SIZE, 0);
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
	
	public static Class<? extends Converter> getConverterClass(InputInfo inputinfo, boolean targetToBlock, 
			int brlen, int bclen)
	{
		return getConverterClass(inputinfo, targetToBlock, brlen, bclen, false);
	}
	
	public static Class<? extends Converter> getConverterClass(InputInfo inputinfo, boolean targetToBlock, 
			int brlen, int bclen, boolean targetToWeightedCell)
	{
		assert(!targetToBlock || !targetToWeightedCell);
		Class<? extends Converter> converterClass;
		if(inputinfo.inputValueClass.equals(MatrixCell.class))
		{
			if(targetToBlock)
				throw new RuntimeException("cannot convert binary cell to binary block representation implicitly");
			else if(targetToWeightedCell)
				converterClass=AddDummyWeightConverter.class;
			else
				converterClass=IdenticalConverter.class;
		}else if(inputinfo.inputValueClass.equals(MatrixBlock.class))
		{
			if(targetToBlock)
				converterClass=IdenticalConverter.class;
			else if(targetToWeightedCell)
				converterClass=AddDummyWeightConverter.class;
			else
				converterClass=BinaryBlockToBinaryCellConverter.class;
		}else if(inputinfo.inputValueClass.equals(Text.class))
		{
			if(targetToBlock && (brlen>1 || bclen>1))
				throw new RuntimeException("cannot convert text cell to binary block representation implicitly");
			else if(targetToWeightedCell)
				converterClass=AddDummyWeightConverter.class;
			else
				converterClass=TextToBinaryCellConverter.class;
		}else
			converterClass=IdenticalConverter.class;
		
		return converterClass;
	}
	
	/**
	 * Unique working dirs required for thread-safe submission of parallel jobs;
	 * otherwise job.xml and other files might be overridden (in local mode).
	 * 
	 * @param job
	 * @param mode
	 */
	public static void setUniqueWorkingDir( JobConf job, ExecMode mode )
	{
		if( isLocalJobTracker(job) )
		{
			StringBuilder tmp = new StringBuilder();
			tmp.append( Lops.FILE_SEPARATOR );
			tmp.append( Lops.PROCESS_PREFIX );
			tmp.append( DMLScript.getUUID() );
			tmp.append( Lops.FILE_SEPARATOR );
			tmp.append( seq.getNextID() );
			String uniqueSubdir = tmp.toString();
			
			//unique local dir
			String[] dirlist = job.get("mapred.local.dir","/tmp").split(",");
			StringBuilder sb2 = new StringBuilder();
			for( String dir : dirlist ) {
				if( sb2.length()>0 )
					sb2.append(",");
				sb2.append(dir);
				sb2.append( uniqueSubdir );
			}
			job.set("mapred.local.dir", sb2.toString() );			
			
			//unique system dir 
			job.set("mapred.system.dir", job.get("mapred.system.dir") + uniqueSubdir);
			
			//unique staging dir 
			job.set( "mapreduce.jobtracker.staging.root.dir",  job.get("mapreduce.jobtracker.staging.root.dir") + uniqueSubdir );
		}
	}
	
	public static String getLocalWorkingDirPrefix()
	{
		JobConf job = new JobConf();
		return job.get("mapred.local.dir");
	}
	
	public static String getSystemWorkingDirPrefix()
	{
		JobConf job = new JobConf();
		return job.get("mapred.system.dir");
	}
	
	public static String getStagingWorkingDirPrefix()
	{
		JobConf job = new JobConf();
		return job.get("mapreduce.jobtracker.staging.root.dir");
	}
	
	public static boolean isLocalJobTracker()
	{
		return isLocalJobTracker(new JobConf());
	}
	
	public static boolean isLocalJobTracker(JobConf job)
	{
		String jobTracker = job.get("mapred.job.tracker", "local");
		return jobTracker.equals("local");
	}
	
	/**
	 * 
	 * @param job
	 */
	public static void setStagingDir( JobConf job )
	{
		String dir = DMLConfig.LOCAL_MR_MODE_STAGING_DIR + 
		             Lops.FILE_SEPARATOR + Lops.PROCESS_PREFIX + DMLScript.getUUID() + Lops.FILE_SEPARATOR;
		job.set( "mapreduce.jobtracker.staging.root.dir", dir );
	}
	
	public static void setInputInfo(JobConf job, byte input, InputInfo inputinfo, boolean targetToBlock, 
			int brlen, int bclen)
	{
		setInputInfo(job, input, inputinfo, targetToBlock, brlen, bclen, false);
	}
	
	public static void setInputInfo(JobConf job, byte input, InputInfo inputinfo, boolean targetToBlock, 
			int brlen, int bclen, boolean targetToWeightedCell)
	{
		Class<? extends Converter> converterClass=getConverterClass(inputinfo, targetToBlock, brlen, bclen, targetToWeightedCell);
		job.setClass(INPUT_CONVERTER_CLASS_PREFIX_CONFIG+input, converterClass, Converter.class);
		job.setClass(INPUT_KEY_CLASS_PREFIX_CONFIG+input, inputinfo.inputKeyClass, Writable.class);
		job.setClass(INPUT_VALUE_CLASS_PREFIX_CONFIG+input, inputinfo.inputValueClass, Writable.class);
	}
	
	
	public static void setOutputInfo(JobConf job, int i, OutputInfo outputinfo, boolean sourceInBlock)
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
				throw new RuntimeException("unsupported conversion: " + outputinfo.outputValueClass);
				// converterClass=IdenticalConverter.class; 
		}
		job.setClass(OUTPUT_CONVERTER_CLASS_PREFIX_CONFIG+i, converterClass, Converter.class);
	}
	
	@SuppressWarnings("unchecked")
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
	
	@SuppressWarnings("unchecked")
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
	public static void setProgramBlocksInMapper(JobConf job, String sProgramBlocks) 
	{
		job.set(PARFOR_PROGRAMBLOCKS_IN_MAPPER_CONFIG, sProgramBlocks);
	}
	
	public static String getProgramBlocksInMapper(JobConf job) 
	{
		String str = job.get(PARFOR_PROGRAMBLOCKS_IN_MAPPER_CONFIG);
		return str;
	}
	
	//partitioning configurations
	public static void setPartitioningInfoInMapper( JobConf job, long rlen, long clen, int brlen, int bclen, InputInfo ii, OutputInfo oi, PDataPartitionFormat dpf, int n, String fnameNew )
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
	
	public static void setPartitioningFilename( JobConf job, String fname )
	{
		job.set(PARTITIONING_OUTPUT_FILENAME_CONFIG, fname);
	}
	
	public static String getPartitioningFilename( JobConf job )
	{
		return job.get(PARTITIONING_OUTPUT_FILENAME_CONFIG);
	}
	
	public static void setResultMergeInfo( JobConf job, String fnameNew, InputInfo ii, String stagingDir )
	{
		job.set(RESULTMERGE_COMPARE_FILENAME_CONFIG, fnameNew);
		job.set(RESULTMERGE_INPUT_INFO_CONFIG, InputInfo.inputInfoToString(ii));
		job.set(RESULTMERGE_STAGING_DIR_CONFIG, stagingDir);
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
	
	public static byte[] getOutputIndexesInMapper(JobConf job)
	{
		String[] istrs=job.get(OUTPUT_INDEXES_IN_MAPPER_CONFIG).split(Instruction.INSTRUCTION_DELIM);
		return stringArrayToByteArray(istrs);
	}
	
	//get the indexes that this matrix file represents, 
	//since one matrix file can occur multiple times in a statement
	public static Vector<Byte> getInputMatrixIndexesInMapper(JobConf job) throws IOException
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
			String[] strs=str.split(",");
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
		Vector<Byte> representativeMatrixes=new Vector<Byte>();
		for(int i=0; i<matrices.length; i++)
		{
			Path p = new Path(matrices[i]).makeQualified(fs);
			if(thisFile.toUri().compareTo(p.toUri())==0 || thisDir.toUri().compareTo(p.toUri())==0)
				representativeMatrixes.add(indexes[i]);
			/*if(thisDirName.endsWith(matrices[i]) || thisFileName.endsWith(matrices[i]))
			{
				representativeMatrixes.add(indexes[i]);
				//LOG.info("add to representative: "+indexes[i]);
			}*/
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
	
	public static void setCombineInstructions(JobConf job, String combineInstructions)
	{
		job.set(COMBINE_INSTRUCTIONS_CONFIG, combineInstructions);
	}
	
	public static void setInstructionsInReducer(JobConf job, String instructionsInReducer)
	{
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
	public static RandInstruction[] getRandInstructions(JobConf job) throws DMLUnsupportedOperationException, DMLRuntimeException {
		String str=job.get(RAND_INSTRUCTIONS_CONFIG);
		return MRInstructionParser.parseRandInstructions(str);
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
	public static GroupedAggregateInstruction[] getGroupedAggregateInstructions(JobConf job) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		String str=job.get(GROUPEDAGG_INSTRUCTIONS_CONFIG);
		return MRInstructionParser.parseGroupedAggInstructions(str);
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
	
	public static void setMatrixDimension(JobConf job, byte matrixIndex, long rlen, long clen)
	{
		job.setLong(INPUT_MATRIX_NUM_ROW_PREFIX_CONFIG+matrixIndex, rlen);
		job.setLong(INPUT_MATRIX_NUM_COLUMN_PREFIX_CONFIG+matrixIndex, clen);
		//System.out.println("matrix "+matrixIndex+" with dimension: "+rlen+", "+clen);
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
		//System.out.println("matrix "+matrixIndex+" with block size: "+brlen+", "+bclen);
	}
	
	public static int getNumRowsPerBlock(JobConf job, byte matrixIndex)
	{
		return job.getInt(INPUT_BLOCK_NUM_ROW_PREFIX_CONFIG+matrixIndex, 1);
	}
	
	public static int getNumColumnsPerBlock(JobConf job, byte matrixIndex)
	{
		return job.getInt(INPUT_BLOCK_NUM_COLUMN_PREFIX_CONFIG+matrixIndex, 1);
	}
	
	public static PartitionParams getPartitionParams(JobConf job) {
		PartitionParams pp = new PartitionParams() ;
		if(job.getInt("isEL", 0) == 0) {	//cv
			pp.isEL = false;
			int ppcvt = job.getInt("cvt", -1) ;
			switch(ppcvt) {
				case 0: pp.cvt = PartitionParams.CrossvalType.kfold ; break ;
				case 1: pp.cvt = PartitionParams.CrossvalType.holdout ; break ;
				case 2: pp.cvt = PartitionParams.CrossvalType.bootstrap ; break ;
			}
			int pppt = job.getInt("pt", -1) ;
			switch(pppt) {
				case 0: pp.pt = PartitionParams.PartitionType.row ; break ;
				case 1: pp.pt = PartitionParams.PartitionType.submatrix ; break ;
				case 2: pp.pt = PartitionParams.PartitionType.cell ; break ;
			}
		}
		else {	//el
			pp.isEL = true;
			int ppet = job.getInt("et", -1) ;
			switch(ppet) {
				case 0: pp.et = PartitionParams.EnsembleType.bagging ; break ;
				case 1: pp.et = PartitionParams.EnsembleType.rsm ; break ;
				case 2: pp.et = PartitionParams.EnsembleType.rowholdout; break;
				case 3: pp.et = PartitionParams.EnsembleType.adaboost ; break ;
			}
		}
		pp.isColumn = (job.getInt("isColumn", 0) == 0) ? false : true;
		pp.toReplicate = (job.getInt("toReplicate", 0) == 0) ? false : true;
		pp.isSupervised = (job.getInt("isSupervised", 1) == 1) ? true : false;
		
		pp.numColGroups = job.getInt("numColGroups", 2) ;
		pp.numRowGroups = job.getInt("numRowGroups", 2) ;
		pp.numFolds = job.getInt("numFolds", 4) ;
		pp.frac = (double) job.getFloat("frac", (float) 0.34) ;
		pp.idToStratify = job.getInt("idToStratify", -1) ;	
		pp.numIterations = job.getInt("numIterations", 1) ;
		pp.sfmapfile = (String)job.get("sfmapfile");
		
		return pp ;
	}
	
	public static void setPartitionParams(JobConf job, PartitionParams pp) {
		job.setInt("numFolds",pp.numFolds) ;
		job.setFloat("frac", (float) pp.frac) ;
		job.set("sfmapfile", pp.sfmapfile);
		if(pp.toReplicate == true)
			job.setInt("toReplicate", 1);
		else
			job.setInt("toReplicate", 0);

		if(pp.isEL == true) {
			job.setInt("isEL", 1);	//el
			switch(pp.et) {
				case bagging: job.setInt("et", 0) ; break ;
				case rsm: job.setInt("et", 1) ; break ;
				case rowholdout: job.setInt("et", 2) ; break;
				case adaboost: job.setInt("et", 3) ; break ;
			}
		}
		else {
			job.setInt("isEL", 0);	//cv
			switch(pp.cvt) {
				case kfold: job.setInt("cvt", 0) ; break ;
				case holdout: job.setInt("cvt", 1) ; break ;
				case bootstrap: job.setInt("cvt", 2) ; break ;
			}
			switch(pp.pt) {
				case row: job.setInt("pt", 0) ; break ;
				case submatrix: job.setInt("pt", 1) ; break ;
				case cell: job.setInt("pt", 2) ; break ;
			}
		}
		
		if(pp.isColumn == true)
			job.setInt("isColumn", 1);
		else
			job.setInt("isColumn", 0);
		if(pp.isSupervised == true)
			job.setInt("isSupervised", 1);
		else
			job.setInt("isSupervised", 0);		
		
		job.setInt("idToStratify", pp.idToStratify) ;
		job.setInt("numIterations", pp.numIterations) ;		
		job.setInt("numRowGroups", pp.numRowGroups) ;
		job.setInt("numColGroups", pp.numColGroups) ;
	}
	
	public static void handleRecordReaderInstrucion(JobConf job, String recordReaderInstruction, String[] inputs, InputInfo[] inputInfos)
	{
		//TODO
	}
	
	public static void setupDistCacheInputs(JobConf job, String indices, String pathsString, ArrayList<String> paths) {
		job.set(DISTCACHE_INPUT_INDICES, indices);
		job.set(DISTCACHE_INPUT_PATHS, pathsString);
		Path p = null;
		for(String spath : paths) {
			p = new Path(spath);
			System.out.println("Adding file DistCache: " + p.toString());
			DistributedCache.addCacheFile(p.toUri(), job);
			DistributedCache.createSymlink(job);
		}
		//createAllSymlink(job, jobCacheDir, workDir)
	}
	
	public static String getDistCacheInputIndices(JobConf job) {
		return job.get(DISTCACHE_INPUT_INDICES);
	}
	
	public static String getDistCacheInputPaths(JobConf job) {
		return job.get(DISTCACHE_INPUT_PATHS);
	}
	
	private static String getCSVString(boolean[] flags) {
		if ( flags == null || flags.length == 0 )
			return "";
		
		StringBuilder s = new StringBuilder();
		s.append(flags[0]);
		for(int i=1; i < flags.length; i++) { 
			s.append(",");
			s.append(flags[i]);
		}
		
		return s.toString();
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
	
	private static String getCSVString(int[] arr) {
		if ( arr == null || arr.length == 0 )
			return "";
		
		StringBuilder s = new StringBuilder();
		s.append(arr[0]);
		for(int i=1; i < arr.length; i++) { 
			s.append(",");
			s.append(arr[i]);
		}
		
		return s.toString();
	}
	
	public static void setInputPartitioningInfo(JobConf job, boolean[] partitioned, PDataPartitionFormat[] pformats, int[] psizes) {
		job.set(PARTITION_FLAGS,  MRJobConfiguration.getCSVString(partitioned));
		job.set(PARTITION_FORMATS, MRJobConfiguration.getCSVString(pformats));
		job.set(PARTITION_SIZES, MRJobConfiguration.getCSVString(psizes));
	}
	
	private static boolean[] csv2boolean(String s) {
		String[] parts = s.split(",");
		boolean[] b = new boolean[parts.length];
		for(int i=0; i < parts.length; i++)
			b[i] = Boolean.parseBoolean(parts[i]);
		return b;
	}

	private static PDataPartitionFormat[] csv2PFormat(String s) {
		String[] parts = s.split(",");
		PDataPartitionFormat[] pformats = new PDataPartitionFormat[parts.length];
		for(int i=0; i < parts.length; i++) {
			pformats[i] = PDataPartitionFormat.parsePDataPartitionFormat(parts[i]);
		}
		return pformats;
	}

	private static int[] csv2int(String s) {
		String[] parts = s.split(",");
		int[] arr = new int[parts.length];
		for(int i=0; i < parts.length; i++) {
			arr[i] = Integer.parseInt(parts[i]);
		}
		return arr;
	}

	public static boolean[] getInputPartitionFlags(JobConf job) {
		return MRJobConfiguration.csv2boolean(job.get(PARTITION_FLAGS));
	}
	
	public static PDataPartitionFormat[] getInputPartitionFormats(JobConf job) {
		return MRJobConfiguration.csv2PFormat(job.get(PARTITION_FORMATS));
	}
	
	public static int[] getInputPartitionSizes(JobConf job) {
		return MRJobConfiguration.csv2int(job.get(PARTITION_SIZES));
	}
	
	public static void setUpMultipleInputs(JobConf job, byte[] inputIndexes, String[] inputs, InputInfo[] inputInfos, 
			boolean inBlockRepresentation, int[] brlens, int[] bclens)
	throws Exception
	{
		setUpMultipleInputs(job, inputIndexes, inputs, inputInfos, inBlockRepresentation, brlens, bclens, true);
	}
	
	public static void setUpMultipleInputs(JobConf job, byte[] inputIndexes, String[] inputs, InputInfo[] inputInfos, 
			boolean inBlockRepresentation, int[] brlens, int[] bclens, boolean setConverter)
	throws Exception
	{
		setUpMultipleInputs(job, inputIndexes, inputs, inputInfos, inBlockRepresentation, brlens, bclens, setConverter, false);
	}

	
	public static void setUpMultipleInputs(JobConf job, byte[] inputIndexes, String[] inputs, InputInfo[] inputInfos, 
			boolean inBlockRepresentation, int[] brlens, int[] bclens, boolean setConverter, boolean forCMJob) 
	throws Exception
	{
		if(inputs.length!=inputInfos.length)
			throw new Exception("number of inputs and inputInfos does not match");
		
		//set up names of the input matrices and their inputformat information
		job.setStrings(INPUT_MATRICIES_DIRS_CONFIG, inputs);
		MRJobConfiguration.setMapFucInputMatrixIndexes(job, inputIndexes);
		
		if(setConverter)
		{
			for(int i=0; i<inputs.length; i++)
				setInputInfo(job, inputIndexes[i], inputInfos[i], inBlockRepresentation, brlens[i], bclens[i], forCMJob);
		}
		
		//remove redundant input files
		Vector<Path> paths=new Vector<Path>();
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
			///////////////////////////////////////////
		//	FileInputFormat.addInputPath(job, p);
		//	job.setInputFormat(inputInfos[i].inputFormatClass);
			///////////////////////////////////////////
			paths.add(p);
		}
		
	}
	
	public static void updateResultDimsUnknown (JobConf job, byte[] updDimsUnknown) {
		job.set(RESULT_DIMS_UNKNOWN_CONFIG, MRJobConfiguration.getIndexesString(updDimsUnknown));
	}
	
	public static void setUpMultipleOutputs(JobConf job, byte[] resultIndexes, byte[] resultDimsUnknwon, String[] outputs, 
			OutputInfo[] outputInfos, boolean inBlockRepresentation, boolean mayContainCtable) 
	throws Exception
	{
		if(resultIndexes.length!=outputs.length)
			throw new Exception("number of outputs and result indexes does not match");
		if(outputs.length!=outputInfos.length)
			throw new Exception("number of outputs and outputInfos indexes does not match");
		
		job.set(RESULT_INDEXES_CONFIG, MRJobConfiguration.getIndexesString(resultIndexes));
		job.set(RESULT_DIMS_UNKNOWN_CONFIG, MRJobConfiguration.getIndexesString(resultDimsUnknwon));
		job.setStrings(OUTPUT_MATRICES_DIRS_CONFIG, outputs);
		job.setOutputCommitter(MultipleOutputCommitter.class);
		
		for(int i=0; i<outputs.length; i++)
		{
			MapReduceTool.deleteFileIfExistOnHDFS(new Path(outputs[i]), job);
			if ( mayContainCtable && resultDimsUnknwon[i] == (byte) 1 ) 
			{
				setOutputInfo(job, i, outputInfos[i], false);
			}
			else
				setOutputInfo(job, i, outputInfos[i], inBlockRepresentation);
			MultipleOutputs.addNamedOutput(job, Integer.toString(i), 
					outputInfos[i].outputFormatClass, outputInfos[i].outputKeyClass, 
					outputInfos[i].outputValueClass);
		}
		job.setOutputFormat(NullOutputFormat.class);
		
		// configure temp output
		Path tempOutputPath = new Path( constructTempOutputFilename() );
		FileOutputFormat.setOutputPath(job, tempOutputPath);
		MapReduceTool.deleteFileIfExistOnHDFS(tempOutputPath, job);
		
	/*	FileOutputFormat.setCompressOutput(job, true);
		SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK); */
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
		n=Math.min(n, ConfigurationManager.getConfig().getIntValue(DMLConfig.NUM_REDUCERS));
		n=Math.min(n, numFromCompiler);
		if(numReducerGroups>0)
			n=(int) Math.min(n, numReducerGroups);
		job.setNumReduceTasks(n); 
	}
	
/*	public static long getNumBlocks(MatrixCharacteristics dim)
	{
		return (long) (Math.ceil((double)dim.numRows/(double)dim.numRowsPerBlock)*Math.ceil((double)dim.numColumns/(double)dim.numColumnsPerBlock));
	}*/
	
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
	
	public static MatrixChar_N_ReducerGroups computeMatrixCharacteristics(JobConf job, byte[] inputIndexes, String randInstructions,
			String instructionsInMapper, String reblockInstructions, String aggInstructionsInReducer, String aggBinInstructions, 
			String otherInstructionsInReducer, byte[] resultIndexes, HashSet<Byte> mapOutputIndexes, boolean forMMCJ) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		HashSet<Byte> intermediateMatrixIndexes=new HashSet<Byte>();
		HashMap<Byte, MatrixCharacteristics> dims=new HashMap<Byte, MatrixCharacteristics>();
		for(byte i: inputIndexes){
			MatrixCharacteristics dim=new MatrixCharacteristics(getNumRows(job, i), getNumColumns(job, i), 
					getNumRowsPerBlock(job, i), getNumColumnsPerBlock(job, i));
			dims.put(i, dim);
		}
		RandInstruction[] randIns = null;
		randIns = MRInstructionParser.parseRandInstructions(randInstructions);
		if(randIns!=null)
		{
			for(RandInstruction ins: randIns)
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
				if(ins instanceof ZeroOutInstruction || ins instanceof AggregateUnaryInstruction|| ins instanceof RangeBasedReIndexInstruction)
				{
					UnaryMRInstructionBase tempIns=(UnaryMRInstructionBase) ins;
					setIntermediateMatrixCharactristics(job, tempIns.input, 
							dims.get(tempIns.input));
					intermediateMatrixIndexes.add(tempIns.input);
				}else if(ins instanceof AppendInstruction)
				{
					AppendInstruction tempIns=(AppendInstruction) ins;
					setIntermediateMatrixCharactristics(job, tempIns.input1, 
							dims.get(tempIns.input1));
					intermediateMatrixIndexes.add(tempIns.input1);
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
			for(Instruction ins: aggIns)
				MatrixCharacteristics.computeDimension(dims, (MRInstruction) ins);
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
					numReduceGroups=(long) Math.ceil((double)dim1.numColumns/(double)dim1.numColumnsPerBlock);
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
				long x=(long)Math.ceil((double)dim.numRows/(double)dim.numRowsPerBlock);
				long y=(long)Math.ceil((double)dim.numColumns/(double)dim.numColumnsPerBlock);
				
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
				if(ins instanceof ZeroOutInstruction || ins instanceof AggregateUnaryInstruction|| ins instanceof RangeBasedReIndexInstruction)
				{
					UnaryMRInstructionBase tempIns=(UnaryMRInstructionBase) ins;
					setIntermediateMatrixCharactristics(job, tempIns.input, 
							dims.get(tempIns.input));
					intermediateMatrixIndexes.add(tempIns.input);
				}
			}
		}
		
		setIntermediateMatrixIndexes(job, intermediateMatrixIndexes);
		
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
		
		job.setLong(INTERMEDIATE_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, dim.numRows);
		job.setLong(INTERMEDIATE_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, dim.numColumns);
		job.setInt(INTERMEDIATE_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, dim.numRowsPerBlock);
		job.setInt(INTERMEDIATE_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, dim.numColumnsPerBlock);
	}
	
	public static MatrixCharacteristics getIntermediateMatrixCharactristics(JobConf job, byte tag)
	{
		MatrixCharacteristics dim=new MatrixCharacteristics();
		dim.numRows=job.getLong(INTERMEDIATE_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, 0);
		dim.numColumns=job.getLong(INTERMEDIATE_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, 0);
		dim.numRowsPerBlock=job.getInt(INTERMEDIATE_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, 1);
		dim.numColumnsPerBlock=job.getInt(INTERMEDIATE_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, 1);
		return dim;
	}

	public static void setMatrixCharactristicsForOutput(JobConf job,
			byte tag, MatrixCharacteristics dim)
	{
		if (dim == null){
			System.out.println("setMatrixCharactristicsForOutput:dim is NULL");
		}
		job.setLong(OUTPUT_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, dim.numRows);
		job.setLong(OUTPUT_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, dim.numColumns);
		job.setInt(OUTPUT_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, dim.numRowsPerBlock);
		job.setInt(OUTPUT_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, dim.numColumnsPerBlock);
	}
	
	public static MatrixCharacteristics getMatrixCharacteristicsForOutput(JobConf job, byte tag)
	{
		MatrixCharacteristics dim=new MatrixCharacteristics();
		dim.numRows=job.getLong(OUTPUT_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, 0);
		dim.numColumns=job.getLong(OUTPUT_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, 0);
		dim.numRowsPerBlock=job.getInt(OUTPUT_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, 1);
		dim.numColumnsPerBlock=job.getInt(OUTPUT_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, 1);
		return dim;
	}
	
	public static void setMatrixCharactristicsForReblock(JobConf job,
			byte tag, MatrixCharacteristics dim)
	{
		job.setLong(REBLOCK_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, dim.numRows);
		job.setLong(REBLOCK_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, dim.numColumns);
		job.setInt(REBLOCK_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, dim.numRowsPerBlock);
		job.setInt(REBLOCK_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, dim.numColumnsPerBlock);
	}
	
	public static MatrixCharacteristics getMatrixCharactristicsForReblock(JobConf job, byte tag)
	{
		MatrixCharacteristics dim=new MatrixCharacteristics();
		dim.numRows=job.getLong(REBLOCK_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, 0);
		dim.numColumns=job.getLong(REBLOCK_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, 0);
		dim.numRowsPerBlock=job.getInt(REBLOCK_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, 1);
		dim.numColumnsPerBlock=job.getInt(REBLOCK_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, 1);
		return dim;
	}
	
	public static void setMatrixCharactristicsForBinAgg(JobConf job,
			byte tag, MatrixCharacteristics dim) {
		job.setLong(AGGBIN_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, dim.numRows);
		job.setLong(AGGBIN_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, dim.numColumns);
		job.setInt(AGGBIN_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, dim.numRowsPerBlock);
		job.setInt(AGGBIN_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, dim.numColumnsPerBlock);
	}
	
	public static MatrixCharacteristics getMatrixCharactristicsForBinAgg(JobConf job, byte tag)
	{
		MatrixCharacteristics dim=new MatrixCharacteristics();
		dim.numRows=job.getLong(AGGBIN_MATRIX_NUM_ROW_PREFIX_CONFIG+tag, 0);
		dim.numColumns=job.getLong(AGGBIN_MATRIX_NUM_COLUMN_PREFIX_CONFIG+tag, 0);
		dim.numRowsPerBlock=job.getInt(AGGBIN_BLOCK_NUM_ROW_PREFIX_CONFIG+tag, 1);
		dim.numColumnsPerBlock=job.getInt(AGGBIN_BLOCK_NUM_COLUMN_PREFIX_CONFIG+tag, 1);
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
		
		RandInstruction[] randIns = null;
		randIns = MRInstructionParser.parseRandInstructions(randInstructions);
		getIndexes(randIns, indexesInMapper);
		
	//	System.out.println("indexes used in the mapper: "+indexesInMapper);
		Instruction[] insMapper = MRInstructionParser.parseMixedInstructions(instructionsInMapper);
		getIndexes(insMapper, indexesInMapper);
	//	System.out.println("indexes used in the mapper: "+indexesInMapper);
		
		ReblockInstruction[] reblockIns = null;
		reblockIns = MRInstructionParser.parseReblockInstructions(reblockInstructions);
		getIndexes(reblockIns, indexesInMapper);
		
		Instruction[] insReducer = MRInstructionParser.parseAggregateInstructions(aggInstructionsInReducer);
		HashSet<Byte> indexesInReducer=new HashSet<Byte>();
		getIndexes(insReducer, indexesInReducer);
	//	System.out.println("indexes used in the reducer: "+indexesInReducer);
		
		insReducer = InstructionParser.parseMixedInstructions(otherInstructionsInReducer);
		getIndexes(insReducer, indexesInReducer);
	//	System.out.println("indexes used in the reducer: "+indexesInReducer);
	
		for(byte ind: resultIndexes)
			indexesInReducer.add(ind);
	//	System.out.println("indexes used in the reducer: "+indexesInReducer);
		
		indexesInMapper.retainAll(indexesInReducer);
	//	System.out.println("indexes needed to be output: "+indexesInMapper);
		
		job.set(OUTPUT_INDEXES_IN_MAPPER_CONFIG, getIndexesString(indexesInMapper));
		return indexesInMapper;
	}
	
	public static CollectMultipleConvertedOutputs getMultipleConvertedOutputs(JobConf job)
	{	
	/*	byte[] resultIndexes=MRJobConfiguration.getResultIndexes(job);
		HashMap<Byte, Converter> outputConverters=new HashMap<Byte, Converter>(resultIndexes.length);
		HashMap<Byte, MatrixCharacteristics> stats=new HashMap<Byte, MatrixCharacteristics>();
		for(byte output: resultIndexes)
		{
			Converter conv=getOuputConverter(job, output);
			outputConverters.put(output, conv);
			stats.put(output, MRJobConfiguration.getMatrixCharactristicsForOutput(job, output));
		}
		
		MultipleOutputs multipleOutputs=new MultipleOutputs(job);
		
		return new CollectMultipleConvertedOutputs(outputConverters, stats, multipleOutputs);*/
	
		byte[] resultIndexes=MRJobConfiguration.getResultIndexes(job);
		Converter[] outputConverters=new Converter[resultIndexes.length];
		MatrixCharacteristics[] stats=new MatrixCharacteristics[resultIndexes.length];
		HashMap<Byte, Vector<Integer>> tagMapping=new HashMap<Byte, Vector<Integer>>();
		for(int i=0; i<resultIndexes.length; i++)
		{
			byte output=resultIndexes[i];
			Vector<Integer> vec=tagMapping.get(output);
			if(vec==null)
			{
				vec=new Vector<Integer>();
				tagMapping.put(output, vec);
			}
			vec.add(i);
			
			outputConverters[i]=getOuputConverter(job, i);
			stats[i]=MRJobConfiguration.getMatrixCharacteristicsForOutput(job, output);
		}
		
		MultipleOutputs multipleOutputs=new MultipleOutputs(job);
		
		return new CollectMultipleConvertedOutputs(outputConverters, stats, multipleOutputs);
		
	}
	
	private static void getIndexes(Instruction[] instructions, HashSet<Byte> indexes) throws DMLRuntimeException
	{
		if(instructions==null)
			return;
		for(Instruction ins: instructions)
		{
			for(byte i: ins.getAllIndexes())
				indexes.add(i);
		}
	}
	
	private static String getIndexesString(HashSet<Byte> indexes)
	{
		if(indexes==null || indexes.isEmpty())
			return "";
		String str="";
		for(Byte ind: indexes)
			str += (ind + Instruction.INSTRUCTION_DELIM);
		return str.substring(0, str.length()-1);//remove the last delim
	}
	
	private static String getIndexesString(byte[] indexes)
	{
		if(indexes==null)
			return "";
		String str="";
		for(byte ind: indexes)
			str+=(ind+Instruction.INSTRUCTION_DELIM);
		return str.substring(0, str.length()-1);//remove the last delim
	}

	private static String getIndexesString(ArrayList<Byte> indexes)
	{
		if(indexes==null)
			return "";
		String str="";
		for(byte ind: indexes)
			str+=(ind+Instruction.INSTRUCTION_DELIM);
		return str.substring(0, str.length()-1);//remove the last delim
	}

	public static void setMapFucInputMatrixIndexes(JobConf job,
			byte[] realIndexes) {
		job.set(MAPFUNC_INPUT_MATRICIES_INDEXES_CONFIG, getIndexesString(realIndexes));
		
	}

	public static boolean deriveRepresentation(InputInfo[] inputInfos) {
		for(InputInfo input: inputInfos)
		{
			if(!(input.inputValueClass==MatrixBlock.class 
					|| input.inputValueClass==MatrixBlock1D.class 
					|| input.inputValueClass==MatrixBlockDSM.class))
			{
				return false;
			}	
		}
		return true;
	}
	
	private static String constructTempOutputFilename() 
	{
		StringBuilder sb = new StringBuilder();
		sb.append(ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE));
		sb.append(Lops.FILE_SEPARATOR);
		sb.append(Lops.PROCESS_PREFIX);
		sb.append(DMLScript.getUUID());
		sb.append(Lops.FILE_SEPARATOR);
		
		sb.append("TmpOutput"+seq.getNextID());
		
		//old unique dir (no guarantees): 
		//sb.append(Integer.toHexString(new Random().nextInt(Integer.MAX_VALUE))); 
		
		return sb.toString(); 
	}
	
	private static String constructPartitionFilename() 
	{
		StringBuilder sb = new StringBuilder();
		sb.append(ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE));
		sb.append(Lops.FILE_SEPARATOR);
		sb.append(Lops.PROCESS_PREFIX);
		sb.append(DMLScript.getUUID());
		sb.append(Lops.FILE_SEPARATOR);
		
		sb.append(SamplingSortMRInputFormat.PARTITION_FILENAME+seq.getNextID());
		
		//old unique dir (no guarantees): 
		//sb.append(Integer.toHexString(new Random().nextInt(Integer.MAX_VALUE))); 
		
		return sb.toString(); 
	}
}

