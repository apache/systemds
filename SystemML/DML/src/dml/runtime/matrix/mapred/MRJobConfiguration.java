package dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Vector;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.hadoop.mapred.lib.NullOutputFormat;

import dml.meta.PartitionParams;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.io.AddDummyWeightConverter;
import dml.runtime.matrix.io.BinaryBlockToBinaryCellConverter;
import dml.runtime.matrix.io.BinaryBlockToTextCellConverter;
import dml.runtime.matrix.io.BinaryCellToTextConverter;
import dml.runtime.matrix.io.Converter;
import dml.runtime.matrix.io.IdenticalConverter;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixCell;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.MultipleOutputCommitter;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.io.TextToBinaryCellConverter;
import dml.runtime.matrix.io.WeightedCellToSortInputConverter;
import dml.runtime.matrix.io.hadoopfix.MultipleInputs;
import dml.runtime.util.MapReduceTool;
import dml.runtime.instructions.*;
import dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import dml.runtime.instructions.MRInstructions.AggregateInstruction;
import dml.runtime.instructions.MRInstructions.AggregateUnaryInstruction;
import dml.runtime.instructions.MRInstructions.CM_N_COVInstruction;
import dml.runtime.instructions.MRInstructions.CombineBinaryInstruction;
import dml.runtime.instructions.MRInstructions.GroupedAggregateInstruction;
import dml.runtime.instructions.MRInstructions.MRInstruction;
import dml.runtime.instructions.MRInstructions.RandInstruction;
import dml.runtime.instructions.MRInstructions.ReblockInstruction;
import dml.runtime.instructions.MRInstructions.SelectInstruction;
import dml.runtime.instructions.MRInstructions.UnaryInstruction;
import dml.runtime.instructions.MRInstructions.UnaryMRInstructionBase;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;
import dml.runtime.matrix.io.WeightedPair;

public class MRJobConfiguration {
	
	//Job configurations
	
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
	
	private static final String PARTIAL_AGG_CACHE_SIZE="partial.aggregate.cache.size";
	
	
	/*
	 * SystemML Counter Group names
	 * 
	 * group name for the counters on number of output nonZeros
	 */
	public static final String NUM_NONZERO_CELLS="nonzeros";
	
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
		
		Path thisFile=new Path(job.get("map.input.file"));
		FileSystem fs=FileSystem.get(job);
		//Path p=new Path(thisFileName);
		
		Path thisDir=thisFile.getParent();
		Vector<Byte> representativeMatrixes=new Vector<Byte>();
		for(int i=0; i<matrices.length; i++)
		{
			Path p = fs.getFileStatus(new Path(matrices[i])).getPath();
			if(thisFile.compareTo(p)==0 || thisDir.compareTo(p)==0)
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
			OutputInfo[] outputInfos, boolean inBlockRepresentation) 
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
			setOutputInfo(job, i, outputInfos[i], inBlockRepresentation);
			MultipleOutputs.addNamedOutput(job, Integer.toString(i), 
					outputInfos[i].outputFormatClass, outputInfos[i].outputKeyClass, 
					outputInfos[i].outputValueClass);
		}
		job.setOutputFormat(NullOutputFormat.class);
		
		// configure temp output
		Path tempOutputPath = new Path(Integer.toHexString(new Random().nextInt(Integer.MAX_VALUE)));
		FileOutputFormat.setOutputPath(job, tempOutputPath);
		MapReduceTool.deleteFileIfExistOnHDFS(tempOutputPath, job);
		
	/*	FileOutputFormat.setCompressOutput(job, true);
		SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK); */
	}
	
	public static MatrixCharacteristics[] computeMatrixCharacteristics(JobConf job, byte[] inputIndexes, 
			String instructionsInMapper, String aggInstructionsInReducer, String aggBinInstructions, 
			String otherInstructionsInReducer, byte[] resultIndexes) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		return computeMatrixCharacteristics(job, inputIndexes, null, instructionsInMapper, null, aggInstructionsInReducer, 
				aggBinInstructions, otherInstructionsInReducer, resultIndexes);
	}
	
	public static MatrixCharacteristics[] computeMatrixCharacteristics(JobConf job, byte[] inputIndexes, 
			String instructionsInMapper, String reblockInstructions, String aggInstructionsInReducer, String aggBinInstructions, 
			String otherInstructionsInReducer, byte[] resultIndexes) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		return computeMatrixCharacteristics(job, inputIndexes, null, instructionsInMapper, reblockInstructions, aggInstructionsInReducer, 
				aggBinInstructions, otherInstructionsInReducer, resultIndexes);
	}
	public static MatrixCharacteristics[] computeMatrixCharacteristics(JobConf job, byte[] inputIndexes, String randInstructions,
			String instructionsInMapper, String reblockInstructions, String aggInstructionsInReducer, String aggBinInstructions, 
			String otherInstructionsInReducer, byte[] resultIndexes) throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		HashSet<Byte> intermediateMatrixIndexes=new HashSet<Byte>();
		HashMap<Byte, MatrixCharacteristics> dims=new HashMap<Byte, MatrixCharacteristics>();
		for(byte i: inputIndexes){
			dims.put(i, new MatrixCharacteristics(getNumRows(job, i), getNumColumns(job, i), 
					getNumRowsPerBlock(job, i), getNumColumnsPerBlock(job, i)));
		}
		RandInstruction[] randIns = null;
		randIns = MRInstructionParser.parseRandInstructions(randInstructions);
		if(randIns!=null)
		{
			for(RandInstruction ins: randIns)
				MatrixCharacteristics.computeDimension(dims, ins);
		}
		
		MRInstruction[] insMapper = MRInstructionParser.parseMixedInstructions(instructionsInMapper);
		if(insMapper!=null)
		{
			for(MRInstruction ins: insMapper)
			{
				MatrixCharacteristics.computeDimension(dims, ins);
				if(ins instanceof SelectInstruction || ins instanceof AggregateUnaryInstruction)
				{
					UnaryMRInstructionBase tempIns=(UnaryMRInstructionBase) ins;
					setIntermediateMatrixCharactristics(job, tempIns.input, 
							dims.get(tempIns.input));
					intermediateMatrixIndexes.add(tempIns.input);
				}
			}
		}
		
		Instruction[] aggIns = MRInstructionParser.parseAggregateInstructions(aggInstructionsInReducer);
		if(aggIns!=null)
		{
			for(Instruction ins: aggIns)
				MatrixCharacteristics.computeDimension(dims, (MRInstruction) ins);
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
			}
		}
		
		MRInstruction[] insReducer = MRInstructionParser.parseMixedInstructions(otherInstructionsInReducer);
		if(insReducer!=null)
		{
			for(MRInstruction ins: insReducer)
			{
				MatrixCharacteristics.computeDimension(dims, ins);
				if(ins instanceof SelectInstruction || ins instanceof AggregateUnaryInstruction)
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
		
		return stats;
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
	
	public static void setUpOutputIndexesForMapper(JobConf job, byte[] inputIndexes, String instructionsInMapper, 
			String aggInstructionsInReducer, String otherInstructionsInReducer, byte[] resultIndexes) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		setUpOutputIndexesForMapper(job, inputIndexes, null, instructionsInMapper, 
				null, aggInstructionsInReducer, otherInstructionsInReducer, resultIndexes);
	}
	
	public static void setUpOutputIndexesForMapper(JobConf job, byte[] inputIndexes, String instructionsInMapper, 
			String reblockInstructions, String aggInstructionsInReducer, String otherInstructionsInReducer, byte[] resultIndexes) 
	throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		setUpOutputIndexesForMapper(job, inputIndexes, null, instructionsInMapper, 
				reblockInstructions, aggInstructionsInReducer, otherInstructionsInReducer, resultIndexes);
	}
	public static void setUpOutputIndexesForMapper(JobConf job, byte[] inputIndexes, String randInstructions, String instructionsInMapper, 
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

	public static void setMapFucInputMatrixIndexes(JobConf job,
			byte[] realIndexes) {
		job.set(MAPFUNC_INPUT_MATRICIES_INDEXES_CONFIG, getIndexesString(realIndexes));
		
	}


}
