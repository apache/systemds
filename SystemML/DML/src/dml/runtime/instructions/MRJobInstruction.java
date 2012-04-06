package dml.runtime.instructions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import dml.lops.Lops;
import dml.lops.compile.JobType;
import dml.meta.PartitionParams;
import dml.runtime.instructions.CPInstructions.Data;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;
import dml.utils.DMLRuntimeException;

/*
---------------------------------------------------------------------------------------
JobType       Rand    RecordReader    Mapper     Shuffle   AggInReducer  OtherInReducer
---------------------------------------------------------------------------------------
GMR                       *            *                         *              *    
RAND            *                      *                         *              *    
REBLOCK                                *            *                           *
MMCJ                                   *            *
MMRJ                                   *            *                           *
CM_COV                                 *            *
GROUPED_AGG                                         *                           *
COMBINE                                             *    
SORT                                   *            *
PARTITION
---------------------------------------------------------------------------------------
 */

public class MRJobInstruction extends Instruction
{

	//public enum JobType {MMCJ, MMRJ, GMR, Partition, RAND, ReBlock, SortKeys, Combine, CMCOV, GroupedAgg}; 
	JobType jobType;
	
	
	public JobType getJobType()
	{
		return jobType;
	}

	public String[] getIv_inputs()
	{
		return iv_inputs;
	}

	public InputInfo[] getIv_inputInfos()
	{
		return iv_inputInfos;
	}

	public String getIv_instructionsInMapper()
	{
		return iv_instructionsInMapper;
	}
	
	public String getIv_recordReaderInstructions()
	{
		return iv_recordReaderInstructions;
	}
	
	public String getIv_randInstructions() 
	{
		return iv_randInstructions;
	}

	public String getIv_shuffleInstructions()
	{
		return iv_shuffleInstructions;
	}

	public String getIv_aggInstructions()
	{
		return iv_aggInstructions;
	}

	public String getIv_otherInstructions()
	{
		return iv_otherInstructions;
	}

	public String[] getIv_outputs()
	{
		return iv_outputs;
	}

	public OutputInfo[] getIv_outputInfos()
	{
		return iv_outputInfos;
	}

	public byte[] getIv_resultIndices()
	{
		return iv_resultIndices;
	}

	public int getIv_numReducers()
	{
		return iv_numReducers;
	}

	public int getIv_replication()
	{
		return iv_replication;
	}

	String[] iv_inputs;
	InputInfo[] iv_inputInfos;
	
	public long[] getIv_rows()
	{
		return iv_rows;
	}

	public long[] getIv_cols()
	{
		return iv_cols;
	}

	public int[] getIv_num_rows_per_block()
	{
		for(int i=0; i < iv_num_rows_per_block.length; i++)
		{
			if(iv_num_rows_per_block[i] == -1)
				iv_num_rows_per_block[i] = 1;
		}
		return iv_num_rows_per_block;
	}

	public int[] getIv_num_cols_per_block()
	{
		for(int i=0; i < iv_num_cols_per_block.length; i++)
		{
			if(iv_num_cols_per_block[i] == -1)
				iv_num_cols_per_block[i] = 1;
		}
		return iv_num_cols_per_block;
	}

	public long [] iv_rows; 
	public long [] iv_cols;
	int [] iv_num_rows_per_block;
	int [] iv_num_cols_per_block;
	public String iv_randInstructions = "";
	String iv_recordReaderInstructions = "";
	String iv_instructionsInMapper = ""; 
	String iv_shuffleInstructions = ""; 
	String iv_aggInstructions = "";
	String iv_otherInstructions = "";
	String[] iv_outputs;
	OutputInfo[] iv_outputInfos;
	byte [] iv_resultIndices;
	int iv_numReducers;
	int iv_replication;
	
	/*
	 *  For each result index i, iv_resultDimsUnknown[i] indicates whether or not the output matrix dimensions are known at compile time
	 *  iv_resultDimsUnknown[i] = 0 --> dimensions are known at compile time
	 *  iv_resultDimsUnknown[i] = 1 --> dimensions are unknown at compile time
	 */
	public byte[] iv_resultDimsUnknown; 
	public byte[] getIv_resultDimsUnknown()
	{
		return iv_resultDimsUnknown;
	}

	
	ArrayList <String> inputLabels;
	public ArrayList<String> getInputLabels()
	{
		return inputLabels;
	}


	ArrayList <String> outputLabels;
	public ArrayList<String> getOutputLabels()
	{
		return outputLabels;
	}

	
	HashMap<String, Data> inputLabelValues;
	public void setInputLabelValueMapping(HashMap <String,Data> labValues)
	{
	
		/*
		Iterator <String> it = labValues.keySet().iterator();
		while(it.hasNext())
		{
			String s = it.next();
			
		}
		*/
		
		inputLabelValues = labValues;
	}
	
	public void setOutputLabelValueMapping(HashMap <String,Data> labValues)
	{
	
		/*
		Iterator <String> it = labValues.keySet().iterator();
		while(it.hasNext())
		{
			String s = it.next();
			
		}
		*/
		
		outputLabelValues = labValues;
	}
	
	
	public HashMap<String, Data> getInputLabelValueMapping()
	{
		return inputLabelValues;
	}
	
	HashMap<String, Data> outputLabelValues;
	public HashMap<String, Data> getOutputLabelValueMapping()
	{
		return outputLabelValues;
	}

	// Used for partitioning jobs..
	private PartitionParams partitionParams;
	
	public PartitionParams getPartitionParams() {
		return partitionParams ;
	}
	

	
	/**
	 * Constructor
	 * @param instType
	 * @param type
	 */
	
	public MRJobInstruction(JobType type)
	{
		setType(Instruction.INSTRUCTION_TYPE.MAPREDUCE_JOB);
		jobType = type;	
	}
	
	/**
	 * Method to set inputs (HDFS file paths, and input formats) of a MapReduce Instruction
	 * @param inputLocations
	 * @param inputInfos
	 * @throws DMLRuntimeException 
	 */
	public void setInputs(String []inputLocations, InputInfo []inputInfos) throws DMLRuntimeException {
		if ( inputLocations.length != inputInfos.length ) {
			throw new DMLRuntimeException("Unexpected error while setting inputs for MapReduce Instruction -- size of inputLocations (" + inputLocations.length + ") and inputInfos (" + inputInfos.length + ") do not match.");
		}
		iv_inputs = inputLocations;
		iv_inputInfos = inputInfos;
	}
	
	/**
	 * Method to set dimensions and block dimensions for inputs of a MapReduce Instruction.
	 *  
	 * @param inputRows
	 * @param inputRowBlocks
	 * @param inputCols
	 * @param inputColBlocks
	 * @throws DMLRuntimeException
	 */
	public void setInputDimensions(long[] inputRows, int[] inputRowBlocks, long[] inputCols, int[] inputColBlocks) throws DMLRuntimeException {
		if ( inputRows.length != inputCols.length || inputRows.length != inputRowBlocks.length || inputRows.length != inputColBlocks.length ) {
			throw new DMLRuntimeException("Unexpected error while setting input dimensions (" + inputRows.length + ", " + inputCols.length + "," + inputRowBlocks.length + "," + inputColBlocks.length + ") for MapReduce Instruction.");
		}
		iv_rows = inputRows;
		iv_cols = inputCols;
		iv_num_rows_per_block = inputRowBlocks; 
		iv_num_cols_per_block = inputColBlocks; 
	}
	
	/**
	 * Method to set outputs (HDFS file paths, output formats, output indices) for a MapReduce instruction.
	 * 
	 * @param outputLocations
	 * @param outputInfos
	 * @throws DMLRuntimeException
	 */
	public void setOutputs(String[] outputLocations, OutputInfo[] outputInfos, byte[] outputIndices) throws DMLRuntimeException {
		if ( outputLocations.length != outputInfos.length || outputLocations.length != outputIndices.length) {
			throw new DMLRuntimeException("Unexpected error while setting outputs (" + outputLocations.length + ", " + outputInfos.length + ", " + outputIndices.length + ") for MapReduce Instruction.");
		}
		iv_outputs = outputLocations;
		iv_outputInfos = outputInfos;
		iv_resultIndices = outputIndices;
	}
	
	public void setOutputDimensions(byte[] outputDimensionsUnknown) {
		iv_resultDimsUnknown = outputDimensionsUnknown;
	}
	
	/**
	 * Method to set the number of reducers for a MapReducer instruction.
	 * @param numReducers
	 */
	public void setNumberOfReducers(int numReducers) {
		iv_numReducers = numReducers;
	}
	
	/**
	 * Method to set the replication factor for outputs produced from a MapReduce instruction.
	 * 
	 * @param replication
	 */
	public void setReplication(int replication) {
		iv_replication = replication;
	}
	
	/**
	 * Method to set input and output labels for a MapReduce instruction.
	 * 
	 * @param inputLabels
	 * @param outputLabels
	 */
	public void setInputOutputLabels(ArrayList<String> inputLabels, ArrayList<String> outputLabels) {
		this.inputLabels = inputLabels;
		this.outputLabels = outputLabels;
	}
	
	public void setRecordReaderInstructions(String rrInstructions) {
		iv_recordReaderInstructions = rrInstructions;
	}
	
	public void setMapperInstructions(String mapperInstructions) {
		iv_instructionsInMapper = mapperInstructions;
	}
	
	public void setShuffleInstructions(String shuffleInstructions) {
		iv_shuffleInstructions = shuffleInstructions;
	}
	
	public void setAggregateInstructionsInReducer(String aggInstructions) {
		iv_aggInstructions = aggInstructions;
	}
	
	public void setOtherInstructionsInReducer(String otherInstructions) {
		iv_otherInstructions = otherInstructions;
	}
	
	public void setRandInstructions(String randInstructions) {
		iv_randInstructions = randInstructions;
	}
	
	public void setPartitionInstructions(String inputs[], InputInfo[] inputInfo, String[] outputs,  int numReducers, int replication,
			long[] nr, long[] nc, int[] bnr, int[] bnc, byte[] resultIndexes, byte[] resultDimsUnknown, PartitionParams pp, ArrayList <String> inLabels, 
			ArrayList <String> outLabels, HashMap <String, Data> outputLabelValueMapping) {
		this.iv_inputs = inputs ;
		this.iv_inputInfos = inputInfo ;
		this.iv_outputs = outputs ;
		
		
		//this.iv_outputInfos = new OutputInfo[1] ; 
		//this.iv_outputInfos[0] = OutputInfo.BinaryBlockOutputInfo ;
		
		this.iv_outputInfos = new OutputInfo[this.iv_outputs.length];
		for (int i = 0; i <this.iv_outputs.length; i++ )
			this.iv_outputInfos[i] = OutputInfo.BinaryBlockOutputInfo; 
		
		
		this.iv_numReducers = numReducers ;
		this.iv_replication = replication ;
		this.iv_rows = nr ;
		this.iv_cols = nc ;
		this.iv_num_rows_per_block = bnr ;
		this.iv_num_cols_per_block = bnc ;
		this.iv_resultIndices = resultIndexes ;
		this.iv_resultDimsUnknown = resultDimsUnknown;
		this.partitionParams = pp ;
		this.inputLabels = inLabels ;
		this.outputLabels = outLabels ;
		this.outputLabelValues = outputLabelValueMapping ;
	}

	public void setGMRInstructions(String [] input, InputInfo [] inputInfo, long [] numRows, long [] numCols, int [] num_rows_per_block, int [] num_cols_per_block, 
			String recordReaderInstructions, String mapperInstructions, 
			String aggInstructions, String otherInstructions, String [] output, OutputInfo [] outputInfo, byte [] resultIndex, byte[] resultDimsUnknown, 
			int numReducers, int replication, ArrayList <String> inLabels, ArrayList <String> outLabels)
	{
		try {
			setInputs(input, inputInfo);
			setInputDimensions(numRows, num_rows_per_block, numCols, num_cols_per_block);
			setOutputs(output, outputInfo, resultIndex);
			setOutputDimensions(resultDimsUnknown);
			
			setRecordReaderInstructions(recordReaderInstructions);
			setMapperInstructions(mapperInstructions);
			setShuffleInstructions("");
			setAggregateInstructionsInReducer(aggInstructions);
			setOtherInstructionsInReducer(otherInstructions);
			
			setInputOutputLabels(inLabels, outLabels);
			
			setNumberOfReducers(numReducers);
			setReplication(replication);
			
		} catch (DMLRuntimeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}	

	public void setRandInstructions(long [] numRows, long [] numCols, int [] num_rows_per_block, int [] num_cols_per_block, 
			String randInstructions, String mapperInstructions, 
			String aggInstructions, String otherInstructions, String [] output, OutputInfo [] outputInfo, byte [] resultIndex, byte[] resultDimsUnknown,
			int numReducers, int replication, ArrayList <String> inLabels, ArrayList <String> outLabels)
	{
		try {
			//setInputs(null, null);
			setInputDimensions(numRows, num_rows_per_block, numCols, num_cols_per_block);
			setOutputs(output, outputInfo, resultIndex);
			setOutputDimensions(resultDimsUnknown);
			
			setRecordReaderInstructions("");
			setRandInstructions(randInstructions);
			setMapperInstructions(mapperInstructions);
			setShuffleInstructions("");
			setAggregateInstructionsInReducer(aggInstructions);
			setOtherInstructionsInReducer(otherInstructions);
			
			setInputOutputLabels(inLabels, outLabels);
			
			setNumberOfReducers(numReducers);
			setReplication(replication);
			
		} catch (DMLRuntimeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
/*		this.iv_inputs = null; // inputs;
		this.iv_inputInfos = null; // inputInfo;
		this.iv_outputs = output;
		this.iv_outputInfos = outputInfo;
		this.iv_randInstructions = randInstructions;
		this.iv_instructionsInMapper = mapperInstructions;
		this.iv_aggInstructions = aggInstructions;
		this.iv_otherInstructions = otherInstructions;
		this.iv_resultIndices = resultIndex;
		this.iv_resultDimsUnknown = resultDimsUnknown;
		this.iv_numReducers = numReducers;
		this.iv_replication = replication;
		this.inputLabels = inLabels;
		this.outputLabels = outLabels;
		//this.outputLabelValues = outputLabelValueMapping;
		
		this.iv_rows = numRows;
		this.iv_cols = numCols;
		this.iv_num_rows_per_block = num_rows_per_block;
		this.iv_num_cols_per_block = num_cols_per_block;*/
	}
	

	public void setMMCJInstructions(String [] input, InputInfo [] inputInfo, long [] numRows, long [] numCols, int [] num_rows_per_block, int [] num_cols_per_block, String mapperInstructions, 
			String shuffleInstructions, String [] output, OutputInfo [] outputInfo, byte [] resultIndex, byte[] resultDimsUnknown, 
			int numReducers, int replication, ArrayList <String> inLabels, ArrayList <String> outLabels)
	{
		try {
			setInputs(input, inputInfo);
			setInputDimensions(numRows, num_rows_per_block, numCols, num_cols_per_block);
			setOutputs(output, outputInfo, resultIndex);
			setOutputDimensions(resultDimsUnknown);
			
			setMapperInstructions(mapperInstructions);
			setShuffleInstructions(shuffleInstructions);
			setAggregateInstructionsInReducer("");
			setOtherInstructionsInReducer("");
			
			setInputOutputLabels(inLabels, outLabels);
			
			setNumberOfReducers(numReducers);
			setReplication(replication);
			
		} catch (DMLRuntimeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
/*		iv_inputs = input;
		iv_inputInfos = inputInfo;
		iv_instructionsInMapper = mapperInstructions;
		iv_shuffleInstructions = aggBinInstructions;
		iv_outputs = output;
		iv_outputInfos = outputInfo;
		iv_resultIndices = resultIndex;
		iv_resultDimsUnknown = resultDimsUnknown;
		iv_numReducers = numReducers;
		iv_replication = replication;
		iv_rows = numRows;
		iv_cols = numCols;
		iv_num_rows_per_block = num_rows_per_block;
		iv_num_cols_per_block = num_cols_per_block;
		inputLabels = inLabels;
		outputLabels = outLabels;
		//outputLabelValues = outputLabelValueMapping;
*/	}

	public void setMMRJInstructions(String [] input, InputInfo [] inputInfo, long [] numRows, long [] numCols, int [] num_rows_per_block, int [] num_cols_per_block, String mapperInstructions, 
			String shuffleInstructions, String aggInstructions, String otherInstructions, String [] output, OutputInfo [] outputInfo, byte [] resultIndex, byte[] resultDimsUnknown, 
			int numReducers, int replication, ArrayList <String> inLabels, ArrayList <String> outLabels)
	{
		try {
			setInputs(input, inputInfo);
			setInputDimensions(numRows, num_rows_per_block, numCols, num_cols_per_block);
			setOutputs(output, outputInfo, resultIndex);
			setOutputDimensions(resultDimsUnknown);
			
			setMapperInstructions(mapperInstructions);
			setShuffleInstructions(shuffleInstructions);
			// TODO: check if aggInstructions are applicable to MMRJ
			setAggregateInstructionsInReducer(aggInstructions);
			setOtherInstructionsInReducer(otherInstructions);
			
			setInputOutputLabels(inLabels, outLabels);
			
			setNumberOfReducers(numReducers);
			setReplication(replication);
			
		} catch (DMLRuntimeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
/*		iv_inputs = input;
		iv_inputInfos = inputInfo;
		iv_instructionsInMapper = mapperInstructions;
		iv_shuffleInstructions = aggBinInstructions;
		iv_otherInstructions = otherInstructions;
		iv_outputs = output;
		iv_outputInfos = outputInfo;
		iv_resultIndices = resultIndex;
		iv_resultDimsUnknown = resultDimsUnknown;
		iv_numReducers = numReducers;
		iv_replication = replication;
		iv_rows = numRows;
		iv_cols = numCols;
		iv_num_rows_per_block = num_rows_per_block;
		iv_num_cols_per_block = num_cols_per_block;
		inputLabels = inLabels;
		outputLabels = outLabels;
		//outputLabelValues = outputLabelValueMapping;
*/	}
	
	// SortKeys Job does not have any instructions either in mapper or in reducer.
	// It just has two inputs
	public void setSORTKEYSInstructions(String [] input, InputInfo [] inputInfo, long [] numRows, long [] numCols, int [] num_rows_per_block, int [] num_cols_per_block,  
			String mapperInstructions, String shuffleInstructions, String [] output, OutputInfo [] outputInfo, byte [] resultIndex, byte[] resultDimsUnknown, 
			int numReducers, int replication, ArrayList <String> inLabels, ArrayList <String> outLabels)
	{
		try {
			setInputs(input, inputInfo);
			setInputDimensions(numRows, num_rows_per_block, numCols, num_cols_per_block);
			setOutputs(output, outputInfo, resultIndex);
			setOutputDimensions(resultDimsUnknown);
			
			setMapperInstructions(mapperInstructions);
			setShuffleInstructions(shuffleInstructions);
			setAggregateInstructionsInReducer("");
			setOtherInstructionsInReducer("");
			
			setInputOutputLabels(inLabels, outLabels);
			
			setNumberOfReducers(numReducers);
			setReplication(replication);
			
		} catch (DMLRuntimeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
/*		iv_inputs = input;
		iv_inputInfos = inputInfo;
		iv_outputs = output;
		iv_outputInfos = outputInfo;
		iv_resultIndices = resultIndex;
		iv_resultDimsUnknown = resultDimsUnknown;
		iv_numReducers = numReducers;
		iv_replication = replication;
		iv_rows = numRows;
		iv_cols = numCols;
		iv_num_rows_per_block = num_rows_per_block;
		iv_num_cols_per_block = num_cols_per_block;
		iv_shuffleInstructions = aggBinInstructions;
		iv_instructionsInMapper = mapperInstructions;
		inputLabels = inLabels;
		outputLabels = outLabels;
*/	}

	public void setCombineInstructions(String [] input, InputInfo [] inputInfo, long [] numRows, long [] numCols, int [] num_rows_per_block, int [] num_cols_per_block, 
			String shuffleInstructions, String [] output, OutputInfo [] outputInfo, byte [] resultIndex, byte[] resultDimsUnknown, 
			int numReducers, int replication, ArrayList <String> inLabels, ArrayList <String> outLabels)
	{
		try {
			setInputs(input, inputInfo);
			setInputDimensions(numRows, num_rows_per_block, numCols, num_cols_per_block);
			setOutputs(output, outputInfo, resultIndex);
			setOutputDimensions(resultDimsUnknown);
			
			setMapperInstructions("");
			setShuffleInstructions(shuffleInstructions);
			setAggregateInstructionsInReducer("");
			setOtherInstructionsInReducer("");
			
			setInputOutputLabels(inLabels, outLabels);
			
			setNumberOfReducers(numReducers);
			setReplication(replication);
			
		} catch (DMLRuntimeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
/*		iv_inputs = input;
		iv_inputInfos = inputInfo;
		iv_instructionsInMapper = mapperInstructions;
		iv_aggInstructions = aggInstructions;
		iv_otherInstructions = otherInstructions;
		iv_outputs = output;
		iv_outputInfos = outputInfo;
		iv_resultIndices = resultIndex;
		iv_resultDimsUnknown = resultDimsUnknown;
		iv_numReducers = numReducers;
		iv_replication = replication;
		iv_rows = numRows;
		iv_cols = numCols;
		iv_num_rows_per_block = num_rows_per_block;
		iv_num_cols_per_block = num_cols_per_block;
		inputLabels = inLabels;
		outputLabels = outLabels;
		//outputLabelValues = outputLabelValueMapping;
*/	}	
	
	public void setCentralMomentInstructions(String [] input, InputInfo [] inputInfo, long [] numRows, long [] numCols, int [] num_rows_per_block, int [] num_cols_per_block, 
			String mapperInstructions, String shuffleInstructions, String [] output, OutputInfo [] outputInfo, byte [] resultIndex, byte[] resultDimsUnknown, 
			int numReducers, int replication, ArrayList <String> inLabels, ArrayList <String> outLabels)
	{
		try {
			setInputs(input, inputInfo);
			setInputDimensions(numRows, num_rows_per_block, numCols, num_cols_per_block);
			setOutputs(output, outputInfo, resultIndex);
			setOutputDimensions(resultDimsUnknown);
			
			setMapperInstructions(mapperInstructions);
			setShuffleInstructions(shuffleInstructions);
			setAggregateInstructionsInReducer("");
			setOtherInstructionsInReducer("");
			
			setInputOutputLabels(inLabels, outLabels);
			
			setNumberOfReducers(numReducers);
			setReplication(replication);
			
		} catch (DMLRuntimeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
/*		iv_inputs = input;
		iv_inputInfos = inputInfo;
		iv_instructionsInMapper = mapperInstructions;
		iv_aggInstructions = aggInstructions;
		iv_outputs = output;
		iv_outputInfos = outputInfo;
		iv_resultIndices = resultIndex;
		iv_resultDimsUnknown = resultDimsUnknown;
		iv_numReducers = numReducers;
		iv_replication = replication;
		iv_rows = numRows;
		iv_cols = numCols;
		iv_num_rows_per_block = num_rows_per_block;
		iv_num_cols_per_block = num_cols_per_block;
		inputLabels = inLabels;
		outputLabels = outLabels;
		//outputLabelValues = outputLabelValueMapping;
*/	}	
	
	public void setGroupedAggInstructions(String [] input, InputInfo [] inputInfo, long [] numRows, long [] numCols, int [] num_rows_per_block, int [] num_cols_per_block, 
			String shuffleInstructions, String otherInstructions, String [] output, OutputInfo [] outputInfo, byte [] resultIndex, byte[] resultDimsUnknown, 
			int numReducers, int replication, ArrayList <String> inLabels, ArrayList <String> outLabels)
	{
		try {
			setInputs(input, inputInfo);
			setInputDimensions(numRows, num_rows_per_block, numCols, num_cols_per_block);
			setOutputs(output, outputInfo, resultIndex);
			setOutputDimensions(resultDimsUnknown);
			
			setMapperInstructions("");
			setShuffleInstructions(shuffleInstructions);
			setAggregateInstructionsInReducer("");
			setOtherInstructionsInReducer(otherInstructions);
			
			setInputOutputLabels(inLabels, outLabels);
			
			setNumberOfReducers(numReducers);
			setReplication(replication);
			
		} catch (DMLRuntimeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
/*
		iv_inputs = input;
		iv_inputInfos = inputInfo;
		iv_instructionsInMapper = mapperInstructions;
		iv_aggInstructions = aggInstructions;
		iv_otherInstructions = otherInstructions;
		iv_outputs = output;
		iv_outputInfos = outputInfo;
		iv_resultIndices = resultIndex;
		iv_resultDimsUnknown = resultDimsUnknown;
		iv_numReducers = numReducers;
		iv_replication = replication;
		iv_rows = numRows;
		iv_cols = numCols;
		iv_num_rows_per_block = num_rows_per_block;
		iv_num_cols_per_block = num_cols_per_block;
		inputLabels = inLabels;
		outputLabels = outLabels;
		//outputLabelValues = outputLabelValueMapping;
*/	}	
	
	public void setReBlockInstructions(String [] input, InputInfo [] inputInfo, long [] numRows, long [] numCols, int [] num_rows_per_block, int [] num_cols_per_block, String mapperInstructions, 
			String reblockInstructions, String otherInstructions, String [] output, OutputInfo [] outputInfo, byte [] resultIndex, byte[] resultDimsUnknown, 
			int numReducers, int replication, ArrayList <String> inLabels, ArrayList <String> outLabels)
	{
		try {
			setInputs(input, inputInfo);
			setInputDimensions(numRows, num_rows_per_block, numCols, num_cols_per_block);
			setOutputs(output, outputInfo, resultIndex);
			setOutputDimensions(resultDimsUnknown);
			
			setMapperInstructions(mapperInstructions);
			setShuffleInstructions(reblockInstructions);
			setAggregateInstructionsInReducer("");
			setOtherInstructionsInReducer(otherInstructions);
			
			setInputOutputLabels(inLabels, outLabels);
			
			setNumberOfReducers(numReducers);
			setReplication(replication);
			
		} catch (DMLRuntimeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	
	public <E>String getString(E [] arr)
	{
		String s = "";
		for(int i = 0; i < arr.length; i++)
			s = s + "," + arr[i];
		
		return s;
	}
	
	
	
	public String getString(byte [] arr)
	{
		String s = "";
		for(int i = 0; i < arr.length; i++)
			s = s + "," + Byte.toString(arr[i]);
		
		return s;
	}
	
	public String getString(long [] arr)
	{
		String s = "";
		for(int i = 0; i < arr.length; i++)
			s = s + "," + Long.toString(arr[i]);
		
		return s;
	}
	
	public String getString(int [] arr)
	{
		String s = "";
		for(int i = 0; i < arr.length; i++)
			s = s + "," + Integer.toString(arr[i]);
		
		return s;
	}
	
	public String getString(OutputInfo[] iv_outputs) {
		String s = "" ;
		for(int i = 0 ; i < iv_outputs.length; i++) {
			if(iv_outputs[i] == OutputInfo.BinaryBlockOutputInfo)
				s = s + ", " + "BinaryBlockOutputInfo" ;
			else if(iv_outputs[i] == OutputInfo.BinaryCellOutputInfo)
				s = s + ", " + "BinaryCellOutputInfo" ;
			else if(iv_outputs[i] == OutputInfo.TextCellOutputInfo)
				s = s + ", " + "TextCellOutputInfo" ;
			else {
				s = s + ", (" + iv_outputs[i].outputFormatClass + "," + iv_outputs[i].outputKeyClass + "," + iv_outputs[i].outputValueClass + ")";
			}
		}
		return s;
	}
	
	public String getString(InputInfo[] iv_inputs) {
		String s = "" ;
		for(int i = 0 ; i < iv_inputs.length; i++) {
			if(iv_inputs[i] == InputInfo.BinaryBlockInputInfo)
				s = s + ", " + "BinaryBlockInputInfo" ;
			else if(iv_inputs[i] == InputInfo.BinaryCellInputInfo)
				s = s + ", " + "BinaryCellInputInfo" ;
			else if(iv_inputs[i] == InputInfo.TextCellInputInfo)
				s = s + ", " + "TextCellInputInfo" ;
			else {
				s = s + ", (" + iv_inputs[i].inputFormatClass + "," + iv_inputs[i].inputKeyClass + "," + iv_inputs[i].inputValueClass + ")";
			}
		}
		return s;
	}
	
	
	public String toString()
	{
		String instruction = "";
		instruction += "jobtype" + jobType + " \n";
		if ( iv_inputs != null )
			instruction += "inputs " + getString(iv_inputs) + " \n";
		if ( iv_inputInfos != null )
			instruction += "input info " + getString(iv_inputInfos) + " \n";
		instruction += "recReader inst " + iv_recordReaderInstructions + " \n";
		instruction += "rand inst " + iv_randInstructions + " \n";
		instruction += "mapper inst " + iv_instructionsInMapper + " \n";
		instruction += "shuffle inst " + iv_shuffleInstructions + " \n";
		instruction += "agg inst " + iv_aggInstructions + " \n";
		//instruction += "reblock inst " + iv_reblockInstructions + " \n";
		instruction += "other inst " + iv_otherInstructions + " \n";
		instruction += "outputs  " + getString(iv_outputs) + " \n";
		instruction += "output info " + getString(iv_outputInfos) + " \n";
		instruction += "result indices " + getString(iv_resultIndices) + " \n";
		instruction += "num reducers " + iv_numReducers + " \n";
		instruction += "replication " + iv_replication + " \n";
		instruction += "result dims unknown " + getString(iv_resultDimsUnknown) + " \n";
		instruction += "num rows " + getString(iv_rows) + " \n";
		instruction += "num cols " + getString(iv_cols) + " \n";
		instruction += "rows per block " + getString(iv_num_rows_per_block) + " \n";
		instruction += "cols per block " + getString(iv_num_cols_per_block) + " \n";
		instruction += "input labels " + inputLabels + "\n";
		instruction += "outputs labels " + outputLabels + "\n";
		//instruction += "output label values " + outputLabelValues +  " " + outputLabelValues.keySet().size() + " " + outputLabelValues.values().size() + "\n";
		return instruction;
	}
	
	public void printMe() {
		System.out.println("MRInstructions: " + this.toString());
	}

	private String getOps(String inst) {
		String s = new String("");
		for ( String i : inst.split(Lops.INSTRUCTION_DELIMITOR)) {
			s += "," + (i.split(Lops.OPERAND_DELIMITOR))[0];
		}
		return s;
	}
	
	@Override
	public String getGraphString() {
		String s = new String("");
		
		s += jobType;
		if (!iv_instructionsInMapper.equals("")) {
			s += ",map("+ getOps(iv_instructionsInMapper) + ")";
		}
		if (!iv_shuffleInstructions.equals("")) {
			s += ",shuffle("+ getOps(iv_shuffleInstructions) + ")";
		}
		if (!iv_aggInstructions.equals("")) {
			s += ",agg("+ getOps(iv_aggInstructions) + ")";
		}
		if (!iv_otherInstructions.equals("")) {
			s += ",other("+ getOps(iv_otherInstructions) + ")";
		}
		return s;
		
	}

	@Override
	public byte[] getAllIndexes() throws DMLRuntimeException {
		throw new DMLRuntimeException("getAllIndexes(): Invalid method invokation for MRJobInstructions class.");
	}

	@Override
	public byte[] getInputIndexes() throws DMLRuntimeException {
		throw new DMLRuntimeException("getAllIndexes(): Invalid method invokation for MRJobInstructions class.");
	}
}
