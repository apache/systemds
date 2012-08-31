package com.ibm.bi.dml.runtime.instructions;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Arrays;

import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.meta.PartitionParams;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.utils.DMLRuntimeException;


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
	
	public String iv_randInstructions = "";
	String iv_recordReaderInstructions = "";
	String iv_instructionsInMapper = ""; 
	String iv_shuffleInstructions = ""; 
	String iv_aggInstructions = "";
	String iv_otherInstructions = "";
	
	String[] inputVars;
	String[] outputVars;
	byte [] iv_resultIndices;
	
	int iv_numReducers;
	int iv_replication;
	public String dimsUnknownFilePrefix;
	
	public JobType getJobType()
	{
		return jobType;
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

	public String getDimsUnknownFilePrefix() {
		return dimsUnknownFilePrefix;
	}
	public void setDimsUnknownFilePrefix(String prefix) {
		dimsUnknownFilePrefix = prefix;
	}
	
	public String[] getInputVars()
	{
		return inputVars;
	}


	public String[] getOutputVars()
	{
		return outputVars;
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
	 * (deep) Copy constructor, primarily used in parfor.
	 * Additionally, replace all occurrences of <code>srcPattern</code> with <code>targetPattern</code>
	 * @throws IllegalAccessException 
	 * @throws IllegalArgumentException 
	 * 
	 */
	public MRJobInstruction(MRJobInstruction that, String srcPattern, String targetPattern) throws IllegalArgumentException, IllegalAccessException {
		this(that.getJobType());
		Class<MRJobInstruction> cla = MRJobInstruction.class;
		
		Field[] fields = cla.getDeclaredFields();
		for( Field f : fields )
		{
			f.setAccessible(true);
			if(!Modifier.isStatic(f.getModifiers()))
				f.set(this, f.get(that));
		}
		this.dimsUnknownFilePrefix.replaceAll(srcPattern, targetPattern);
	}
	
	/**
	 * Method to set outputs (output indices) for a MapReduce instruction.
	 * 
	 * @param outputIndices
	 * @throws DMLRuntimeException
	 */
	public void setOutputs(byte[] outputIndices) {
		iv_resultIndices = outputIndices;
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
	public void setInputOutputLabels(String[] inputLabels, String[] outputLabels) {
		this.inputVars = inputLabels;
		this.outputVars = outputLabels;
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
	
	/*public void setPartitionInstructions(String inputs[], InputInfo[] inputInfo, String[] outputs,  int numReducers, int replication,
			long[] nr, long[] nc, int[] bnr, int[] bnc, byte[] resultIndexes, byte[] resultDimsUnknown, PartitionParams pp, ArrayList <String> inLabels, 
			ArrayList <String> outLabels, LocalVariableMap outputLabelValueMapping) {
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
	}*/

	public void setGMRInstructions(String[] inLabels,  
			String recordReaderInstructions, String mapperInstructions, 
			String aggInstructions, String otherInstructions, String [] outLabels, byte [] resultIndex,  
			int numReducers, int replication)
	{
		setOutputs(resultIndex);

		setRecordReaderInstructions(recordReaderInstructions);
		setMapperInstructions(mapperInstructions);
		setShuffleInstructions("");
		setAggregateInstructionsInReducer(aggInstructions);
		setOtherInstructionsInReducer(otherInstructions);

		setInputOutputLabels(inLabels, outLabels);

		setNumberOfReducers(numReducers);
		setReplication(replication);
	}	

	public void setRandInstructions(long [] numRows, String[] inLabels,  
			String randInstructions, String mapperInstructions, 
			String aggInstructions, String otherInstructions, String [] outLabels, byte [] resultIndex, 
			int numReducers, int replication)
	{
		setOutputs(resultIndex);
			
		setRecordReaderInstructions("");
		setRandInstructions(randInstructions);
		setMapperInstructions(mapperInstructions);
		setShuffleInstructions("");
		setAggregateInstructionsInReducer(aggInstructions);
		setOtherInstructionsInReducer(otherInstructions);
			
		setInputOutputLabels(inLabels, outLabels);
			
		setNumberOfReducers(numReducers);
		setReplication(replication);
	}
	

	public void setMMCJInstructions(String[] inLabels, 
			String mapperInstructions, String shuffleInstructions, 
			String [] outLabels, byte [] resultIndex,  
			int numReducers, int replication)
	{
		setOutputs(resultIndex);

		setMapperInstructions(mapperInstructions);
		setShuffleInstructions(shuffleInstructions);
		setAggregateInstructionsInReducer("");
		setOtherInstructionsInReducer("");

		setInputOutputLabels(inLabels, outLabels);

		setNumberOfReducers(numReducers);
		setReplication(replication);
	}

	public void setMMRJInstructions(String[] inLabels, 
			String mapperInstructions, String shuffleInstructions, String aggInstructions, String otherInstructions, 
			String [] outLabels, byte [] resultIndex,  
			int numReducers, int replication)
	{
		setOutputs(resultIndex);

		setMapperInstructions(mapperInstructions);
		setShuffleInstructions(shuffleInstructions);
		setAggregateInstructionsInReducer(aggInstructions);
		setOtherInstructionsInReducer(otherInstructions);

		setInputOutputLabels(inLabels, outLabels);

		setNumberOfReducers(numReducers);
		setReplication(replication);
	}
	
	// SortKeys Job does not have any instructions either in mapper or in reducer.
	// It just has two inputs
	public void setSORTKEYSInstructions(String [] inLabels,   
			String mapperInstructions, String shuffleInstructions, 
			String[] outLabels, byte [] resultIndex,  
			int numReducers, int replication)
	{
		setOutputs(resultIndex);

		setMapperInstructions(mapperInstructions);
		setShuffleInstructions(shuffleInstructions);
		setAggregateInstructionsInReducer("");
		setOtherInstructionsInReducer("");

		setInputOutputLabels(inLabels, outLabels);

		setNumberOfReducers(numReducers);
		setReplication(replication);
	}

	public void setCombineInstructions(String[] inLabels,  
			String shuffleInstructions, String[] outLabels, byte[] resultIndex,  
			int numReducers, int replication)
	{
		setOutputs(resultIndex);

		setMapperInstructions("");
		setShuffleInstructions(shuffleInstructions);
		setAggregateInstructionsInReducer("");
		setOtherInstructionsInReducer("");

		setInputOutputLabels(inLabels, outLabels);

		setNumberOfReducers(numReducers);
		setReplication(replication);
	}	
	
	public void setCentralMomentInstructions(String[] inLabels, 
			String mapperInstructions, String shuffleInstructions, 
			String[] outLabels, byte [] resultIndex,  
			int numReducers, int replication)
	{
		setOutputs(resultIndex);

		setMapperInstructions(mapperInstructions);
		setShuffleInstructions(shuffleInstructions);
		setAggregateInstructionsInReducer("");
		setOtherInstructionsInReducer("");

		setInputOutputLabels(inLabels, outLabels);

		setNumberOfReducers(numReducers);
		setReplication(replication);
	}	
	
	public void setGroupedAggInstructions(String[] inLabels, 
			String shuffleInstructions, String otherInstructions, 
			String[] outLabels, byte [] resultIndex,  
			int numReducers, int replication)
	{
		setOutputs(resultIndex);

		setMapperInstructions("");
		setShuffleInstructions(shuffleInstructions);
		setAggregateInstructionsInReducer("");
		setOtherInstructionsInReducer(otherInstructions);

		setInputOutputLabels(inLabels, outLabels);

		setNumberOfReducers(numReducers);
		setReplication(replication);
	}	
	
	public void setReBlockInstructions(String[] inLabels, 
			String mapperInstructions, String reblockInstructions, String otherInstructions, 
			String[] outLabels, byte [] resultIndex,  
			int numReducers, int replication)
	{
		setOutputs(resultIndex);

		setMapperInstructions(mapperInstructions);
		setShuffleInstructions(reblockInstructions);
		setAggregateInstructionsInReducer("");
		setOtherInstructionsInReducer(otherInstructions);

		setInputOutputLabels(inLabels, outLabels);

		setNumberOfReducers(numReducers);
		setReplication(replication);
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
		instruction += "input labels " + Arrays.toString(inputVars) + " \n";
		instruction += "recReader inst " + iv_recordReaderInstructions + " \n";
		instruction += "rand inst " + iv_randInstructions + " \n";
		instruction += "mapper inst " + iv_instructionsInMapper + " \n";
		instruction += "shuffle inst " + iv_shuffleInstructions + " \n";
		instruction += "agg inst " + iv_aggInstructions + " \n";
		instruction += "other inst " + iv_otherInstructions + " \n";
		instruction += "output labels " + Arrays.toString(outputVars) + " \n";
		instruction += "result indices " + getString(iv_resultIndices) + " \n";
		//instruction += "result dims unknown " + getString(iv_resultDimsUnknown) + " \n";
		instruction += "num reducers " + iv_numReducers + " \n";
		instruction += "replication " + iv_replication + " \n";
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
