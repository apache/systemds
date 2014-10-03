/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.meta.PartitionParams;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.NumItemsByEachReducerMetaData;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//public enum JobType {MMCJ, MMRJ, GMR, Partition, RAND, ReBlock, SortKeys, Combine, CMCOV, GroupedAgg}; 
	private JobType jobType;
	
	private String _randInstructions = "";
	private String _recordReaderInstructions = "";
	private String _mapperInstructions = ""; 
	private String _shuffleInstructions = ""; 
	private String _aggInstructions = "";
	private String _otherInstructions = "";
	
	private String[] inputVars;
	private String[] outputVars;
	private byte [] _resultIndices;
	
	private int iv_numReducers;
	private int iv_replication;
	private String dimsUnknownFilePrefix;

	private double _mapperMem = -1;
	
	/**
	 * This structure contains the DML script line number
	 * of each MR instructions within this MR job
	 */
	private ArrayList<Integer> MRJobInstructionsLineNumbers;

	
	/*
	 * Following attributes are populated by pulling out information from Symbol Table.
	 * This is done just before a job is submitted/spawned.
	 */
	private String[] inputs;
	private InputInfo[] inputInfos;
	private long[] rlens;
	private long[] clens;
	private int[] brlens;
	private int[] bclens;
	private String[] outputs;
	private OutputInfo[] outputInfos;

	// Member variables to store partitioning-related information for all input matrices 
	private boolean[] partitioned;
	private PDataPartitionFormat[] pformats;
	private int[] psizes;
	
	/*
	 * These members store references to MatrixObjects corresponding to different 
	 * MATRIX variables in inputVars and outputVars, respectively. Note that the 
	 * references to SCALAR input variables are not stored in <code>inputMatrices</code>.
	 * Every reference in <code>outputMatrices</code> is always points to MATRIX 
	 * since MR jobs always produces matrices. 
	 */
	private MatrixObject[] inputMatrices, outputMatrices;

	// Indicates the data type of inputVars
	private DataType[] inputDataTypes;


	// Used for partitioning jobs..
	private PartitionParams partitionParams;
	
	
	
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
	public MRJobInstruction(MRJobInstruction that) 
		throws IllegalArgumentException, IllegalAccessException 
	{
		this(that.getJobType());
		Class<MRJobInstruction> cla = MRJobInstruction.class;
		
		//copy fields
		Field[] fields = cla.getDeclaredFields();
		for( Field f : fields )
		{
			f.setAccessible(true);
			if(!Modifier.isStatic(f.getModifiers()))
				f.set(this, f.get(that));
		}
		
		//replace inputs/outputs (arrays)
		inputVars = that.inputVars.clone();
		outputVars = that.outputVars.clone();
		_resultIndices = that._resultIndices.clone();
	}	
	
	
	
	public JobType getJobType()
	{
		return jobType;
	}

	public String getIv_instructionsInMapper()
	{
		return _mapperInstructions;
	}
	
	public void setIv_instructionsInMapper(String inst)
	{
		_mapperInstructions = inst;
	}
	
	public String getIv_recordReaderInstructions()
	{
		return _recordReaderInstructions;
	}
	
	public void setIv_recordReaderInstructions(String inst)
	{
		_recordReaderInstructions = inst;
	}
	
	public String getIv_randInstructions() 
	{
		return _randInstructions;
	}
	
	public void setIv_randInstructions(String inst) 
	{
		_randInstructions = inst;
	}

	public String getIv_shuffleInstructions()
	{
		return _shuffleInstructions;
	}
	
	public void setIv_shuffleInstructions(String inst)
	{
		_shuffleInstructions = inst;
	}

	public String getIv_aggInstructions()
	{
		return _aggInstructions;
	}
	
	public void setIv_aggInstructions(String inst)
	{
		_aggInstructions = inst;
	}

	public String getIv_otherInstructions()
	{
		return _otherInstructions;
	}
	
	public void setIv_otherInstructions(String inst)
	{
		_otherInstructions = inst;
	}

	public byte[] getIv_resultIndices()
	{
		return _resultIndices;
	}

	public int getIv_numReducers()
	{
		return iv_numReducers;
	}

	public int getIv_replication()
	{
		return iv_replication;
	}

	public double getMemoryRequirements(){
		return _mapperMem;
	}

	public void setMemoryRequirements(double mem) {
		_mapperMem = mem;
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

	/**
	 * Getter for MRJobInstructionslineNumbers
	 * @return TreeMap containing all instructions indexed by line number   
	 */
	public ArrayList<Integer> getMRJobInstructionsLineNumbers()
	{
		return MRJobInstructionsLineNumbers;
	}
	
	public PartitionParams getPartitionParams() {
		return partitionParams ;
	}
	
	/**
	 * Method to set outputs (output indices) for a MapReduce instruction.
	 * 
	 * @param outputIndices
	 * @throws DMLRuntimeException
	 */
	public void setOutputs(byte[] outputIndices) {
		_resultIndices = outputIndices;
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
		_recordReaderInstructions = rrInstructions;
	}
	
	public void setMapperInstructions(String mapperInstructions) {
		_mapperInstructions = mapperInstructions;
	}
	
	public void setShuffleInstructions(String shuffleInstructions) {
		_shuffleInstructions = shuffleInstructions;
	}
	
	public void setAggregateInstructionsInReducer(String aggInstructions) {
		_aggInstructions = aggInstructions;
	}
	
	public void setOtherInstructionsInReducer(String otherInstructions) {
		_otherInstructions = otherInstructions;
	}
	
	public void setRandInstructions(String randInstructions) {
		_randInstructions = randInstructions;
	}
	
	/**
	 * Setter for MRJobInstructionslineNumbers field
	 * @param MRJobLineNumbers Line numbers for each instruction in this MRJob  
	 */
	public void setMRJobInstructionsLineNumbers(ArrayList<Integer> MRJobLineNumbers) {
		MRJobInstructionsLineNumbers = MRJobLineNumbers;
	}
	
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
	
	/**
	 * Search whether or not this MR job contains at least one 
	 * MR instruction with specified line number parameter 
	 * @param lineNum Line number in DML script
	 * @return Return true if found, otherwise return false 
	 */
	public boolean findMRInstructions(int lineNum) {
		if (!DMLScript.ENABLE_DEBUG_MODE) {
			System.err.println("Error: Expecting debug mode to be enabled for this functionality");
			return false;
		}
		for (Integer lineNumber : MRJobInstructionsLineNumbers) {
			if (lineNum == lineNumber)
				return true;
		}
		return false;
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
		instruction += "jobtype = " + jobType + " \n";
		instruction += "input labels = " + Arrays.toString(inputVars) + " \n";
		instruction += "recReader inst = " + _recordReaderInstructions + " \n";
		instruction += "rand inst = " + _randInstructions + " \n";
		instruction += "mapper inst = " + _mapperInstructions + " \n";
		instruction += "shuffle inst = " + _shuffleInstructions + " \n";
		instruction += "agg inst = " + _aggInstructions + " \n";
		instruction += "other inst = " + _otherInstructions + " \n";
		instruction += "output labels = " + Arrays.toString(outputVars) + " \n";
		instruction += "result indices = " + getString(_resultIndices) + " \n";
		//instruction += "result dims unknown " + getString(iv_resultDimsUnknown) + " \n";
		instruction += "num reducers = " + iv_numReducers + " \n";
		instruction += "replication = " + iv_replication + " \n";
		return instruction;
	}
	
	/**
	 * Method for displaying MR instructions interspersed with source code 
	 * ONLY USED IN DEBUG MODE
	 * @param debug Flag for displaying instructions in debugger test integration
	 * @return
	 */
	public String getMRString(boolean debug)
	{
		if (!DMLScript.ENABLE_DEBUG_MODE) {
			System.err.println("Error: Expecting debug mode to be enabled for this functionality");
			return "";
		}
		
		String instruction = "MR-Job[\n";
		instruction += "\t\t\t\tjobtype        = " + jobType + " \n";
		
		if (!debug)
			instruction += "\t\t\t\tinput labels   = " + Arrays.toString(inputVars) + " \n";
		
		if (_recordReaderInstructions.length() > 0) {
			String [] instArray = _recordReaderInstructions.split(Lop.INSTRUCTION_DELIMITOR);
			if (!debug)
				instruction += "\t\t\t\trecReader inst = " + instArray[0] + " \n";
			else {
				String [] instStr = prepareInstruction(instArray[0]).split(" ");
				instruction += "\t\t\t\trecReader inst = " + instStr[0] + " " + instStr[1] + " \n";
			}
			for (int i = 1; i < instArray.length ; i++) {
				if (!debug)
					instruction += "\t\t\t\t                 " + instArray[i] + " \n";
				else {
					String [] instStr = prepareInstruction(instArray[i]).split(" ");
					instruction += "\t\t\t\t                 " + instStr[0] + " " + instStr[1] + " \n";
				}			
			}
		}
		if (_randInstructions.length() > 0) {			
			String [] instArray = _randInstructions.split(Lop.INSTRUCTION_DELIMITOR);
			if (!debug)
				instruction += "\t\t\t\trand inst      = " + instArray[0] + " \n";
			else {
				String [] instStr = prepareInstruction(instArray[0]).split(" ");
				instruction += "\t\t\t\trand inst      = " + instStr[0] + " " + instStr[1] + " \n";
			}
			for (int i = 1; i < instArray.length ; i++) {
				if (!debug)
					instruction += "\t\t\t\t                 " + instArray[i] + " \n";
				else {
					String [] instStr = prepareInstruction(instArray[i]).split(" ");
					instruction += "\t\t\t\t                 " + instStr[0] + " " + instStr[1] + " \n";
				}
			}
		}
		if (_mapperInstructions.length() > 0) {
			String [] instArray = _mapperInstructions.split(Lop.INSTRUCTION_DELIMITOR);
			if (!debug)
				instruction += "\t\t\t\tmapper inst    = " + instArray[0] + " \n";
			else {
				String [] instStr = prepareInstruction(instArray[0]).split(" ");
				instruction += "\t\t\t\tmapper inst    = " + instStr[0] + " " + instStr[1] + " \n";
			}
			for (int i = 1; i < instArray.length ; i++) {
				if (!debug)
					instruction += "\t\t\t\t                 " + instArray[i] + " \n";
				else {
					String [] instStr = prepareInstruction(instArray[i]).split(" ");
					instruction += "\t\t\t\t                 " + instStr[0] + " " + instStr[1] + " \n";
				}
			}
		}
		if (_shuffleInstructions.length() > 0) {
			String [] instArray = _shuffleInstructions.split(Lop.INSTRUCTION_DELIMITOR);
			if (!debug)
				instruction += "\t\t\t\tshuffle inst   = " + instArray[0] + " \n";
			else {
				String [] instStr = prepareInstruction(instArray[0]).split(" ");
				instruction += "\t\t\t\tshuffle inst   = " + instStr[0] + " " + instStr[1] + " \n";
			}
			for (int i = 1; i < instArray.length ; i++) {
				if (!debug)
					instruction += "\t\t\t\t                 " + instArray[i] + " \n";
				else {
					String [] instStr = prepareInstruction(instArray[i]).split(" ");
					instruction += "\t\t\t\t                 " + instStr[0] + " " + instStr[1] + " \n";
				}
			}
		}
		if (_aggInstructions.length() > 0) {			
			String [] instArray = _aggInstructions.split(Lop.INSTRUCTION_DELIMITOR);
			if (!debug)
				instruction += "\t\t\t\tagg inst       = " + instArray[0] + " \n";
			else {
				String [] instStr = prepareInstruction(instArray[0]).split(" ");
				instruction += "\t\t\t\tagg inst       = " + instStr[0] + " " + instStr[1] + " \n";
			}
			for (int i = 1; i < instArray.length ; i++) {
				if (!debug)
					instruction += "\t\t\t\t                 " + instArray[i] + " \n";
				else {
					String [] instStr = prepareInstruction(instArray[i]).split(" ");
					instruction += "\t\t\t\t                 " + instStr[0] + " " + instStr[1] + " \n";
				}
			}
		}
		if (_otherInstructions.length() > 0) {
			String [] instArray = _otherInstructions.split(Lop.INSTRUCTION_DELIMITOR);
			if (!debug)
				instruction += "\t\t\t\tother inst     = " + instArray[0] + " \n";
			else {
				String [] instStr = prepareInstruction(instArray[0]).split(" ");
				instruction += "\t\t\t\tother inst     = " + instStr[0] + " " + instStr[1] + " \n";
			}
			for (int i = 1; i < instArray.length ; i++) {
				if (!debug)
					instruction += "\t\t\t\t                 " + instArray[i] + " \n";
				else {
					String [] instStr = prepareInstruction(instArray[i]).split(" ");
					instruction += "\t\t\t\t                 " + instStr[0] + " " + instStr[1] + " \n";
				}
			}			
		}
		if (!debug)
			instruction += "\t\t\t\toutput labels  = " + Arrays.toString(outputVars)  + " \n";
		instruction += "\t\t\t        ]";

		return instruction;
	}
	
	public void printMe() {
		LOG.debug("\nMRInstructions: \n" + this.toString());
	}

	private String getOps(String inst) {
		String s = new String("");
		for ( String i : inst.split(Lop.INSTRUCTION_DELIMITOR)) {
			s += "," + (i.split(Lop.OPERAND_DELIMITOR))[0];
		}
		return s;
	}
	
	@Override
	public String getGraphString() {
		String s = new String("");
		
		s += jobType;
		if (!_mapperInstructions.equals("")) {
			s += ",map("+ getOps(_mapperInstructions) + ")";
		}
		if (!_shuffleInstructions.equals("")) {
			s += ",shuffle("+ getOps(_shuffleInstructions) + ")";
		}
		if (!_aggInstructions.equals("")) {
			s += ",agg("+ getOps(_aggInstructions) + ")";
		}
		if (!_otherInstructions.equals("")) {
			s += ",other("+ getOps(_otherInstructions) + ")";
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

	public boolean isMapOnly()
	{
		return (   (_shuffleInstructions == null || _shuffleInstructions.length()==0)
				&& (_aggInstructions == null || _aggInstructions.length()==0)
				&& (_otherInstructions == null || _otherInstructions.length()==0) );
	}

	public String[] getInputs() {
		return inputs;
	}

	public InputInfo[] getInputInfos() {
		return inputInfos;
	}

	public long[] getRlens() {
		return rlens;
	}

	public long[] getClens() {
		return clens;
	}

	public int[] getBrlens() {
		return brlens;
	}

	public int[] getBclens() {
		return bclens;
	}

	public String[] getOutputs() {
		return outputs;
	}

	public OutputInfo[] getOutputInfos() {
		return outputInfos;
	}

	public MatrixObject[] getInputMatrices() {
		return inputMatrices;
	}
	
	public boolean[] getPartitioned() {
		return partitioned;
	}

	public void setPartitioned(boolean[] partitioned) {
		this.partitioned = partitioned;
	}

	public PDataPartitionFormat[] getPformats() {
		return pformats;
	}

	public void setPformats(PDataPartitionFormat[] pformats) {
		this.pformats = pformats;
	}

	public int[] getPsizes() {
		return psizes;
	}

	public void setPsizes(int[] psizes) {
		this.psizes = psizes;
	}
	
	/**
	 * Extracts input variables with MATRIX data type, and stores references to
	 * corresponding matrix objects in <code>inputMatrices</code>. Also, stores 
	 * the data types in <code>inputDataTypes</code>.
	 * 
	 * @param pb
	 */
	public MatrixObject[] extractInputMatrices(ExecutionContext ec) {
		ArrayList<MatrixObject> inputmat = new ArrayList<MatrixObject>();
		inputDataTypes = new DataType[inputVars.length];
		for ( int i=0; i < inputVars.length; i++ ) {
			Data d = ec.getVariable(inputVars[i]);
			inputDataTypes[i] = d.getDataType();
			if ( d.getDataType() == DataType.MATRIX ) {
				inputmat.add((MatrixObject) d);
			}
		}
		inputMatrices = inputmat.toArray(new MatrixObject[inputmat.size()]);
		
		// populate auxiliary data structures
		populateInputs();
		
		return inputMatrices;
	}

	public MatrixObject[] getOutputMatrices() {
		return outputMatrices;
	}

	/**
	 * Extracts MatrixObject references to output variables, all of which will be
	 * of MATRIX data type, and stores them in <code>outputMatrices</code>. Also, 
	 * populates auxiliary data structures.
	 * 
	 * @param pb
	 */
	public MatrixObject[] extractOutputMatrices(ExecutionContext ec) throws DMLRuntimeException {
		outputMatrices = new MatrixObject[getOutputVars().length];
		int ind = 0;
		for(String oo: getOutputVars()) {
			Data d = ec.getVariable(oo);
			if ( d.getDataType() == DataType.MATRIX ) {
				outputMatrices[ind++] = (MatrixObject)d;
			}
			else {
				throw new DMLRuntimeException(getJobType() + ": invalid datatype (" + d.getDataType() + ") for output variable " + oo);
			}
		}
		
		// populate auxiliary data structures
		populateOutputs();
		
		return outputMatrices;
	}

	/**
	 * Auxiliary data structures that store information required to spawn MR jobs.
	 * These data structures are populated by pulling out information from symbol
	 * table. More specifically, from information stored in <code>inputMatrices</code>
	 * and <code>outputMatrices</code>.   
	 */
	private void populateInputs() {
		
		// Since inputVars can potentially contain scalar variables,
		// auxiliary data structures of size <code>inputMatrices.length</code>
		// are allocated instead of size <code>inputVars.length</code>
		
		// Allocate space
		inputs = new String[inputMatrices.length];
		inputInfos = new InputInfo[inputMatrices.length];
		rlens = new long[inputMatrices.length];
		clens = new long[inputMatrices.length];
		brlens = new int[inputMatrices.length];
		bclens = new int[inputMatrices.length];

		partitioned = new boolean[inputMatrices.length];
		pformats = new PDataPartitionFormat[inputMatrices.length];
		psizes = new int[inputMatrices.length];

		
		// populate information
		for ( int i=0; i < inputMatrices.length; i++ ) {
			inputs[i] = inputMatrices[i].getFileName();
			MatrixCharacteristics mc = ((MatrixDimensionsMetaData) inputMatrices[i].getMetaData()).getMatrixCharacteristics();
			rlens[i] = mc.get_rows();
			clens[i] = mc.get_cols();
			brlens[i] = mc.get_rows_per_block();
			bclens[i] = mc.get_cols_per_block();
			if ( inputMatrices[i].getMetaData() instanceof MatrixFormatMetaData ) {
				inputInfos[i] = ((MatrixFormatMetaData) inputMatrices[i].getMetaData()).getInputInfo();
			}
			else if (inputMatrices[i].getMetaData() instanceof NumItemsByEachReducerMetaData ) {
				inputInfos[i] = InputInfo.InputInfoForSortOutput;
				inputInfos[i].metadata = inputMatrices[i].getMetaData();
			}
			
			partitioned[i] = inputMatrices[i].isPartitioned();
			pformats[i] = inputMatrices[i].getPartitionFormat();
			psizes[i] = inputMatrices[i].getPartitionSize();
		}
	}

	/**
	 * Pulls out information from symbol table for output variables (i.e., outputMatrices) 
	 * and populates auxiliary data structutes that are used in setting up MR jobs.
	 */
	private void populateOutputs() {
		// Note: (outputVars.length == outputMatrices.length) -> true 
		
		// Allocate space
		outputs = new String[outputVars.length];
		outputInfos = new OutputInfo[outputVars.length];
		
		// Populate information
		for(int i=0; i < outputVars.length; i++) {
			outputs[i] = outputMatrices[i].getFileName();
			MatrixFormatMetaData md = (MatrixFormatMetaData) outputMatrices[i].getMetaData();
			outputInfos[i] = md.getOutputInfo();
		}
	}
	
	/**
	 * Prepare current instruction for printing
	 * by removing internal delimiters.  
	 * @param inst Instruction to be displayed 
	 * @return Post-processed instruction in string format
	 */
	private static String prepareInstruction(String inst) {
		String tmp = inst;
		tmp = tmp.replaceAll(Lop.OPERAND_DELIMITOR, " ");
		tmp = tmp.replaceAll(Lop.DATATYPE_PREFIX, ".");
		tmp = tmp.replaceAll(Lop.INSTRUCTION_DELIMITOR, ", ");

		return tmp;
	}
	
	public void printCompleteMRJobInstruction(MatrixCharacteristics[] resultStats) throws DMLRuntimeException {
		LOG.trace("jobtype" + jobType);
		LOG.trace("Inputs: \n");
		for(int i=0, mi=0; i < inputVars.length; i++ ) {
			if(inputDataTypes[i] == DataType.SCALAR) {
				LOG.trace("    " + inputVars[i] + " - SCALAR input (replaced w/ value)");
			}
			else if ( inputDataTypes[i] == DataType.MATRIX ) {
				LOG.trace("    " + inputVars[i] + 
						" - [" + inputs[mi] + 
						"]  [" + rlens[mi] + ", " + clens[mi] + 
						"]  nnz[" + inputMatrices[mi].getNnz() +
						"]  block[" + brlens[mi] + ", " + bclens[mi] +
						"]  [" + InputInfo.inputInfoToString(inputInfos[mi]) +  
						"]");
				mi++;
			}
			else 
				LOG.trace("    " + inputVars[i] + " - " + inputDataTypes[i]);
		}
		
		LOG.trace("  Instructions:");
		if ( !_recordReaderInstructions.equals("")) 
			LOG.trace("    recReader inst - " + _recordReaderInstructions );
		if ( !_randInstructions.equals("")) 
			LOG.trace("    rand inst - " + _randInstructions );
		if ( !_mapperInstructions.equals("")) 
			LOG.trace("    mapper inst - " + _mapperInstructions );
		if ( !_shuffleInstructions.equals("")) 
			LOG.trace("    shuffle inst - " + _shuffleInstructions );
		if ( !_aggInstructions.equals("")) 
			LOG.trace("    agg inst - " + _aggInstructions );
		if ( !_otherInstructions.equals("")) 
			LOG.trace("    other inst - " + _otherInstructions );

		LOG.trace("  Outputs:");
		for(int i=0; i < outputVars.length; i++ ) {
			LOG.trace("    " + _resultIndices[i] + " : " + outputVars[i] + 
					" - [" + outputs[i] + 
					"]  [" + resultStats[i].get_rows() + ", " + resultStats[i].get_cols() + 
					"]  nnz[" + outputMatrices[i].getNnz() +
					"]  block[" + resultStats[i].get_rows() + ", " + resultStats[i].get_cols_per_block() + 
					"]  [" + OutputInfo.outputInfoToString(outputInfos[i]) +
					"]");
		}
		LOG.trace("  #Reducers - " + iv_numReducers);
		LOG.trace("  Replication - " + iv_replication);
	}
	
	@Override
	public void updateInstructionThreadID(String pattern, String replace)
	{
		this.dimsUnknownFilePrefix.replaceAll(pattern, replace);
		
		if( getJobType() == JobType.DATAGEN )
		{
			//update string representation (because parsing might fail due to pending instruction patching)
			String rndinst = getIv_randInstructions();
			String rndinst2 = "";
			if( rndinst!=null && rndinst.length()>0 )
			{
				String[] instSet = rndinst.split( Lop.INSTRUCTION_DELIMITOR );
				for( String dginst : instSet )
				{
					if( rndinst2.length()>0 )
						rndinst2 += Lop.INSTRUCTION_DELIMITOR;
					
					//handle single instruction
					String[] parts = dginst.split(Lop.OPERAND_DELIMITOR);
					int pos = -1;
					if( parts[1].equals("Rand") ) pos = 13;
					if( parts[1].equals("seq") ) pos = 11;
					if( pos>0 )
					{
						StringBuilder sb = new StringBuilder();
						for( int i=0; i<parts.length; i++ )
						{
							if( i>0 ) 
								sb.append(Lop.OPERAND_DELIMITOR);
							if( i==pos )
								sb.append(ProgramConverter.saveReplaceFilenameThreadID(parts[i], pattern, replace));
							else
								sb.append(parts[i]);
						}
						rndinst2 += sb.toString();
					}
					else
						rndinst2 += dginst;		
				}
				
				setRandInstructions(rndinst2);
			}		
		}
	}
	
	/**
	 * 
	 * @param that
	 * @return
	 */
	public boolean isMergableMRJobInstruction( MRJobInstruction that )
	{
		boolean ret = true;
		
		//check max memory requirements of mapper instructions
		if(   (_mapperMem + that._mapperMem) 
			> OptimizerUtils.getRemoteMemBudgetMap(true) )
		{
			ret = false;
		}
		
		//check max possible byte indexes (worst-case: no sharing)
		int maxIx1 = UtilFunctions.max(_resultIndices);
		int maxIx2 = UtilFunctions.max(that._resultIndices);
		if( (maxIx1+maxIx2) > Byte.MAX_VALUE )
		{
			ret = false;
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param that
	 */
	public void mergeMRJobInstruction( MRJobInstruction that )
	{
		LOG.debug("Current instruction:\n"+this.toString());
		LOG.debug("Next instruction:\n"+that.toString());
		
		//compute offsets (inputs1, inputs2, intermediates1, intermediates2, outputs1, outputs2)
		byte maxIxInst1 = UtilFunctions.max(_resultIndices);
		byte maxIxInst2 = UtilFunctions.max(that._resultIndices);
		byte sharedIx = 0;
		
		//compute input index map (based on distinct filenames)
		HashMap<String, Byte> inMap = new HashMap<String, Byte>();
		for( int i=0; i<inputs.length; i++ )
			inMap.put(inputs[i], (byte) i);
		
		//compute shared input indexes
		for( int i=0; i<that.inputs.length; i++ ) 
			if( inMap.containsKey(that.inputs[i]) )
				sharedIx++;
		
		byte lenInputs = (byte)(inputs.length + that.inputs.length - sharedIx);
		
		//compute transition index map for instruction 1
		HashMap<Byte, Byte> transMap1 = new HashMap<Byte,Byte>();
		for( int i=0; i<inputs.length; i++ )
			transMap1.put((byte)i, (byte)i);
		for( int i=inputs.length; i<=maxIxInst1; i++ ) //remap intermediates and 
		{
			transMap1.put((byte)i, (byte)(that.inputs.length-sharedIx+i));
		}
			
		//compute transition index max for instruction 2
		HashMap<Byte, Byte> transMap2 = new HashMap<Byte,Byte>();
		byte nextIX = (byte)inputs.length;
		for( int i=0; i<that.inputs.length; i++ ) {
			if( !inMap.containsKey(that.inputs[i]) )
				inMap.put(that.inputs[i], nextIX++);
			transMap2.put((byte)i, inMap.get(that.inputs[i]));
		}
		nextIX = (byte) (lenInputs + (maxIxInst1+1 - inputs.length));
		for( int i=that.inputs.length; i<=maxIxInst2; i++ )
		{
			transMap2.put((byte)i, (byte)nextIX++);
		}
		
		//construct merged inputs and meta data
		int llen = lenInputs; int len = inputs.length;
		int olen = outputs.length+that.outputs.length;
		String[] linputs = new String[llen];
		InputInfo[] linputInfos = new InputInfo[llen];
		PDataPartitionFormat[] lpformats = new PDataPartitionFormat[llen];
		long[] lrlens = new long[llen];
		long[] lclens = new long[llen];
		int[] lbrlens = new int[llen];
		int[] lbclens = new int[llen];
		String[] loutputs = new String[olen];
		OutputInfo[] loutputInfos = new OutputInfo[olen];
		byte[] lresultIndexes = new byte[olen];
		System.arraycopy(inputs, 0, linputs, 0, len);
		System.arraycopy(inputInfos, 0, linputInfos, 0, len);
		System.arraycopy(pformats, 0, lpformats, 0, len);
		System.arraycopy(rlens, 0, lrlens, 0, len);
		System.arraycopy(clens, 0, lclens, 0, len);
		System.arraycopy(brlens, 0, lbrlens, 0, len);
		System.arraycopy(bclens, 0, lbclens, 0, len);
		System.arraycopy(outputs, 0, loutputs, 0, outputs.length);
		System.arraycopy(outputInfos, 0, loutputInfos, 0, outputs.length);
		for( int i=0; i<that.inputs.length; i++ ){
			byte ixSrc = (byte) i;
			byte ixTgt = transMap2.get((byte)i);
			linputs[ixTgt] = that.inputs[ixSrc];
			linputInfos[ixTgt] = that.inputInfos[ixSrc];
			lpformats[ixTgt] = that.pformats[ixSrc];
			lrlens[ixTgt] = that.rlens[ixSrc];
			lclens[ixTgt] = that.clens[ixSrc];
			lbrlens[ixTgt] = that.brlens[ixSrc];
			lbclens[ixTgt] = that.bclens[ixSrc];
		}
		for( int i=0; i<_resultIndices.length; i++ )
			lresultIndexes[i] = transMap1.get(_resultIndices[i]);
		for( int i=0; i<that._resultIndices.length; i++ ){
			loutputs[_resultIndices.length+i] = that.outputs[i];
			loutputInfos[_resultIndices.length+i] = that.outputInfos[i];
			lresultIndexes[_resultIndices.length+i] = transMap2.get(that._resultIndices[i]);
		}
		inputs = linputs; inputInfos = linputInfos; 
		pformats = lpformats;
		outputs = loutputs; outputInfos = loutputInfos;
		rlens = lrlens; clens = lclens; brlens = lbrlens; bclens = lbclens;
		_resultIndices = lresultIndexes;
		
		//replace merged instructions with all transition map entries 
		String randInst1 = replaceInstructionStringWithTransMap(this.getIv_randInstructions(), transMap1);
		String randInst2 = replaceInstructionStringWithTransMap(that.getIv_randInstructions(), transMap2);
		String rrInst1 = replaceInstructionStringWithTransMap(this.getIv_recordReaderInstructions(), transMap1);
		String rrInst2 = replaceInstructionStringWithTransMap(that.getIv_recordReaderInstructions(), transMap2);
		String mapInst1 = replaceInstructionStringWithTransMap(this.getIv_instructionsInMapper(), transMap1);
		String mapInst2 = replaceInstructionStringWithTransMap(that.getIv_instructionsInMapper(), transMap2);
		String shuffleInst1 = replaceInstructionStringWithTransMap(this.getIv_shuffleInstructions(), transMap1);
		String shuffleInst2 = replaceInstructionStringWithTransMap(that.getIv_shuffleInstructions(), transMap2);
		String aggInst1 = replaceInstructionStringWithTransMap(this.getIv_aggInstructions(), transMap1);
		String aggInst2 = replaceInstructionStringWithTransMap(that.getIv_aggInstructions(), transMap2);
		String otherInst1 = replaceInstructionStringWithTransMap(this.getIv_otherInstructions(), transMap1);
		String otherInst2 = replaceInstructionStringWithTransMap(that.getIv_otherInstructions(), transMap2);
		
		//concatenate instructions
		setIv_randInstructions( concatenateInstructions(randInst1, randInst2) );
		setIv_recordReaderInstructions( concatenateInstructions(rrInst1, rrInst2) );
		setIv_instructionsInMapper( concatenateInstructions(mapInst1, mapInst2) );
		setIv_shuffleInstructions( concatenateInstructions(shuffleInst1, shuffleInst2) );
		setIv_aggInstructions( concatenateInstructions(aggInst1, aggInst2) );
		setIv_otherInstructions( concatenateInstructions(otherInst1, otherInst2) );
		
		//merge memory requirements
		_mapperMem = _mapperMem + that._mapperMem;
		
		LOG.debug("Merged instruction:\n"+this.toString());
	}
	
	/**
	 * Safe replacement of mr indexes based on transition map. Multiple string replacements
	 * would fail for crossing transitions: e.g., 1->2, 2->1.
	 * 
	 * @param inst
	 * @param transMap
	 * @return
	 */
	private String replaceInstructionStringWithTransMap( String inst, HashMap<Byte,Byte> transMap )
	{
		//prevent unnecessary parsing and reconstruction
		if( inst == null || inst.length()==0 || transMap.size()==0 )
			return inst;
		
		String[] pinst = inst.split(Lop.INSTRUCTION_DELIMITOR);
		StringBuilder instOut = new StringBuilder();
		for( String lpinst : pinst ){ //for each instruction
			//split instruction into parts
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(lpinst);
			//replace instruction parts
			for( int i=0; i<parts.length; i++ )
			{
				String lpart = parts[i];
				int pos = lpart.indexOf(Instruction.DATATYPE_PREFIX);
				if( pos>0 ){
					String index = lpart.substring(0, pos);	
					String newindex = String.valueOf(transMap.get(Byte.parseByte(index)));
					parts[i] = newindex + lpart.substring(pos); 
				}
			}
			
			if( instOut.length()>0 )
				instOut.append(Lop.INSTRUCTION_DELIMITOR);
			
			//reconstruct instruction
			instOut.append("MR");
			for( String lpart : parts ){
				instOut.append(Lop.OPERAND_DELIMITOR);
				instOut.append(lpart);
			}
		}
		
		return instOut.toString();
	}

	/**
	 * 
	 * @param inst1
	 * @param inst2
	 * @return
	 */
	private String concatenateInstructions(String inst1, String inst2)
	{
		boolean emptyInst1 = (inst1 == null || inst1.length()==0);
		boolean emptyInst2 = (inst2 == null || inst2.length()==0);
		String ret = "";
		
		if( !emptyInst1 && !emptyInst2 )
			ret = inst1 + Lop.INSTRUCTION_DELIMITOR + inst2;
		else if( !emptyInst1 )
			ret = inst1;
		else if( !emptyInst2 )
			ret = inst2;
		
		return ret;
	}
	
}
