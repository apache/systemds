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

package org.apache.sysds.lops;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.AggBinaryOp.SparkAggType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;

/**
 * Base class for all Lops.
 */

public abstract class Lop 
{
	protected static final Log LOG =  LogFactory.getLog(Lop.class.getName());
	
	public enum Type {
		Data, DataGen,                                      //CP/MR read/write/datagen 
		ReBlock, CSVReBlock,                                //MR reblock operations
		MatMultCP,
		MMCJ, MMRJ, MMTSJ, PMMJ, MapMult, MapMultChain,     //MR matrix multiplications
		UnaryCP, UNARY, BinaryCP, Binary, Ternary, Nary,    //CP/MR unary/binary/ternary
		RightIndex, LeftIndex, ZeroOut,                     //CP/MR indexing 
		Aggregate, PartialAggregate,                        //CP/MR aggregation
		BinUaggChain, UaggOuterChain,                       //CP/MR aggregation
		TernaryAggregate,                                   //CP ternary-binary aggregates
		Grouping,                                           //MR grouping
		Append,                                             //CP/MR append (column append)
		CombineUnary, CombineBinary, CombineTernary,        //MR combine (stitch together)
		CentralMoment, CoVariance, GroupedAgg, GroupedAggM,
		Transform, DataPartition, RepMat,                   //CP/MR reorganization, partitioning, replication
		ParameterizedBuiltin,                               //CP/MR parameterized ops (name/value)
		FunctionCallCP, FunctionCallCPSingle,               //CP function calls 
		CumulativePartialAggregate, CumulativeSplitAggregate, CumulativeOffsetBinary, //MR cumsum/cumprod/cummin/cummax
		WeightedSquaredLoss, WeightedSigmoid, WeightedDivMM, WeightedCeMM, WeightedUMM,
		SortKeys, PickValues, Ctable,
		Checkpoint,                                         //Spark persist into storage level
		PlusMult, MinusMult,                                //CP
		SpoofFused,                                         //CP/SP generated fused operator
		Sql,                                                //CP sql read
		Federated,                                           //FED federated read
		Tee,                                                //OOC Tee operator
	}
	

	/**
	 * Lop types
	 */
	public enum SimpleInstType {
		Scalar
	}

	public enum VisitStatus {
		DONE, NOTVISITED
	}
	
	public static final String FILE_SEPARATOR = "/";
	public static final String PROCESS_PREFIX = "_p";
	public static final String CP_ROOT_THREAD_ID = "_t0";
	public static final String CP_CHILD_THREAD = "_t";
	
	//special delimiters w/ extended ASCII characters to avoid collisions 
	public static final String INSTRUCTION_DELIMITOR = "\u2021";
	public static final String OPERAND_DELIMITOR = "\u00b0"; 
	public static final String VALUETYPE_PREFIX = "\u00b7" ; 
	public static final String DATATYPE_PREFIX = VALUETYPE_PREFIX; 
	public static final String LITERAL_PREFIX = VALUETYPE_PREFIX; 
	public static final String VARIABLE_NAME_PLACEHOLDER = "\u00b6"; 
	
	public static final String NAME_VALUE_SEPARATOR = "="; // e.g., used in parameterized builtins
	public static final String MATRIX_VAR_NAME_PREFIX = "_mVar";
	public static final String FRAME_VAR_NAME_PREFIX = "_fVar";
	public static final String SCALAR_VAR_NAME_PREFIX = "_Var";
	public static final String UPDATE_INPLACE_PREFIX = "_uip";

	// Boolean array to hold the list of nodes(lops) in the DAG that are reachable from this lop.
	private boolean[] reachable = null;
	private DataType _dataType;
	private ValueType _valueType;

	private VisitStatus _visited = VisitStatus.NOTVISITED;

	protected Lop.Type type;

	/**
	 * handle to all inputs and outputs.
	 */
	protected ArrayList<Lop> inputs;
	protected ArrayList<Lop> outputs;

	/**
	 * Field defining if prefetch should be activated for operation.
	 * When prefetch is activated, the output will be transferred from
	 * remote federated sites to local before one of the subsequent
	 * local operations.
	 */
	protected boolean activatePrefetch;

	/**
	 * Enum defining if the output of the operation should be forced federated, forced local or neither.
	 * If it is FOUT, the output should be kept at federated sites.
	 * If it is LOUT, the output should be retrieved by the coordinator.
	 */
	protected FederatedOutput _fedOutput = null;
	
	/**
	 * refers to #lops whose input is equal to the output produced by this lop.
	 * This is used in generating rmvar instructions as soon as the output produced
	 * by this lop is consumed. Otherwise, such rmvar instructions are added 
	 * at the end of program blocks. 
	 * 
	 */
	protected int consumerCount;

	/**
	 * handle to output parameters, dimensions, blocking, etc.
	 */

	protected OutputParameters outParams = null;

	protected LopProperties lps = null;

	/**
	 * Indicates if this lop is a candidate for asynchronous execution.
	 * Examples include spark unary aggregate, mapmm, prefetch
	 */
	protected boolean _asynchronous = false;

	/**
	 * Refers to the pipeline to which this lop belongs to.
	 * This is used for identifying parallel execution of lops.
	 */
	protected int _pipelineID = -1;

	/**
	 * Estimated size for the output produced by this Lop in bytes.
	 */
	protected double _outputMemEstimate = OptimizerUtils.INVALID_SIZE;

	/*
	 * Estimated size for the entire operation represented by this Lop
	 * It includes the memory required for all inputs as well as the output
	 * For Spark collects, _memEstimate equals _outputMemEstimate.
	 */
	protected double _memEstimate = OptimizerUtils.INVALID_SIZE;

	/**
	 * Estimated size for the intermediates produced by this Lop in bytes.
	 */
	protected double _processingMemEstimate = 0;

	/**
	 * Estimated size for the broadcast partitions.
	 */
	protected double _spBroadcastMemEstimate = 0;

	/*
	 * Compute cost for this Lop based on the number of floating point operations per
	 * output cell and the total number of output cells.
	 */
	protected double _computeCost = 0;

	/**
	 * Constructor to be invoked by base class.
	 * 
	 * @param t lop type
	 * @param dt data type of the output
	 * @param vt value type of the output
	 */
	public Lop(Type t, DataType dt, ValueType vt) {
		type = t;
		_dataType = dt; // data type of the output produced from this LOP
		_valueType = vt; // value type of the output produced from this LOP
		inputs = new ArrayList<>();
		outputs = new ArrayList<>();
		outParams = new OutputParameters();
		lps = new LopProperties();
	}
	
	/**
	 * get visit status of node
	 * 
	 * @return visit status
	 */

	public VisitStatus getVisited() {
		return _visited;
	}

	/**
	 * set visit status of node
	 * 
	 * @param visited visit status
	 */
	public void setVisited(VisitStatus visited) {
		_visited = visited;
	}

	public void setVisited() {
		setVisited(VisitStatus.DONE);
	}

	public boolean isVisited() {
		return _visited == VisitStatus.DONE;
	}

	
	public boolean[] getReachable() {
		return reachable;
	}

	public boolean[] createReachable(int size) {
		reachable = new boolean[size];
		return reachable;
	}
	
	public boolean isDataExecLocation() {
		return this instanceof Data;
	}

	protected void setupLopProperties(ExecType et) {
		//setup Spark parameters 
		lps.setProperties( inputs, et);
	}
	
	/**
	 * get data type of the output that is produced by this lop
	 * 
	 * @return data type
	 */

	public DataType getDataType() {
		return _dataType;
	}

	/**
	 * set data type of the output that is produced by this lop
	 * 
	 * @param dt data type
	 */
	public void setDataType(DataType dt) {
		_dataType = dt;
	}

	/**
	 * get value type of the output that is produced by this lop
	 * 
	 * @return value type
	 */

	public ValueType getValueType() {
		return _valueType;
	}

	/**
	 * set value type of the output that is produced by this lop
	 * 
	 * @param vt value type
	 */
	public void setValueType(ValueType vt) {
		_valueType = vt;
	}


	/**
	 * Method to get Lop type.
	 * 
	 * @return lop type
	 */

	public Lop.Type getType() {
		return type;
	}

	/**
	 * Method to get input of Lops
	 * 
	 * @return list of input lops
	 */
	public ArrayList<Lop> getInputs() {
		return inputs;
	}

	public Lop getInput(int index) {
		return inputs.get(index);
	}

	/**
	 * Method to get output of Lops
	 * 
	 * @return list of output lops
	 */

	public ArrayList<Lop> getOutputs() {
		return outputs;
	}

	/**
	 * Method to add input to Lop
	 * 
	 * @param op input lop
	 */

	public void addInput(Lop op) {
		inputs.add(op);
	}
	
	/**
	 * Method to replace an input to a Lop
	 * @param oldInp old input Lop
	 * @param newInp new input Lop
	 */
	public void replaceInput(Lop oldInp, Lop newInp) {
		if (inputs.contains(oldInp)) {
			int index = inputs.indexOf(oldInp);
			inputs.set(index, newInp);
		}
	}

	public void replaceAllInputs(ArrayList<Lop> newInputs) {
		inputs = newInputs;
	}

	public void replaceAllOutputs(ArrayList<Lop> newOutputs) {
		outputs = newOutputs;
	}

	public void removeInput(Lop op) {
		inputs.remove(op);
	}

	/**
	 * Method to add output to Lop
	 * 
	 * @param op output lop
	 */

	public void addOutput(Lop op) {
		outputs.add(op);
	}
	
	/**
	 * Method to remove output from Lop
	 * @param op Lop to remove
	 */
	public void removeOutput(Lop op) {
		outputs.remove(op);
	}

	public void activatePrefetch(){
		activatePrefetch = true;
	}

	public boolean prefetchActivated(){
		return activatePrefetch;
	}

	public void setFederatedOutput(FederatedOutput fedOutput){
		_fedOutput = fedOutput;
		LOG.trace("Set federated output: " + fedOutput + " of lop " + this);
	}

	public FederatedOutput getFederatedOutput(){
		return _fedOutput;
	}
	
	public void setConsumerCount(int cc) {
		consumerCount = cc;
	}
	
	public int removeConsumer() {
		consumerCount--;
		return consumerCount;
	}

	public void setAsynchronous(boolean isAsync) {
		_asynchronous = isAsync;
	}

	public boolean isAsynchronousOp() {
		return _asynchronous;
	}

	public void setPipelineID(int id) {
		_pipelineID = id;
	}

	public int getPipelineID() {
		return _pipelineID;
	}

	public void setMemoryEstimates(double outMem, double totMem, double interMem, double bcMem) {
		_outputMemEstimate = outMem;
		_memEstimate = totMem;
		_processingMemEstimate = interMem;
		_spBroadcastMemEstimate = bcMem;
	}

	public double getTotalMemoryEstimate() {
		return _memEstimate;
	}

	public double getOutputMemoryEstimate() {
		return _outputMemEstimate;
	}

	public void setComputeEstimate(double compCost) {
		_computeCost = compCost;
	}

	public double getComputeEstimate() {
		return _computeCost;
	}

	/**
	 * Method to have Lops print their state. This is for debugging purposes.
	 */
	@Override
	public abstract String toString();

	public void resetVisitStatus() {
		if (this.getVisited() == Lop.VisitStatus.NOTVISITED)
			return;
		for (int i = 0; i < this.getInputs().size(); i++) {
			this.getInputs().get(i).resetVisitStatus();
		}
		this.setVisited(Lop.VisitStatus.NOTVISITED);
	}

	/**
	 * Method to return the ID of LOP
	 * 
	 * @return lop ID
	 */
	public long getID() {
		return lps.getID();
	}

	public void setNewID() {
		lps.setNewID();
	}
	
	public int getLevel() {
		return lps.getLevel();
	}
	
	protected void setLevel() {
		lps.setLevel(inputs);
	}
	
	protected void updateLevel(int newLevel) {
		if(newLevel < getLevel()) {
			throw new RuntimeException("Decrement the levels not supported.");
		}
		else if(newLevel > getLevel()) {
			lps.setLevel(newLevel);
			for(Lop out : outputs) {
				if(out.getLevel() < newLevel+1)
					out.updateLevel(newLevel+1);
			}
		}
	}

	/**
	 * Method to get the execution type (CP, CP_FILE, MR, SPARK, GPU, FED, INVALID) of LOP
	 * 
	 * @return execution type
	 */
 	public ExecType getExecType() {
		return lps.getExecType();
	}

	/**
	 * Set the execution type of LOP.
	 * 
	 * @param newExecType new execution type
	 */
	public void setExecType(ExecType newExecType) {
		lps.setExecType(newExecType);
	}

	public boolean isExecSpark () {
		return (lps.getExecType() == ExecType.SPARK);
	}

	public boolean isExecGPU () {
		return (lps.getExecType() == ExecType.GPU);
	}

	public boolean isExecCP () {
		return (lps.getExecType() == ExecType.CP);
	}

	public boolean getProducesIntermediateOutput() {
		return lps.getProducesIntermediateOutput();
	}

	/**
	 * Method to recursively add LOPS to a DAG
	 * 
	 * @param dag lop DAG
	 */
	public final void addToDag(Dag<Lop> dag) {
		if( dag.addNode(this) )
			for( Lop l : getInputs() )
				l.addToDag(dag);
	}

	/**
	 * Method to get output parameters
	 * 
	 * @return output parameters
	 */

	public OutputParameters getOutputParameters() {
		return outParams;
	}

	public long getNumRows() {
		return getOutputParameters().getNumRows();
	}

	public long getNumCols() {
		return getOutputParameters().getNumCols();
	}

	public long getNnz() {
		return getOutputParameters().getNnz();
	}

	/**
	 * Method to get aggregate type if applicable.
	 * This method is overridden by the Lops with aggregate types (e.g. MapMult)
	 * @return SparkAggType
	 */
	public SparkAggType getAggType() {
		return SparkAggType.NONE;
	}
	
	/**
	 * Method to get the input to be broadcast.
	 * This method is overridden by the Lops which require broadcasts (e.g. AppendM)
	 * @return An input Lop or Null
	 */
	public Lop getBroadcastInput() {
		return null;
	}
	

	/** Method should be overridden if needed
	 * 
	 * @param output output
	 * @return instructions as string
	 */
	public String getInstructions(String output) {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}
	
	/** Method should be overridden if needed
	 * 
	 * @param input1 input 1
	 * @param output output
	 * @return instructions as string
	 */
	public String getInstructions(String input1, String output) {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed
	 * 
	 * @param input1 input 1
	 * @param input2 input 2
	 * @param output output
	 * @return instructions as string
	 */
	public String getInstructions(String input1, String input2, String output) {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}
	
	/**
	 * Method should be overridden if needed
	 * 
	 * @param input1 input 1
	 * @param input2 input 2
	 * @param input3 input 3
	 * @param output output
	 * @return instructions as string
	 */
	public String getInstructions(String input1, String input2, String input3, String output) {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}
	
	/**
	 * Method should be overridden if needed
	 * 
	 * @param input1 input 1
	 * @param input2 input 2
	 * @param input3 input 3
	 * @param input4 input 4
	 * @param output output
	 * @return instructions as string
	 */
	public String getInstructions(String input1, String input2, String input3, String input4, String output) {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/**
	 * Method should be overridden if needed
	 * 
	 * @param input1 input 1
	 * @param input2 input 2
	 * @param input3 input 3
	 * @param input4 input 4
	 * @param input5 input 5
	 * @param output output
	 * @return instructions as string
	 */
	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String output) {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed
	 * 
	 * @param input1 input 1
	 * @param input2 input 2
	 * @param input3 input 3
	 * @param input4 input 4
	 * @param input5 input 5
	 * @param input6 input 6
	 * @param output output
	 * @return instructions as string
	 */
	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String input6, String output) {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String input6, String input7, String output) {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}
	
	public String getInstructions(String[] inputs, String outputs) {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed
	 * 
	 * @param inputs array of inputs
	 * @param outputs array of outputs
	 * @return instructions as string
	 */
	public String getInstructions(String[] inputs, String[] outputs) {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}
	
	/** Method should be overridden if needed
	 * 
	 * @return instructions as string
	 */
	public String getInstructions() {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed
	 * 
	 * @return simple instruction type
	 */
	public SimpleInstType getSimpleInstructionType() {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	///////////////////////////////////////////////////////////////////////////
	// store position information for Lops
	///////////////////////////////////////////////////////////////////////////
	public int _beginLine, _beginColumn;
	public int _endLine, _endColumn;
	public String _filename;
	
	public void setBeginLine(int passed)    { _beginLine = passed;   }
	public void setBeginColumn(int passed) 	{ _beginColumn = passed; }
	public void setEndLine(int passed) 		{ _endLine = passed;   }
	public void setEndColumn(int passed)	{ _endColumn = passed; }
	public void setFilename(String passed) { _filename = passed; }
	
	public void setAllPositions(String filename, int blp, int bcp, int elp, int ecp){
		_filename = filename;
		_beginLine	 = blp; 
		_beginColumn = bcp; 
		_endLine 	 = elp;
		_endColumn 	 = ecp;
	}

	public int getBeginLine()	{ return _beginLine;   }
	public int getBeginColumn() { return _beginColumn; }
	public int getEndLine() 	{ return _endLine;   }
	public int getEndColumn()	{ return _endColumn; }
	public String getFilename()	{ return _filename; }
	
	public String printErrorLocation(){
		return "ERROR: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}

	public String getInstructions(int input, int rowl, int rowu,
			int coll, int colu, int leftRowDim,
			int leftColDim, int output) {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/**
	 * Function that determines if the output of a LOP is defined by a variable or not.
	 * 
	 * @return true if lop output defined by a variable
	 */
	public boolean isVariable() {
		return ( (isDataExecLocation() && !((Data)this).isLiteral()) 
				 || !isDataExecLocation() );
	}
	
	/**
	 * Function that determines if all the outputs of a LOP are of CP execution types
	 * 
	 * @return true if all outputs are CP
	 */
	public boolean isAllOutputsCP() {
		if (outputs.isEmpty())
			return false;

		boolean outCP = true;
		for (Lop out : getOutputs()) {
			if (out.getExecType() != ExecType.CP) {
				outCP = false;
				break;
			}
		}
		return outCP;
	}

	/**
	 * Function that determines if all the outputs of a LOP are of GPU execution types
	 *
	 * @return true if all outputs are CP
	 */
	public boolean isAllOutputsGPU() {
		if (outputs.isEmpty())
			return false;

		boolean outGPU = true;
		for (Lop out : getOutputs()) {
			if (out.getExecType() != ExecType.GPU) {
				outGPU = false;
				break;
			}
		}
		return outGPU;
	}

	/**
	 * Method to prepare instruction operand with given parameters.
	 * 
	 * @param label instruction label
	 * @param dt data type
	 * @param vt value type
	 * @return instruction operand with data type and value type
	 */
	public String prepOperand(String label, DataType dt, ValueType vt) {
		StringBuilder sb = new StringBuilder();
		sb.append(label);
		sb.append(Lop.DATATYPE_PREFIX);
		sb.append(dt);
		sb.append(Lop.VALUETYPE_PREFIX);
		sb.append(vt);
		return sb.toString();
	}

	/**
	 * Method to prepare instruction operand with given parameters.
	 * 
	 * @param label instruction label
	 * @param dt data type
	 * @param vt value type
	 * @param literal true if literal
	 * @return instruction operand with data type, value type, and literal status
	 */
	public String prepOperand(String label, DataType dt, ValueType vt, boolean literal) {
		StringBuilder sb = new StringBuilder();
		sb.append(label);
		sb.append(Lop.DATATYPE_PREFIX);
		sb.append(dt);
		sb.append(Lop.VALUETYPE_PREFIX);
		sb.append(vt);
		sb.append(Lop.LITERAL_PREFIX);
		sb.append(literal);
		return sb.toString();
	}

	/**
	 * Method to prepare instruction operand with given label. Data type
	 * and Value type are derived from Lop's properties.
	 * 
	 * @param label instruction label
	 * @return instruction operand with data type and value type
	 */
	private String prepOperand(String label) {
		StringBuilder sb = new StringBuilder("");
		sb.append(label);
		sb.append(Lop.DATATYPE_PREFIX);
		sb.append(getDataType());
		sb.append(Lop.VALUETYPE_PREFIX);
		sb.append(getValueType());
		return sb.toString();
	}
	
	public String prepOutputOperand() {
		return prepOperand(getOutputParameters().getLabel());
	}
	
	public String prepOutputOperand(int index) {
		return prepOperand(String.valueOf(index));
	}
	public String prepOutputOperand(String label) {
		return prepOperand(label);
	}
	
	/**
	 * Function to prepare label for scalar inputs while generating instructions.
	 * It attaches placeholder suffix and prefixes if the Lop denotes a variable.
	 * 
	 * @return prepared scalar label
	 */
	public String prepScalarLabel() {
		String ret = getOutputParameters().getLabel();
		if ( isVariable() ){
			ret = Lop.VARIABLE_NAME_PLACEHOLDER + ret + Lop.VARIABLE_NAME_PLACEHOLDER;
		}
		return ret;
	}
	
	/**
	 * Function to be used in creating instructions for creating scalar
	 * operands. It decides whether or not attach placeholders for instruction
	 * patching. Resulting string also encodes if the operand is a literal.
	 * 
	 * For non-literals: 
	 * Placeholder prefix and suffix need to be attached for Instruction 
	 * Patching during execution. However, they should NOT be attached IF: 
	 *   - the operand is a literal 
	 *     OR 
	 *   - the execution type is CP. This is because CP runtime has access 
	 *     to symbol table and the instruction encodes sufficient information
	 *     to determine if an operand is a literal or not.
	 * 
	 * @param et execution type
	 * @param label instruction label
	 * @return prepared scalar operand
	 */
	public String prepScalarOperand(ExecType et, String label) {
		boolean isData = isDataExecLocation();
		boolean isLiteral = (isData && ((Data)this).isLiteral());
		
		StringBuilder sb = new StringBuilder("");
		if ( et == ExecType.CP || et == ExecType.SPARK || et == ExecType.GPU || (isData && isLiteral)) {
			sb.append(label);
		}
		else {
			sb.append(Lop.VARIABLE_NAME_PLACEHOLDER);
			sb.append(label);
			sb.append(Lop.VARIABLE_NAME_PLACEHOLDER);
		}
		
		sb.append(Lop.DATATYPE_PREFIX);
		sb.append(getDataType());
		sb.append(Lop.VALUETYPE_PREFIX);
		sb.append(getValueType());
		sb.append(Lop.LITERAL_PREFIX);
		sb.append(isLiteral);
		
		return sb.toString();
	}

	public String prepScalarInputOperand(ExecType et) {
		return prepScalarOperand(et, getOutputParameters().getLabel());
	}
	
	public String prepScalarInputOperand(String label) {
		boolean isData = isDataExecLocation();
		boolean isLiteral = (isData && ((Data)this).isLiteral());
		
		StringBuilder sb = new StringBuilder("");
		sb.append(label);
		sb.append(Lop.DATATYPE_PREFIX);
		sb.append(getDataType());
		sb.append(Lop.VALUETYPE_PREFIX);
		sb.append(getValueType());
		sb.append(Lop.LITERAL_PREFIX);
		sb.append(isLiteral);
		
		return sb.toString();
	}

	public String prepInputOperand(int index) {
		return prepInputOperand(String.valueOf(index));
	}

	public String prepInputOperand(String label) {
		DataType dt = getDataType();
		if ( dt == DataType.MATRIX ) {
			return prepOperand(label);
		}
		else {
			return prepScalarInputOperand(label);
		}
	}
}
