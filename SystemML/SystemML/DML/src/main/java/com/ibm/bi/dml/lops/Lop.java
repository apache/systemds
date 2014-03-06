/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.Dag;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


/**
 * Base class for all Lops.
 */

public abstract class Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/**
	 * Lop types
	 */
	public enum SimpleInstType {
		Scalar, Variable, File
	};

	public enum Type {
		Aggregate, MMCJ, Grouping, Data, Transform, UNARY, Binary, PartialAggregate, BinaryCP, UnaryCP, DataGen, ReBlock,  
		PartitionLop, CrossvalLop, GenericFunctionLop, ExtBuiltInFuncLop, ParameterizedBuiltin, 
		Tertiary, SortKeys, PickValues, CombineUnary, CombineBinary, CombineTertiary, MMRJ, CentralMoment, CoVariance, GroupedAgg, 
		Append, RangeReIndex, LeftIndex, ZeroOut, MVMult, MMTSJ, DataPartition, FunctionCallCP, CSVReBlock
	};

	public enum VISIT_STATUS {DONE, VISITING, NOTVISITED}

	protected static final Log LOG =  LogFactory.getLog(Lop.class.getName());
	
	private VISIT_STATUS _visited = VISIT_STATUS.NOTVISITED;
	
	// Boolean array to hold the list of nodes(lops) in the DAG that are reachable from this lop.
	private boolean[] reachable = null;
	private DataType _dataType;
	private ValueType _valueType;

	public static final String FILE_SEPARATOR = "/";
	public static final String PROCESS_PREFIX = "_p";
	
	//TODO MB: change delimiters to specific chars or check literals in script; otherwise potential conflicts on instruction serialization
	public static final String INSTRUCTION_DELIMITOR = "\u2021"; // "\u002c"; //",";
	public static final String OPERAND_DELIMITOR = "\u00b0"; //\u2021"; //00ea"; //"::#::";
	public static final String VALUETYPE_PREFIX = "\u00b7" ; //":#:";
	public static final String DATATYPE_PREFIX = VALUETYPE_PREFIX; //":#:";
	public static final String LITERAL_PREFIX = VALUETYPE_PREFIX; //":#:";
	public static final String NAME_VALUE_SEPARATOR = "="; // e.g., used in parameterized builtins
	public static final String SEPARATOR_WITHIN_OPRAND = "\u204F";
	public static final String VARIABLE_NAME_PLACEHOLDER = "##"; //TODO: use in LOPs 
	public static final String MATRIX_VAR_NAME_PREFIX = "_mVar";
	public static final String SCALAR_VAR_NAME_PREFIX = "_Var";
	
	/**
	 * get visit status of node
	 * 
	 * @return
	 */

	public VISIT_STATUS get_visited() {
		return _visited;
	}
	
	public boolean[] get_reachable() {
		return reachable;
	}

	public boolean[] create_reachable(int size) {
		reachable = new boolean[size];
		return reachable;
	}

	/**
	 * set visit status of node
	 * 
	 * @param visited
	 */
	public void set_visited(VISIT_STATUS visited) {
		_visited = visited;
	}

	/**
	 * get data type of the output that is produced by this lop
	 * 
	 * @return
	 */

	public DataType get_dataType() {
		return _dataType;
	}

	/**
	 * set data type of the output that is produced by this lop
	 * 
	 * @param dt
	 */
	public void set_dataType(DataType dt) {
		_dataType = dt;
	}

	/**
	 * get value type of the output that is produced by this lop
	 * 
	 * @return
	 */

	public ValueType get_valueType() {
		return _valueType;
	}

	/**
	 * set value type of the output that is produced by this lop
	 * 
	 * @param vt
	 */
	public void set_valueType(ValueType vt) {
		_valueType = vt;
	}

	Lop.Type type;

	/**
	 * transient indicator
	 */

	boolean hasTransientParameters = false;

	/**
	 * handle to all inputs and outputs.
	 */

	ArrayList<Lop> inputs;
	ArrayList<Lop> outputs;
	
	/**
	 * refers to #lops whose input is equal to the output produced by this lop.
	 * This is used in generating rmvar instructions as soon as the output produced
	 * by this lop is consumed. Otherwise, such rmvar instructions are added 
	 * at the end of program blocks. 
	 * 
	 */
	int consumerCount;

	/**
	 * handle to output parameters, dimensions, blocking, etc.
	 */

	OutputParameters outParams = null;

	LopProperties lps = null;
	
	/**
	 * Constructor to be invoked by base class.
	 * 
	 * @param t
	 */

	public Lop(Type t, DataType dt, ValueType vt) {
		type = t;
		_dataType = dt; // data type of the output produced from this LOP
		_valueType = vt; // value type of the output produced from this LOP
		inputs = new ArrayList<Lop>();
		outputs = new ArrayList<Lop>();
		outParams = new OutputParameters();
		lps = new LopProperties();
	}

	/**
	 * Method to get Lop type.
	 * 
	 * @return
	 */

	public Lop.Type getType() {
		return type;
	}

	/**
	 * Method to get input of Lops
	 * 
	 * @return
	 */
	public ArrayList<Lop> getInputs() {
		return inputs;
	}

	/**
	 * Method to get output of Lops
	 * 
	 * @return
	 */

	public ArrayList<Lop> getOutputs() {
		return outputs;
	}

	/**
	 * Method to add input to Lop
	 * 
	 * @param op
	 */

	public void addInput(Lop op) {
		inputs.add(op);
	}

	/**
	 * Method to add output to Lop
	 * 
	 * @param op
	 */

	public void addOutput(Lop op) {
		outputs.add(op);
	}
	
	public int getConsumerCount() {
		return consumerCount;
	}
	
	public void setConsumerCount(int cc) {
		consumerCount = cc;
	}
	
	public int removeConsumer() {
		consumerCount--;
		return consumerCount;
	}

	/**
	 * Method to have Lops print their state. This is for debugging purposes.
	 */

	public abstract String toString();

	public void resetVisitStatus() {
		if (this.get_visited() == Lop.VISIT_STATUS.NOTVISITED)
			return;
		for (int i = 0; i < this.getInputs().size(); i++) {
			this.getInputs().get(i).resetVisitStatus();
		}
		this.set_visited(Lop.VISIT_STATUS.NOTVISITED);
	}

	/**
	 * Method to have recursively print state of Lop graph.
	 */

	public final void printMe() {
		if (LOG.isDebugEnabled()){
			StringBuilder s = new StringBuilder("");
			if (this.get_visited() != VISIT_STATUS.DONE) {
				s.append(getType() + ": " + getID() + "\n" ); // hashCode());
				s.append("Inputs: ");
				for (int i = 0; i < this.getInputs().size(); i++) {
					s.append(" " + this.getInputs().get(i).getID() + " ");
				}

				s.append("\n");
				s.append("Outputs: ");
				for (int i = 0; i < this.getOutputs().size(); i++) {
					s.append(" " + this.getOutputs().get(i).getID() + " ");
				}

				s.append("\n");
				s.append(this.toString());
				s.append("Begin Line: " + _beginLine + ", Begin Column: " + _beginColumn + ", End Line: " + _endLine + ", End Column: " + _endColumn + "\n");
				s.append("FORMAT:" + this.getOutputParameters().getFormat() + ", rows="
						+ this.getOutputParameters().getNum_rows() + ", cols=" + this.getOutputParameters().getNum_cols()
						+ ", Blocked?: " + this.getOutputParameters().isBlocked_representation() + ", rowsInBlock=" + 
						this.getOutputParameters().get_rows_in_block() + ", colsInBlock=" + 
						this.getOutputParameters().get_cols_in_block() + "\n");
				this.set_visited(VISIT_STATUS.DONE);
				s.append("\n");

				for (int i = 0; i < this.getInputs().size(); i++) {
					this.getInputs().get(i).printMe();
				}
			}
			LOG.debug(s.toString());
		}
	}

	/**
	 * Method to return the ID of LOP
	 */
	public long getID() {
		return lps.getID();
	}
	
	public int getLevel() {
		return lps.getLevel();
	}
	
	public void setLevel() {
		lps.setLevel(inputs);
	}
	
	/**
	 * Method to get the location property of LOP
	 * 
	 * @return
	 */
 	public ExecLocation getExecLocation() {
		return lps.getExecLocation();
	}
 
	/**
	 * Method to get the execution type (CP or MR) of LOP
	 * 
	 * @return
	 */
 	public ExecType getExecType() {
		return lps.getExecType();
	}
 
	/**
	 * Method to get the compatible job type for the LOP
	 * @return
	 */
	
	public int getCompatibleJobs() {
		return lps.getCompatibleJobs();
	}
	
	/**
	 * Method to find if the lop breaks alignment
	 */
	public boolean getBreaksAlignment() {
		return lps.getBreaksAlignment();
	}
	
	
	public boolean isAligner()
	{
		return lps.isAligner();
	}

	public boolean definesMRJob()
	{
		return lps.getDefinesMRJob();
	}

	/**
	 * Method to recursively add LOPS to a DAG
	 * 
	 * @param dag
	 */
	public final void addToDag(Dag<Lop> dag) 
	{
		if( dag.addNode(this) )
			for( Lop l : getInputs() )
				l.addToDag(dag);
	}

	/**
	 * Method should be overridden if needed
	 * 
	 * @throws LopsException
	 **/
	public String getInstructions(int input_index, int output_index) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass. Lop Type: " + this.getType());

	}

	/**
	 * Method to get output parameters
	 * 
	 * @return
	 */

	public OutputParameters getOutputParameters() {
		return outParams;
	}
	
	/** Method should be overridden if needed **/
	public String getInstructions(String input1, String input2, String input3, String output) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}
	
	/** Method should be overridden if needed **/
	public String getInstructions(String input1, String input2, String input3, String input4, String output) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String output) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String input6, String output) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}
	
	/** Method should be overridden if needed **/
	public String getInstructions(int input_index1, int input_index2, int input_index3, int output_index) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}
	
	/** Method should be overridden if needed **/
	public String getInstructions(int input_index1, int input_index2, int input_index3, int input_index4, int output_index) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public String getInstructions(int input_index1, int input_index2, int input_index3, int input_index4, int input_index5, int output_index) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public String getInstructions(int input_index1, int input_index2, int output_index) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public String getInstructions(String input1, String input2, String output) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public String getInstructions(String input1, String output) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public String getInstructions(String output) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public String getInstructions(String[] inputs, String[] outputs) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}
	
	/** Method should be overridden if needed **/
	public String getInstructions() throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public SimpleInstType getSimpleInstructionType() throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	///////////////////////////////////////////////////////////////////////////
	// store position information for Lops
	///////////////////////////////////////////////////////////////////////////
	public int _beginLine, _beginColumn;
	public int _endLine, _endColumn;
	
	public void setBeginLine(int passed)    { _beginLine = passed;   }
	public void setBeginColumn(int passed) 	{ _beginColumn = passed; }
	public void setEndLine(int passed) 		{ _endLine = passed;   }
	public void setEndColumn(int passed)	{ _endColumn = passed; }
	
	public void setAllPositions(int blp, int bcp, int elp, int ecp){
		_beginLine	 = blp; 
		_beginColumn = bcp; 
		_endLine 	 = elp;
		_endColumn 	 = ecp;
	}

	public int getBeginLine()	{ return _beginLine;   }
	public int getBeginColumn() { return _beginColumn; }
	public int getEndLine() 	{ return _endLine;   }
	public int getEndColumn()	{ return _endColumn; }
	
	public String printErrorLocation(){
		return "ERROR: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	public String printWarningLocation(){
		return "WARNING: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}

	//TODO: Leo This might get confused with Rand.getInstructions
	public String getInstructions(String input, String rowl, String rowu,
			String coll, String colu, String leftRowDim,
			String leftColDim, String output) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}
	
	public String getInstructions(int input, int rowl, int rowu,
			int coll, int colu, int leftRowDim,
			int leftColDim, int output) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");
	}

	/**
	 * Function that determines if the output of a LOP is defined by a variable or not.
	 * 
	 * @return
	 */
	public boolean isVariable() {
		return ( (getExecLocation() == ExecLocation.Data && !((Data)this).isLiteral()) 
				 || !(getExecLocation() == ExecLocation.Data ) );
	}
	
	
	
	/**
	 * Method to prepare instruction operand with given parameters.
	 * 
	 * @param label
	 * @param dt
	 * @param vt
	 * @return
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
	 * @param label
	 * @param dt
	 * @param vt
	 * @return
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
	 * @param label
	 * @return
	 */
	private String prepOperand(String label) {
		StringBuilder sb = new StringBuilder("");
		sb.append(label);
		sb.append(Lop.DATATYPE_PREFIX);
		sb.append(get_dataType());
		sb.append(Lop.VALUETYPE_PREFIX);
		sb.append(get_valueType());
		return sb.toString();
	}
	
	public String prepOutputOperand() {
		return prepOperand(getOutputParameters().getLabel());
	}
	
	public String prepOutputOperand(int index) {
		return prepOperand(index+"");
	}
	public String prepOutputOperand(String label) {
		return prepOperand(label);
	}
	
	/**
	 * Function to prepare label for scalar inputs while generating instructions.
	 * It attaches placeholder suffix and prefixes if the Lop denotes a variable.
	 * 
	 * @return
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
	 * @param et
	 * @return
	 */
	public String prepScalarOperand(ExecType et, String label) {
		boolean isData = (getExecLocation() == ExecLocation.Data);
		boolean isLiteral = (isData && ((Data)this).isLiteral());
		
		StringBuilder sb = new StringBuilder("");
		if ( et == ExecType.CP || (isData && isLiteral)) {
			sb.append(label);
		}
		else {
			sb.append(Lop.VARIABLE_NAME_PLACEHOLDER);
			sb.append(label);
			sb.append(Lop.VARIABLE_NAME_PLACEHOLDER);
		}
		
		sb.append(Lop.DATATYPE_PREFIX);
		sb.append(get_dataType());
		sb.append(Lop.VALUETYPE_PREFIX);
		sb.append(get_valueType());
		sb.append(Lop.LITERAL_PREFIX);
		sb.append(isLiteral);
		
		return sb.toString();
	}

	public String prepScalarInputOperand(ExecType et) {
		return prepScalarOperand(et, getOutputParameters().getLabel());
	}
	
	public String prepScalarInputOperand(String label) {
		boolean isData = (getExecLocation() == ExecLocation.Data);
		boolean isLiteral = (isData && ((Data)this).isLiteral());
		
		StringBuilder sb = new StringBuilder("");
		sb.append(label);
		sb.append(Lop.DATATYPE_PREFIX);
		sb.append(get_dataType());
		sb.append(Lop.VALUETYPE_PREFIX);
		sb.append(get_valueType());
		sb.append(Lop.LITERAL_PREFIX);
		sb.append(isLiteral);
		
		return sb.toString();
	}

	public String prepInputOperand(int index) {
		return prepInputOperand(index+"");
	}

	public String prepInputOperand(String label) {
		DataType dt = get_dataType();
		if ( dt == DataType.MATRIX ) {
			return prepOperand(label);
		}
		else {
			return prepScalarInputOperand(label);
		}
	}
	
	/**
	 * Method to check if a LOP expects an input from the Distributed Cache.
	 * The method in parent class always returns <code>false</code> (default).
	 * It must be overridden by individual LOPs that use the cache.
	 */
	public boolean usesDistributedCache() {
		return false;
	}
	
	public int distributedCacheInputIndex() {
		return -1;
	}

}
