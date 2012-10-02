package com.ibm.bi.dml.lops;

import java.util.ArrayList;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.Dag;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LopsException;


/**
 * Base class for all Lops.
 */

public abstract class Lops {
	/**
	 * Lop types
	 */
	public enum SimpleInstType {
		Scalar, Variable, File
	};

	public enum Type {
		Aggregate, MMCJ, Grouping, Data, Transform, UNARY, Binary, PartialAggregate, BinaryCP, UnaryCP, RandLop, ReBlock,  
		PartitionLop, CrossvalLop, GenericFunctionLop, ExtBuiltInFuncLop, ParameterizedBuiltin, 
		Tertiary, SortKeys, PickValues, CombineUnary, CombineBinary, CombineTertiary, MMRJ, CentralMoment, CoVariance, GroupedAgg, Append, RangeReIndex, LeftIndex, ZeroOut
	};

	public enum VISIT_STATUS {DONE, VISITING, NOTVISITED}

	private VISIT_STATUS _visited = VISIT_STATUS.NOTVISITED;
	private DataType _dataType;
	private ValueType _valueType;

	public static final String FILE_SEPARATOR = "/";
	public static final String PROCESS_PREFIX = "_p";
	
	//TODO MB: change delimiters to specific chars or check literals in script; otherwise potential conflicts on instruction serialization
	public static final String INSTRUCTION_DELIMITOR = "\u002c"; //",";
	public static final String OPERAND_DELIMITOR = "\u00b0"; //\u2021"; //00ea"; //"::#::";
	public static final String VALUETYPE_PREFIX = "\u00b7" ; //":#:";
	public static final String DATATYPE_PREFIX = "\u00b7" ; //":#:";
	public static final String NAME_VALUE_SEPARATOR = "="; // e.g., used in parameterized builtins
	public static final String VARIABLE_NAME_PLACEHOLDER = "##"; //TODO: use in LOPs 
	
	
	/**
	 * get visit status of node
	 * 
	 * @return
	 */

	public VISIT_STATUS get_visited() {
		return _visited;
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

	Lops.Type type;

	/**
	 * transient indicator
	 */

	boolean hasTransientParameters = false;

	/**
	 * handle to all inputs and outputs.
	 */

	ArrayList<Lops> inputs;
	ArrayList<Lops> outputs;

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

	public Lops(Type t, DataType dt, ValueType vt) {
		type = t;
		_dataType = dt; // data type of the output produced from this LOP
		_valueType = vt; // value type of the output produced from this LOP
		inputs = new ArrayList<Lops>();
		outputs = new ArrayList<Lops>();
		outParams = new OutputParameters();
		lps = new LopProperties();
	}

	/**
	 * Method to get Lop type.
	 * 
	 * @return
	 */

	public Lops.Type getType() {
		return type;
	}

	/**
	 * Method to get input of Lops
	 * 
	 * @return
	 */
	public ArrayList<Lops> getInputs() {
		return inputs;
	}

	/**
	 * Method to get output of Lops
	 * 
	 * @return
	 */

	public ArrayList<Lops> getOutputs() {
		return outputs;
	}

	/**
	 * Method to add input to Lop
	 * 
	 * @param op
	 */

	public void addInput(Lops op) {
		inputs.add(op);
	}

	/**
	 * Method to add output to Lop
	 * 
	 * @param op
	 */

	public void addOutput(Lops op) {
		outputs.add(op);
	}

	/**
	 * Method to have Lops print their state. This is for debugging purposes.
	 */

	public abstract String toString();

	public void resetVisitStatus() {
		if (this.get_visited() == Lops.VISIT_STATUS.NOTVISITED)
			return;
		for (int i = 0; i < this.getInputs().size(); i++) {
			this.getInputs().get(i).resetVisitStatus();
		}
		this.set_visited(Lops.VISIT_STATUS.NOTVISITED);
	}

	/**
	 * Method to have recursively print state of Lop graph.
	 */

	public final void printMe() {
		if (this.get_visited() != VISIT_STATUS.DONE) {
			System.out.println(getType() + ": " + getID() ); // hashCode());
			System.out.print("Inputs: ");
			for (int i = 0; i < this.getInputs().size(); i++) {
				System.out.print(" " + this.getInputs().get(i).getID() + " ");
			}

			System.out.print("\n");
			System.out.print("Outputs: ");
			for (int i = 0; i < this.getOutputs().size(); i++) {
				System.out.print(" " + this.getOutputs().get(i).getID() + " ");
			}

			System.out.print("\n");
			System.out.println(this.toString());
			System.out.println("Begin Line: " + _beginLine + ", Begin Column: " + _beginColumn + ", End Line: " + _endLine + ", End Column: " + _endColumn);
			System.out.println("FORMAT:" + this.getOutputParameters().getFormat() + ", rows="
					+ this.getOutputParameters().getNum_rows() + ", cols=" + this.getOutputParameters().getNum_cols()
					+ ", Blocked?: " + this.getOutputParameters().isBlocked_representation() + ", rowsInBlock=" + 
					this.getOutputParameters().get_rows_in_block() + ", colsInBlock=" + 
					this.getOutputParameters().get_cols_in_block());
			this.set_visited(VISIT_STATUS.DONE);
			System.out.print("\n");

			for (int i = 0; i < this.getInputs().size(); i++) {
				this.getInputs().get(i).printMe();
			}
		}
	}

	/**
	 * Method to return the ID of LOP
	 */
	public int getID() {
		return lps.getID();
	}
	
	public int getLevel() {
		return lps.getLevel();
	}
	
	public void setLevel(int l) {
		lps.setLevel(l);
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

	public final void addToDag(Dag<Lops> dag) {
		dag.addNode(this);
		for (int i = 0; i < this.getInputs().size(); i++) {
			this.getInputs().get(i).addToDag(dag);
		}

	}

	/**
	 * Method should be overridden if needed
	 * 
	 * @throws LopsException
	 **/
	public String getInstructions(int input_index, int output_index) throws LopsException {
		throw new LopsException(this.printErrorLocation() + "Should never be invoked in Baseclass");

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
	
}
