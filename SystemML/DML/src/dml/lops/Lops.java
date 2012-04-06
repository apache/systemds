package dml.lops;

import java.util.ArrayList;

import dml.lops.LopProperties.ExecLocation;
import dml.lops.LopProperties.ExecType;
import dml.lops.compile.Dag;
import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.utils.LopsException;

/**
 * Base class for all Lops.
 * 
 * @author aghoting
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
		Tertiary, SortKeys, PickValues, CombineUnary, CombineBinary, CombineTertiary, MMRJ, CentralMoment, CoVariance, GroupedAgg, Append
	};

	public enum VISIT_STATUS {DONE, VISITING, NOTVISITED}

	private VISIT_STATUS _visited = VISIT_STATUS.NOTVISITED;
	private DataType _dataType;
	private ValueType _valueType;

	public static final String INSTRUCTION_DELIMITOR = ",";
	public static final String OPERAND_DELIMITOR = "\u00b0"; //\u2021"; //00ea"; //"::#::";
	public static final String VALUETYPE_PREFIX = "\u00b7" ; //":#:";
	public static final String DATATYPE_PREFIX = "\u00b7" ; //":#:";
	public static final String NAME_VALUE_SEPARATOR = "="; // e.g., used in parameterized builtins

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
			System.out.println("FORMAT:" + this.getOutputParameters().getFormat() + ", rows="
					+ this.getOutputParameters().getNum_rows() + ", cols=" + this.getOutputParameters().getNum_cols()
					+ ", Blocked?: " + this.getOutputParameters().isBlocked_representation());
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
		throw new LopsException("Should never be invoked in Baseclass");

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
		throw new LopsException("Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public String getInstructions(int input_index1, int input_index2, int input_index3, int output_index) throws LopsException {
		throw new LopsException("Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public String getInstructions(int input_index1, int input_index2, int output_index) throws LopsException {
		throw new LopsException("Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public String getInstructions(String input1, String input2, String output) throws LopsException {
		throw new LopsException("Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public String getInstructions(String input1, String output) throws LopsException {
		throw new LopsException("Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public String getInstructions() throws LopsException {
		throw new LopsException("Should never be invoked in Baseclass");
	}

	/** Method should be overridden if needed **/
	public SimpleInstType getSimpleInstructionType() throws LopsException {
		throw new LopsException("Should never be invoked in Baseclass");
	}

}
