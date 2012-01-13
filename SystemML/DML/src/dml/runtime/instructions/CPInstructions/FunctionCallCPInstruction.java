package dml.runtime.instructions.CPInstructions;

import java.util.ArrayList;
import java.util.HashMap;

//import org.netlib.lapack.LAPACK;

import dml.api.DMLScript;
import dml.lops.OutputParameters.Format;
import dml.parser.DataIdentifier;
import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.runtime.controlprogram.FunctionProgramBlock;
import dml.runtime.controlprogram.Program;
import dml.runtime.controlprogram.ProgramBlock;
import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;
import dml.utils.ExtBuiltinHelper;
import dml.utils.LanguageException;

/**
 * 
 */
public class FunctionCallCPInstruction extends CPInstruction {

	public String _functionName;
	
	//private CPOperand _output;
	
	private HashMap<String, Data> _inputs = new HashMap<String, Data>();
	
	// stores both the bound input and output parameters
	private ArrayList<String> _boundInputParamNames;
	private ArrayList<String> _boundOutputParamNames;
	
	public FunctionCallCPInstruction(String functName, ArrayList<String> boundInParamNames, ArrayList<String> boundOutParamNames, String istr) {
		super(null);
		
		cptype = CPINSTRUCTION_TYPE.External;
		_functionName = functName;
		instString = istr;
		_boundInputParamNames = boundInParamNames;
		_boundOutputParamNames = boundOutParamNames;
		
	}
		
	/**
	 * Instruction format extFunct:::[FUNCTION NAME]:::[num input params]:::[num output params]:::[list of delimited input params ]:::[list of delimited ouput params]
	 * These are the "bound names" for the inputs / outputs.  For example, out1 = foo(in1, in2) yields
	 * extFunct:::foo:::2:::1:::in1:::in2:::out1
	 * 
	 */
	public static Instruction parseInstruction(String str) throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		HashMap<String, Data> ins = new HashMap<String,Data>(); 
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		String functionName = parts[1];
		int numInputs = new Integer(parts[2]).intValue();
		int numOutputs = new Integer(parts[3]).intValue();
		ArrayList<String> boundInParamNames = new ArrayList<String>();
		ArrayList<String> boundOutParamNames = new ArrayList<String>();
		
		int FIRST_PARAM_INDEX = 4;
		for (int i = 0; i < numInputs; i++) {
			boundInParamNames.add(parts[FIRST_PARAM_INDEX + i]);
		}
		for (int i = 0; i < numOutputs; i++) {
			boundOutParamNames.add(parts[FIRST_PARAM_INDEX + numInputs + i]);
		}
		
		return new FunctionCallCPInstruction ( functionName, boundInParamNames, boundOutParamNames, str );
	}

	public Data processInstruction(ProgramBlock pb) throws DMLRuntimeException, DMLUnsupportedOperationException {
		if (DMLScript.DEBUG)
			System.out.println("executing instruction : " + this.toString());

		// get the function program block (stored in the Program object)
		FunctionProgramBlock fpb = pb.getProgram().getFunctionProgramBlock(this._functionName);

		// create bindings to formal parameters for given function call
		// These are the bindings passed to the FunctionProgramBlock for function execution 
		HashMap<String, Data> functionVariables = new HashMap<String, Data>();
		
		for (int i=0; i<fpb.getInputParams().size();i++) {
			
			// for each formal parameter:  create a [formalParamName, variable] binding based on the function call, 
			// and place each binding into functionVariables. 4 cases for corresponding functionCall param:
			//  (1) is a variable 
			//			-- will be DataIdentifier expression (but not FunctionCallIdentifer) 
			//			-- look in _variables for the Data value 
			//			-- create [formalParamName, ScalarObject] binding
			//  (2) is a constant value -- will be ConstIdentifier expression -- create [formalParamName, ScalarObject] binding where ScalarObject is constant value
			//  (3) is default value -- will be NO expression --  create [formalParamName, ScalarObject] where ScalarObject is default value
			//	(4) is an expression [NOT SUPPORTED YET]
			
			DataIdentifier currFormalParam = fpb.getInputParams().get(i);
			String currFormalParamName = currFormalParam.getName();
			Data currFormalParamValue = null; 
			ValueType valType = fpb.getInputParams().get(i).getValueType();
					
			if (i > this._boundInputParamNames.size() || (pb.getVariables().get(this._boundInputParamNames.get(i)) == null)){
				// CASE (3): using default value 
				
				if (valType == ValueType.BOOLEAN){
					boolean defaultVal = (i > this._boundInputParamNames.size()) ? new Boolean(fpb.getInputParams().get(i).getDefaultValue()).booleanValue() : new Boolean(this._boundInputParamNames.get(i)).booleanValue();
					currFormalParamValue = new BooleanObject(defaultVal);
				}
				else if (valType == ValueType.DOUBLE){
					double defaultVal = (i > this._boundInputParamNames.size()) ? new Double(fpb.getInputParams().get(i).getDefaultValue()).doubleValue() : new Double(this._boundInputParamNames.get(i)).doubleValue();
					currFormalParamValue = new DoubleObject(defaultVal);
				}
				else if (valType == ValueType.INT){
					int defaultVal = (i > this._boundInputParamNames.size()) ? new Integer(fpb.getInputParams().get(i).getDefaultValue()).intValue() : new Integer(this._boundInputParamNames.get(i)).intValue();
					currFormalParamValue = new IntObject(defaultVal);
				}
				else if (valType == ValueType.STRING){
					String defaultVal = (i > this._boundInputParamNames.size()) ? fpb.getInputParams().get(i).getDefaultValue() : this._boundInputParamNames.get(i);
					currFormalParamValue = new StringObject(defaultVal);
				}
				else{
					throw new DMLUnsupportedOperationException(currFormalParamValue + " has inapporpriate value type");
				}
			}
			
			else {
				currFormalParamValue = pb.getVariables().get(this._boundInputParamNames.get(i));
			}
				
			functionVariables.put(currFormalParamName,currFormalParamValue);	
					
		}
			
		// execute the function block
		fpb.setVariables(functionVariables);
		fpb.setMetaData(pb.getMetaData());
	
		fpb.execute(null);
		
		HashMap<String,Data> returnedVariables = fpb.getVariables(); 
		
		// add the updated binding for each return variable to the program block variables
		for (int i=0; i< fpb.getOutputParams().size(); i++){
		
			String boundVarName = this._boundOutputParamNames.get(i); 
			Data boundValue = returnedVariables.get(fpb.getOutputParams().get(i).getName());
			if (boundValue == null)
				throw new DMLUnsupportedOperationException(boundVarName + " was not assigned a return value");
		
			pb.getVariables().put(boundVarName, boundValue);
		}
	
		// this is the place where we can check if the function wrote any files out.
		pb.setMetaData(fpb.getMetaData());
			
		//return result;
		
		return null;
	}

	//
	//public String getOutputVariableName() {
	//	return null; //output.get_name();
	//}

	@Override
	public void printMe() {
		System.out.println("ExternalBuiltInFunction: " + this.toString());
	}

	public String getGraphString() {
		return "ExtBuiltinFunc: " + _functionName;
	}
}
