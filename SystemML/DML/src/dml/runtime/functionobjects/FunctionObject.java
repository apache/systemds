package dml.runtime.functionobjects;

import java.util.HashMap;

import dml.runtime.instructions.CPInstructions.Data;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.utils.DMLRuntimeException;

public class FunctionObject {

	/*
	 * execute() methods related to ValueFunctions
	 */
	public double execute ( double in1, double in2 ) throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(double,double): should never get called in the base class");
	}
	
	public double execute ( double in1, int in2 ) throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(double,int): should never get called in the base class");
	}
	
	public double execute ( int in1, double in2 ) throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(int,double): should never get called in the base class");
	}
	
	public double execute ( int in1, int in2 )  throws DMLRuntimeException  {
		throw new DMLRuntimeException("FunctionObject.execute(int,int): should never get called in the base class");
	}
	
	public double execute ( double in )  throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(double): should never get called in the base class");
	}
	
	public double execute ( int in )  throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(int): should never get called in the base class");
	}
	
	public boolean execute ( boolean in1, boolean in2 )  throws DMLRuntimeException  {
		throw new DMLRuntimeException("FunctionObject.execute(int,int): should never get called in the base class");
	}
	
	public boolean execute ( boolean in )  throws DMLRuntimeException  {
		throw new DMLRuntimeException("FunctionObject.execute(int,int): should never get called in the base class");
	}
	
	// this version is for parameterized builtins with input parameters of form: name=value 
	public double execute ( HashMap<String,String> params )  throws DMLRuntimeException  {
		throw new DMLRuntimeException("FunctionObject.execute(int,int): should never get called in the base class");
	}
	
	
	/*
	 *  execute() methods related to IndexFunctions
	 */
	public void execute(MatrixIndexes in, MatrixIndexes out) throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(MatrixIndexes,MatrixIndexes): should never get called in the base class");
	}
	
	public void execute(CellIndex in, CellIndex out) throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(CellIndex,CellIndex): should never get called in the base class");
	}
	
	//return whether dimension has been reduced
	public boolean computeDimension(int row, int col, CellIndex retDim) throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(int,int,CellIndex): should never get called in the base class");
	}

	public boolean computeDimension(MatrixCharacteristics in, MatrixCharacteristics out) throws DMLRuntimeException{
		throw new DMLRuntimeException("computeDimension(MatrixCharacteristics in, MatrixCharacteristics out): should never get called in the base class");
	}
	
	/*
	 * execute() methods related to FileFunctions (rm, mv)
	 */
	public String execute ( String in1 ) throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(String): should never get called in the base class");
	}
	
	public String execute ( String in1, String in2 ) throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(String,String): should never get called in the base class");
	}
	
	/*
	 * compare() methods related to ValueFunctions (relational operators)
	 */
	public boolean compare(double in1, double in2) throws DMLRuntimeException {
		throw new DMLRuntimeException("compare(): should not be invoked from base class.");
	}

	public boolean compare(int in1, int in2) throws DMLRuntimeException {
		throw new DMLRuntimeException("compare(): should not be invoked from base class.");
	}

	public boolean compare(double in1, int in2) throws DMLRuntimeException {
		throw new DMLRuntimeException("compare(): should not be invoked from base class.");
	}

	public boolean compare(int in1, double in2) throws DMLRuntimeException {
		throw new DMLRuntimeException("compare(): should not be invoked from base class.");
	}

	public boolean compare(boolean in1, boolean in2) throws DMLRuntimeException {
		throw new DMLRuntimeException("compare(): should not be invoked from base class.");
	}

	
	/////////////////////////////////////////////////////////////////////////////////////
	/*
	 * For complext function object that operates on objects instead of native values 
	 */

	public Data execute(Data in1, double in2) throws DMLRuntimeException {
		throw new DMLRuntimeException("execute(): should not be invoked from base class.");
	}
	
	public Data execute(Data in1, double in2, double in3) throws DMLRuntimeException {
		throw new DMLRuntimeException("execute(): should not be invoked from base class.");
	}
	
	public Data execute(Data in1, Data in2) throws DMLRuntimeException {
		throw new DMLRuntimeException("execute(): should not be invoked from base class.");
	}

}
