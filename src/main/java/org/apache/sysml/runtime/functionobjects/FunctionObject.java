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

package org.apache.sysml.runtime.functionobjects;

import java.util.HashMap;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;


public class FunctionObject 
{
	
	/*
	 * execute() methods related to ValueFunctions
	 */
	public double execute ( double in1, double in2 ) throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(double,double): should never get called in the base class");
	}
	
	public double execute ( double in1, long in2 ) throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(double,int): should never get called in the base class");
	}
	
	public double execute ( long in1, double in2 ) throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(int,double): should never get called in the base class");
	}
	
	public double execute ( long in1, long in2 )  throws DMLRuntimeException  {
		throw new DMLRuntimeException("FunctionObject.execute(int,int): should never get called in the base class");
	}
	
	public double execute ( double in )  throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(double): should never get called in the base class");
	}
	
	public double execute ( long in )  throws DMLRuntimeException {
		throw new DMLRuntimeException("FunctionObject.execute(int): should never get called in the base class");
	}
	
	public boolean execute ( boolean in1, boolean in2 )  throws DMLRuntimeException  {
		throw new DMLRuntimeException("FunctionObject.execute(boolean,boolean): should never get called in the base class");
	}
	
	public boolean execute ( boolean in )  throws DMLRuntimeException  {
		throw new DMLRuntimeException("FunctionObject.execute(boolean): should never get called in the base class");
	}
	
	// this version is for parameterized builtins with input parameters of form: name=value 
	public double execute ( HashMap<String,String> params )  throws DMLRuntimeException  {
		throw new DMLRuntimeException("FunctionObject.execute(HashMap<String,String> params): should never get called in the base class");
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

	public boolean compare(long in1, long in2) throws DMLRuntimeException {
		throw new DMLRuntimeException("compare(): should not be invoked from base class.");
	}

	public boolean compare(double in1, long in2) throws DMLRuntimeException {
		throw new DMLRuntimeException("compare(): should not be invoked from base class.");
	}

	public boolean compare(long in1, double in2) throws DMLRuntimeException {
		throw new DMLRuntimeException("compare(): should not be invoked from base class.");
	}

	public boolean compare(boolean in1, boolean in2) throws DMLRuntimeException {
		throw new DMLRuntimeException("compare(): should not be invoked from base class.");
	}
	
	public boolean compare(String in1, String in2) throws DMLRuntimeException {
		throw new DMLRuntimeException("compare(): should not be invoked from base class.");
	}

	
	/////////////////////////////////////////////////////////////////////////////////////
	/*
	 * For complex function object that operates on objects instead of native values 
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
