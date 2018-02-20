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

package org.apache.sysml.udf;

import java.util.ArrayList;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.cp.BooleanObject;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.DoubleObject;
import org.apache.sysml.runtime.instructions.cp.IntObject;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.instructions.cp.StringObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MetaDataFormat;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.udf.Matrix.ValueType;
import org.apache.sysml.udf.Scalar.ScalarValueType;

/**
 * Class to maintain external function invocation instructions.
 * 
 * 
 * 
 */
public class ExternalFunctionInvocationInstruction extends Instruction 
{
	private static final IDSequence _defaultSeq = new IDSequence();
	
	protected final CPOperand[] inputs;
	protected final CPOperand[] outputs;
	protected final PackageFunction fun;
	protected final String baseDir;
	protected final InputInfo iinfo;
	
	public ExternalFunctionInvocationInstruction(CPOperand[] inputs, CPOperand[] outputs,
		PackageFunction fun, String baseDir, InputInfo format) {
		this.inputs = inputs;
		this.outputs = outputs;
		this.fun = fun;
		this.baseDir = baseDir;
		this.iinfo = format;
	}

	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException 
	{
		// get the inputs, wrapped into external data types
		fun.setFunctionInputs(getInputObjects(inputs, ec.getVariables()));
		
		//executes function
		fun.execute(ec);
		
		// get and verify the outputs
		verifyAndAttachOutputs(ec, fun, outputs);
	}
	
	@SuppressWarnings("incomplete-switch")
	private ArrayList<FunctionParameter> getInputObjects(CPOperand[] inputs, LocalVariableMap vars) {
		ArrayList<FunctionParameter> ret = new ArrayList<>();
		for( CPOperand input : inputs ) {
			switch( input.getDataType() ) {
				case MATRIX:
					MatrixObject mobj = (MatrixObject) vars.get(input.getName());
					ret.add(new Matrix(mobj, getMatrixValueType(input.getValueType())));
					break;
				case SCALAR:
					ScalarObject so = (ScalarObject) vars.get(input.getName());
					ret.add(new Scalar(getScalarValueType(input.getValueType()), so.getStringValue()));
					break;
				case OBJECT:
					ret.add(new BinaryObject(vars.get(input.getName())));
					break;
			}
		}
		return ret;
	}
	
	private ScalarValueType getScalarValueType(Expression.ValueType vt) {
		switch(vt) {
			case STRING: return ScalarValueType.Text;
			case DOUBLE: return ScalarValueType.Double;
			case INT: return ScalarValueType.Integer;
			case BOOLEAN: return ScalarValueType.Boolean;
			default:
				throw new RuntimeException("Unknown type: "+vt.name());
		}
	}
	
	private ValueType getMatrixValueType(Expression.ValueType vt) {
		switch(vt) {
			case DOUBLE: return ValueType.Double;
			case INT: return ValueType.Integer;
			default:
				throw new RuntimeException("Unknown type: "+vt.name());
		}
	}
	
	private void verifyAndAttachOutputs(ExecutionContext ec, PackageFunction fun, CPOperand[] outputs) 
		throws DMLRuntimeException 
	{
		for( int i = 0; i < outputs.length; i++) {
			CPOperand output = outputs[i];
			switch( fun.getFunctionOutput(i).getType() ) {
				case Matrix:
					Matrix m = (Matrix) fun.getFunctionOutput(i);
					MatrixObject newVar = createOutputMatrixObject( m );
					ec.setVariable(output.getName(), newVar);
					break;
				case Scalar:
					Scalar s = (Scalar) fun.getFunctionOutput(i);
					ScalarObject scalarObject = null;
					switch( s.getScalarType() ) {
						case Integer:
							scalarObject = new IntObject(Long.parseLong(s.getValue()));
							break;
						case Double:
							scalarObject = new DoubleObject(Double.parseDouble(s.getValue()));
							break;
						case Boolean:
							scalarObject = new BooleanObject(Boolean.parseBoolean(s.getValue()));
							break;
						case Text:
							scalarObject = new StringObject(s.getValue());
							break;
						default:
							throw new DMLRuntimeException("Unknown scalar value type '"
								+ s.getScalarType()+"' of output '"+output.getName()+"'.");
					}
					ec.setVariable(output.getName(), scalarObject);
					break;
				default:
					throw new DMLRuntimeException("Unsupported data type: "
						+fun.getFunctionOutput(i).getType().name());
			}
		}
	}
	
	private MatrixObject createOutputMatrixObject(Matrix m) throws DMLRuntimeException
	{
		MatrixObject ret = m.getMatrixObject();
		
		if( ret == null ) { //otherwise, pass in-memory matrix from extfunct back to invoking program
			MatrixCharacteristics mc = new MatrixCharacteristics(m.getNumRows(),m.getNumCols(),
				ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize());
			MetaDataFormat mfmd = new MetaDataFormat(mc, InputInfo.getMatchingOutputInfo(iinfo), iinfo);
			ret = new MatrixObject(Expression.ValueType.DOUBLE, m.getFilePath(), mfmd);
		}
		
		//for allowing in-memory packagesupport matrices w/o file names
		if( ret.getFileName().equals( Matrix.DEFAULT_FILENAME ) ) {
			ret.setFileName( createDefaultOutputFilePathAndName() );
		}
		
		return ret;
	}
	
	private String createDefaultOutputFilePathAndName( ) {
		StringBuilder sb = new StringBuilder();
		sb.append(baseDir);
		sb.append(Matrix.DEFAULT_FILENAME);
		sb.append(_defaultSeq.getNextID());
		return sb.toString();
	}
}
