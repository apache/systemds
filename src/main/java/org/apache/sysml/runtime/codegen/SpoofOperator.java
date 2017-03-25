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

package org.apache.sysml.runtime.codegen;

import java.io.Serializable;
import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;

public abstract class SpoofOperator implements Serializable
{
	private static final long serialVersionUID = 3834006998853573319L;
	private static final Log LOG = LogFactory.getLog(SpoofOperator.class.getName());
	
	public abstract void execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalars, MatrixBlock out) 
		throws DMLRuntimeException;
	
	public void execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalars, MatrixBlock out, int k) 
		throws DMLRuntimeException 
	{
		//default implementation serial execution
		execute(inputs, scalars, out);
	}
	
	public ScalarObject execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalars) throws DMLRuntimeException {
		throw new RuntimeException("Invalid invocation in base class.");
	}
	
	public ScalarObject execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalars, int k) 
		throws DMLRuntimeException 
	{
		//default implementation serial execution
		return execute(inputs, scalars);
	}
	
	protected double[][] prepInputMatrices(ArrayList<MatrixBlock> inputs) {
		return prepInputMatrices(inputs, 1, inputs.size()-1);
	}
	
	protected double[][] prepInputMatrices(ArrayList<MatrixBlock> inputs, int offset) {
		return prepInputMatrices(inputs, offset, inputs.size()-offset);
	}
	
	protected double[][] prepInputMatrices(ArrayList<MatrixBlock> inputs, int offset, int len) {
		double[][] b = new double[len][]; 
		for(int i=offset; i<offset+len; i++) {
			//convert empty or sparse to dense temporary block (note: we don't do
			//this in place because this block might be used by multiple threads)
			if( (inputs.get(i).isEmptyBlock(false) && !inputs.get(i).isAllocated())
				|| inputs.get(i).isInSparseFormat() ) {
				MatrixBlock tmp = inputs.get(i);
				b[i-offset] = DataConverter.convertToDoubleVector(tmp);
				LOG.warn(getClass().getName()+": Converted "+tmp.getNumRows()+"x"+tmp.getNumColumns()+
						", nnz="+tmp.getNonZeros()+" sideways input matrix from sparse to dense.");
			}
			//use existing dense block
			else {
				b[i-offset] = inputs.get(i).getDenseBlock();
			}
		}
		
		return b;
	}
	
	protected double[] prepInputScalars(ArrayList<ScalarObject> scalarObjects) {
		double[] scalars = new double[scalarObjects.size()]; 
		for(int i=0; i < scalarObjects.size(); i++)
			scalars[i] = scalarObjects.get(i).getDoubleValue();
		return scalars;
	}
}
