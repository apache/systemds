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
import org.apache.sysml.runtime.compress.CompressedMatrixBlock;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.UtilFunctions;

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
	
	public abstract String getSpoofType(); 
	
	public ScalarObject execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalars) throws DMLRuntimeException {
		throw new RuntimeException("Invalid invocation in base class.");
	}
	
	public ScalarObject execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalars, int k) 
		throws DMLRuntimeException 
	{
		//default implementation serial execution
		return execute(inputs, scalars);
	}
	
	protected double[][] prepInputMatricesDense(ArrayList<MatrixBlock> inputs) 
		throws DMLRuntimeException 
	{
		return prepInputMatricesDense(inputs, 1, inputs.size()-1);
	}
	
	protected double[][] prepInputMatricesDense(ArrayList<MatrixBlock> inputs, int offset) 
		throws DMLRuntimeException 
	{
		return prepInputMatricesDense(inputs, offset, inputs.size()-offset);
	}
	
	protected double[][] prepInputMatricesDense(ArrayList<MatrixBlock> inputs, int offset, int len) 
		throws DMLRuntimeException 
	{
		double[][] b = new double[len][]; 
		for(int i=offset; i<offset+len; i++) {
			if( inputs.get(i) instanceof CompressedMatrixBlock ) 
				inputs.set(i, ((CompressedMatrixBlock)inputs.get(i)).decompress());
			
			//convert empty or sparse to dense temporary block (note: we don't do
			//this in place because this block might be used by multiple threads)
			if( inputs.get(i).isInSparseFormat() && inputs.get(i).isAllocated() ) {
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
	
	protected SideInput[] prepInputMatricesAbstract(ArrayList<MatrixBlock> inputs) 
		throws DMLRuntimeException 
	{
		return prepInputMatricesAbstract(inputs, 1, inputs.size()-1);
	}
	
	protected SideInput[] prepInputMatricesAbstract(ArrayList<MatrixBlock> inputs, int offset) 
		throws DMLRuntimeException 
	{
		return prepInputMatricesAbstract(inputs, offset, inputs.size()-offset);
	}
	
	protected SideInput[] prepInputMatricesAbstract(ArrayList<MatrixBlock> inputs, int offset, int len) 
		throws DMLRuntimeException 
	{
		SideInput[] b = new SideInput[len]; 
		for(int i=offset; i<offset+len; i++) {
			if( inputs.get(i) instanceof CompressedMatrixBlock ) 
				inputs.set(i, ((CompressedMatrixBlock)inputs.get(i)).decompress());
			
			if( inputs.get(i).isInSparseFormat() && inputs.get(i).isAllocated() )
				b[i-offset] = new SideInput(null, inputs.get(i));
			else
				b[i-offset] = new SideInput(inputs.get(i).getDenseBlock(), null);
		}
		
		return b;
	}
	
	protected double[] prepInputScalars(ArrayList<ScalarObject> scalarObjects) {
		double[] scalars = new double[scalarObjects.size()]; 
		for(int i=0; i < scalarObjects.size(); i++)
			scalars[i] = scalarObjects.get(i).getDoubleValue();
		return scalars;
	}
	
	//abstraction for safely accessing sideways matrices without the need 
	//to allocate empty matrices as dense, see prepInputMatrices
	
	protected static double getValue(double[] data, double index) {
		int iindex = UtilFunctions.toInt(index);
		return getValue(data, iindex);
	}
	
	protected static double getValue(double[] data, int index) {
		return (data!=null) ? data[index] : 0;
	}
	
	protected static double getValue(double[] data, int n, double rowIndex, double colIndex) {
		int irowIndex = UtilFunctions.toInt(rowIndex);
		int icolIndex = UtilFunctions.toInt(colIndex);
		return getValue(data, n, irowIndex, icolIndex);
	}
	
	protected static double getValue(double[] data, int n, int rowIndex, int colIndex) {
		return (data!=null) ? data[rowIndex*n+colIndex] : 0;
	}
	
	protected static double getValue(SideInput data, double rowIndex) {
		int irowIndex = UtilFunctions.toInt(rowIndex);
		return getValue(data, irowIndex);
	}
	
	protected static double getValue(SideInput data, int rowIndex) {
		//note: wrapper sideinput guaranteed to exist
		return (data.dBlock!=null) ? data.dBlock[rowIndex] : 
			(data.mBlock!=null) ? data.mBlock.quickGetValue(rowIndex, 0) : 0;
	}
	
	protected static double getValue(SideInput data, int n, double rowIndex, double colIndex) {
		int irowIndex = UtilFunctions.toInt(rowIndex);
		int icolIndex = UtilFunctions.toInt(colIndex);
		return getValue(data, n, irowIndex, icolIndex);
	}
	
	protected static double getValue(SideInput data, int n, int rowIndex, int colIndex) {
		//note: wrapper sideinput guaranteed to exist
		return (data.dBlock!=null) ? data.dBlock[rowIndex*n+colIndex] : 
			(data.mBlock!=null) ? data.mBlock.quickGetValue(rowIndex, colIndex) : 0;
	}
	
	public static class SideInput {
		private final double[] dBlock;
		private final MatrixBlock mBlock;
	
		public SideInput(double[] ddata, MatrixBlock mdata) {
			dBlock = ddata;
			mBlock = mdata;
		}
	}
}
