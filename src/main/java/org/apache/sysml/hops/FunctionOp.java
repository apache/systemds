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

package org.apache.sysml.hops;

import java.util.ArrayList;

import org.apache.sysml.lops.FunctionCallCP;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.parfor.opt.CostEstimatorHops;

/**
 * This FunctionOp represents the call to a DML-bodied or external function.
 * 
 * Note: Currently, we support expressions in function arguments but no function calls
 * in expressions.
 */
public class FunctionOp extends Hop
{
	
	public static String OPSTRING = "extfunct";
	
	public enum FunctionType{
		DML,
		EXTERNAL_MEM,
		EXTERNAL_FILE,
		MULTIRETURN_BUILTIN,
		UNKNOWN
	}
	
	private FunctionType _type = null;
	private String _fnamespace = null;
	private String _fname = null; 
	private String[] _outputs = null; 
	private ArrayList<Hop> _outputHops = null;
	
	private FunctionOp() {
		//default constructor for clone
	}

	public FunctionOp(FunctionType type, String fnamespace, String fname, ArrayList<Hop> finputs, String[] outputs, ArrayList<Hop> outputHops) {
		this(type, fnamespace, fname, finputs, outputs);
		_outputHops = outputHops;
	}

	public FunctionOp(FunctionType type, String fnamespace, String fname, ArrayList<Hop> finputs, String[] outputs) 
	{
		super(fnamespace + Program.KEY_DELIM + fname, DataType.UNKNOWN, ValueType.UNKNOWN );
		
		_type = type;
		_fnamespace = fnamespace;
		_fname = fname;
		_outputs = outputs;
		
		for( Hop in : finputs )
		{			
			getInput().add(in);
			in.getParent().add(this);
		}
	}
	
	public String getFunctionNamespace()
	{
		return _fnamespace;
	}
	
	public String getFunctionName()
	{
		return _fname;
	}
	
	public void setFunctionName( String fname )
	{
		_fname = fname;
	}
	
	public ArrayList<Hop> getOutputs() {
		return _outputHops;
	}
	
	public String[] getOutputVariableNames()
	{
		return _outputs;
	}
	
	public FunctionType getFunctionType()
	{
		return _type;
	}

	@Override
	public boolean allowsAllExecTypes() {
		return false;
	}

	@Override
	public void computeMemEstimate( MemoTable memo ) 
	{
		//overwrites default hops behavior
		
		if( _type == FunctionType.DML )
			_memEstimate = 1; //minimal mem estimate
		else if( _type == FunctionType.EXTERNAL_MEM )
			_memEstimate = 2* getInputSize(); //in/out
		else if(    _type == FunctionType.EXTERNAL_FILE || _type == FunctionType.UNKNOWN )
			_memEstimate = CostEstimatorHops.DEFAULT_MEM_MR;
		else if ( _type == FunctionType.MULTIRETURN_BUILTIN ) {
			boolean outputDimsKnown = true;
			for(Hop out : getOutputs()){
				outputDimsKnown &= out.dimsKnown();
			}
			if( outputDimsKnown ) { 
				long lnnz = ((_nnz>=0)?_nnz:_dim1*_dim2); 
				_outputMemEstimate = computeOutputMemEstimate( _dim1, _dim2, lnnz );
				_processingMemEstimate = computeIntermediateMemEstimate(_dim1, _dim2, lnnz);
			}
			_memEstimate = getInputOutputSize();
		}
	}
	
	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		if ( getFunctionType() != FunctionType.MULTIRETURN_BUILTIN )
			throw new RuntimeException("Invalid call of computeOutputMemEstimate in FunctionOp.");
		else {
			if ( getFunctionName().equalsIgnoreCase("qr") ) {
				// upper-triangular and lower-triangular matrices
				long outputH = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 0.5);
				long outputR = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), getOutputs().get(1).getDim2(), 0.5);
				return outputH+outputR; 
			}
			else if ( getFunctionName().equalsIgnoreCase("lu") ) {
				// upper-triangular and lower-triangular matrices
				long outputP = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), getOutputs().get(1).getDim2(), 1.0/getOutputs().get(1).getDim2());
				long outputL = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 0.5);
				long outputU = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), getOutputs().get(1).getDim2(), 0.5);
				return outputL+outputU+outputP; 
			}
			else if ( getFunctionName().equalsIgnoreCase("eigen") ) {
				long outputVectors = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 1.0);
				long outputValues = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), 1, 1.0);
				return outputVectors+outputValues; 
			}
			else
				throw new RuntimeException("Invalid call of computeOutputMemEstimate in FunctionOp.");
		}
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		if ( getFunctionType() != FunctionType.MULTIRETURN_BUILTIN )
			throw new RuntimeException("Invalid call of computeIntermediateMemEstimate in FunctionOp.");
		else {
			if ( getFunctionName().equalsIgnoreCase("qr") ) {
				// matrix of size same as the input
				double interOutput = OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 1.0); 
				//System.out.println("QRInter " + interOutput/1024/1024);
				return interOutput;
			}
			else if ( getFunctionName().equalsIgnoreCase("lu")) {
				// 1D vector 
				double interOutput = OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).getDim1(), 1, 1.0); 
				//System.out.println("LUInter " + interOutput/1024/1024);
				return interOutput;
			}
			else if ( getFunctionName().equalsIgnoreCase("eigen")) {
				// One matrix of size original input and three 1D vectors (used to represent tridiagonal matrix)
				double interOutput = OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 1.0) 
						+ 3*OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).getDim1(), 1, 1.0); 
				//System.out.println("EigenInter " + interOutput/1024/1024);
				return interOutput;
			}
			else
				throw new RuntimeException("Invalid call of computeIntermediateMemEstimate in FunctionOp.");
		}
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		throw new RuntimeException("Invalid call of inferOutputCharacteristics in FunctionOp.");
	}
	
	@Override
	public Lop constructLops() 
		throws HopsException, LopsException 
	{
		//return already created lops
		if( getLops() != null )
			return getLops();

		ExecType et = optFindExecType();
		
		//construct input lops (recursive)
		ArrayList<Lop> tmp = new ArrayList<Lop>();
		for( Hop in : getInput() )
			tmp.add( in.constructLops() );
		
		//construct function call
		FunctionCallCP fcall = new FunctionCallCP( tmp, _fnamespace, _fname, _outputs, _outputHops, et );
		setLineNumbers( fcall );
		setLops( fcall );
	
		//note: no reblock lop because outputs directly bound
		
		return getLops();
	}

	@Override
	public String getOpString() 
	{
		return OPSTRING;
	}

	@Override
	protected ExecType optFindExecType() 
		throws HopsException 
	{
		checkAndSetForcedPlatform();
		
		if ( getFunctionType() == FunctionType.MULTIRETURN_BUILTIN ) {
			
			// check if there is sufficient memory to execute this function
			if( getFunctionName().equalsIgnoreCase("transformencode") ) {
				_etype = ((_etypeForced==ExecType.SPARK 
					|| (getMemEstimate() >= OptimizerUtils.getLocalMemBudget()
						&& OptimizerUtils.isSparkExecutionMode())) ? ExecType.SPARK : ExecType.CP);
			}	
			else {
				// Since the memory estimate is only conservative, do not throw
				// exception if the estimated memory is larger than the budget
				// Nevertheless, memory estimates these functions are useful for 
				// other purposes, such as compiling parfor
				_etype = ExecType.CP;
			}
		}
		else {
			// the actual function call is always CP
			_etype = ExecType.CP;
		}
		
		return _etype;
	}

	@Override
	public void refreshSizeInformation()
	{
		//do nothing
	}
	
	@Override
	@SuppressWarnings("unchecked")
	public Object clone() throws CloneNotSupportedException 
	{
		FunctionOp ret = new FunctionOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._type = _type;
		ret._fnamespace = _fnamespace;
		ret._fname = _fname;
		ret._outputs = _outputs.clone();
		if( _outputHops != null )
			ret._outputHops = (ArrayList<Hop>) _outputHops.clone();
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		return false;
	}
}
