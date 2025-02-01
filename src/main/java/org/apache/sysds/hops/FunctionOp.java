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

package org.apache.sysds.hops;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.lops.Compression;
import org.apache.sysds.lops.FunctionCallCP;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.compress.SingletonLookupHashMap;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.parfor.opt.CostEstimatorHops;
import org.apache.sysds.runtime.meta.DataCharacteristics;

/**
 * This FunctionOp represents the call to a DML-bodied or external function.
 * 
 * Note: Currently, we support expressions in function arguments along with function calls
 * in expressions with single outputs, leaving multiple outputs handling as it is.
 */
public class FunctionOp extends MultiThreadedHop
{
	public enum FunctionType{
		DML,
		MULTIRETURN_BUILTIN,
		UNKNOWN
	}
	
	public static final String OPCODE = "fcall";
	
	private FunctionType _type = null;
	private String _fnamespace = null;
	private String _fname = null;
	private boolean _opt = true; //call to optimized/unoptimized
	private boolean _pseudo = false;
	
	private String[] _inputNames = null;  // A,B in C = foo(A=X, B=Y)
	private String[] _outputNames = null; // C in C = foo(A=X, B=Y)
	private ArrayList<Hop> _outputHops = null;
	
	private FunctionOp() {
		//default constructor for clone
	}

	public FunctionOp(FunctionType type, String fnamespace, String fname, String[] inputNames, List<Hop> inputs, String[] outputNames, ArrayList<Hop> outputHops) {
		this(type, fnamespace, fname, inputNames, inputs, outputNames, false);
		_outputHops = outputHops;
	}
	public FunctionOp(FunctionType type, String fnamespace, String fname, String[] inputNames, List<Hop> inputs, String[] outputNames, boolean singleOut) {
		this(type, fnamespace, fname, inputNames, inputs, outputNames, singleOut, false);
	}
	
	public FunctionOp(FunctionType type, String fnamespace, String fname, String[] inputNames, List<Hop> inputs, String[] outputNames, boolean singleOut, boolean pseudo) 
	{
		super(fnamespace + Program.KEY_DELIM + fname, DataType.UNKNOWN, ValueType.UNKNOWN );
		
		_type = type;
		_fnamespace = fnamespace;
		_fname = fname;
		_inputNames = inputNames;
		_outputNames = outputNames;
		_pseudo = pseudo;
		
		for( Hop in : inputs ) {
			getInput().add(in);
			in.getParent().add(this);
		}
	}

	public String getFunctionKey() {
		return DMLProgram.constructFunctionKey(
			getFunctionNamespace(), getFunctionName());
	}
	
	public String getFunctionNamespace() {
		return _fnamespace;
	}
	
	public String getFunctionName() {
		return _fname;
	}
	
	public void setFunctionName( String fname ) {
		_fname = fname;
	}
	
	public void setFunctionNamespace( String fnamespace ) {
		_fnamespace = fnamespace;
	}
	
	public void setInputVariableNames(String[] names) {
		_inputNames = names;
	}
	
	public ArrayList<Hop> getOutputs() {
		return _outputHops;
	}
	
	public String[] getInputVariableNames() {
		return _inputNames;
	}
	
	public String[] getOutputVariableNames() {
		return _outputNames;
	}
	
	public boolean containsOutput(String varname) {
		return Arrays.stream(getOutputVariableNames())
			.anyMatch(outName -> outName.equals(varname));
	}
	
	public FunctionType getFunctionType() {
		return _type;
	}
	
	public void setCallOptimized(boolean opt) {
		_opt = opt;
	}
	
	public boolean isPseudoFunctionCall() {
		return _pseudo;
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
		else if( _type == FunctionType.UNKNOWN )
			_memEstimate = CostEstimatorHops.DEFAULT_MEM_SP;
		else if ( _type == FunctionType.MULTIRETURN_BUILTIN ) {
			boolean outputDimsKnown = true;
			for(Hop out : getOutputs()){
				outputDimsKnown &= out.dimsKnown();
			}
			if( outputDimsKnown ) { 
				long lnnz = (getNnz()>=0)?getNnz():getLength(); 
				_outputMemEstimate = computeOutputMemEstimate(getDim1(), getDim2(), lnnz);
				_processingMemEstimate = computeIntermediateMemEstimate(getDim1(), getDim2(), lnnz);
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
			if ( getFunctionName().equalsIgnoreCase(Opcodes.QR.toString()) ) {
				// upper-triangular and lower-triangular matrices
				long outputH = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 0.5);
				long outputR = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), getOutputs().get(1).getDim2(), 0.5);
				return outputH+outputR; 
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.LU.toString()) ) {
				// upper-triangular and lower-triangular matrices
				long outputP = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), getOutputs().get(1).getDim2(), 1.0/getOutputs().get(1).getDim2());
				long outputL = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 0.5);
				long outputU = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), getOutputs().get(1).getDim2(), 0.5);
				return outputL+outputU+outputP; 
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.EIGEN.toString()) ) {
				long outputVectors = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 1.0);
				long outputValues = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), 1, 1.0);
				return outputVectors+outputValues; 
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.FFT.toString()) ) {
				long outputRe = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 1.0);
				long outputIm = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), getOutputs().get(1).getDim2(), 1.0);
				return outputRe+outputIm;
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.IFFT.toString()) ) {
				long outputRe = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 1.0);
				long outputIm = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), getOutputs().get(1).getDim2(), 1.0);
				return outputRe+outputIm;
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.FFT_LINEARIZED.toString()) ) {
				long outputRe = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 1.0);
				long outputIm = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), getOutputs().get(1).getDim2(), 1.0);
				return outputRe+outputIm;
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.IFFT_LINEARIZED.toString()) ) {
				long outputRe = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 1.0);
				long outputIm = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), getOutputs().get(1).getDim2(), 1.0);
				return outputRe+outputIm;
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.STFT.toString()) ) {
				long outputRe = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 1.0);
				long outputIm = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), getOutputs().get(1).getDim2(), 1.0);
				return outputRe+outputIm;
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.LSTM.toString()) || getFunctionName().equalsIgnoreCase(Opcodes.LSTM_BACKWARD.toString()) ) {
				// TODO: To allow for initial version to always run on the GPU
				return 0; 
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.BATCH_NORM2D.toString()) || getFunctionName().equalsIgnoreCase("batch_norm2d_train")) {
				return OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 1.0) +
						OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), getOutputs().get(1).getDim2(), 1.0) +
						OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(2).getDim1(), getOutputs().get(2).getDim2(), 1.0) +
						OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(3).getDim1(), getOutputs().get(3).getDim2(), 1.0) + 
						OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(4).getDim1(), getOutputs().get(4).getDim2(), 1.0);
			}
			else if ( getFunctionName().equalsIgnoreCase("batch_norm2d_test") ) {
			return OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 1.0);
		}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.BATCH_NORM2D_BACKWARD.toString()) ) {
				return OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 1.0) +
						OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), getOutputs().get(1).getDim2(), 1.0) +
						OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(2).getDim1(), getOutputs().get(2).getDim2(), 1.0);
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.SVD.toString()) ) {
				long outputU = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).getDim1(), getOutputs().get(0).getDim2(), 1.0);
				long outputSigma = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).getDim1(), getOutputs().get(1).getDim2(), 1.0);
				long outputV = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(2).getDim1(), getOutputs().get(2).getDim2(), 1.0);
				return outputU+outputSigma+outputV;
			}
			else if( getFunctionName().equalsIgnoreCase(Opcodes.RCM.toString()) ) {
				long nr = Math.max(getInput(0).getDim1(), getInput(1).getDim1());
				long nc = Math.max(getInput(0).getDim2(), getInput(1).getDim2());
				return 2*OptimizerUtils.estimateSizeExactSparsity(nr, nc, 1.0); 
			}
			else
				throw new RuntimeException("Invalid call of computeOutputMemEstimate in FunctionOp: "+getFunctionName());
		}
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		if ( getFunctionType() != FunctionType.MULTIRETURN_BUILTIN )
			throw new RuntimeException("Invalid call of computeIntermediateMemEstimate in FunctionOp.");
		else {
			if ( getFunctionName().equalsIgnoreCase(Opcodes.QR.toString()) ) {
				// matrix of size same as the input
				return OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 1.0); 
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.LU.toString())) {
				// 1D vector 
				return OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).getDim1(), 1, 1.0); 
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.EIGEN.toString())) {
				// One matrix of size original input and three 1D vectors (used to represent tridiagonal matrix)
				return OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 1.0) 
						+ 3*OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).getDim1(), 1, 1.0); 
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.FFT.toString()) ) {
				// 2 matrices of size same as the input
				return 2*OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 1.0);
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.IFFT.toString()) ) {
				// 2 matrices of size same as the input
				return 2*OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 1.0);
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.FFT_LINEARIZED.toString()) ) {
				// 2 matrices of size same as the input
				return 2*OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 1.0);
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.IFFT_LINEARIZED.toString()) ) {
				// 2 matrices of size same as the input
				return 2*OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 1.0);
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.STFT.toString()) ) {
				// 2 matrices of size same as the input
				return 2*OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 1.0);
			}
			else if (getFunctionName().equalsIgnoreCase(Opcodes.BATCH_NORM2D.toString()) || getFunctionName().equalsIgnoreCase(Opcodes.BATCH_NORM2D_BACKWARD.toString()) ||
					getFunctionName().equalsIgnoreCase("batch_norm2d_train") || getFunctionName().equalsIgnoreCase("batch_norm2d_test")) {
				return 0; 
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.LSTM.toString())
					||  getFunctionName().equalsIgnoreCase(Opcodes.LSTM_BACKWARD.toString())
					|| getFunctionName().equalsIgnoreCase(Opcodes.RCM.toString())) {
				// TODO: To allow for initial version to always run on the GPU
				return 0; 
			}
			else if ( getFunctionName().equalsIgnoreCase(Opcodes.SVD.toString())) {
				double interOutput = OptimizerUtils.estimateSizeExactSparsity(1, getInput().get(0).getDim2(), 1.0);
				return interOutput;
			}
			else
				throw new RuntimeException("Invalid call of computeIntermediateMemEstimate in FunctionOp.");
		}
	}
	
	@Override
	protected DataCharacteristics inferOutputCharacteristics( MemoTable memo ) {
		throw new RuntimeException("Invalid call of inferOutputCharacteristics in FunctionOp.");
	}
	
	@Override
	public boolean isGPUEnabled() {
		if(getFunctionName().equalsIgnoreCase(Opcodes.LSTM.toString()) || getFunctionName().equalsIgnoreCase(Opcodes.LSTM_BACKWARD.toString()) ||
			getFunctionName().equalsIgnoreCase(Opcodes.BATCH_NORM2D.toString()) || getFunctionName().equalsIgnoreCase(Opcodes.BATCH_NORM2D_BACKWARD.toString()) ||
			getFunctionName().equalsIgnoreCase("batch_norm2d_train") || getFunctionName().equalsIgnoreCase("batch_norm2d_test")) 
			return true;
		else
			return false;
	}
	
	@Override
	public Lop constructLops() 
	{
		//return already created lops
		if( getLops() != null )
			return getLops();
		
		ExecType et = optFindExecType();
		
		//construct input lops (recursive)
		ArrayList<Lop> tmp = new ArrayList<>();
		for( Hop in : getInput() )
			tmp.add( in.constructLops() );
		
		//construct function call
		final FunctionCallCP fcall;
		if(isMultiThreadedOpType()) {
			fcall = new FunctionCallCP(tmp, _fnamespace, _fname, _inputNames, _outputNames, _outputHops, _opt, et,
				OptimizerUtils.getConstrainedNumThreads(_maxNumThreads));
		}
		else {
			fcall = new FunctionCallCP(tmp, _fnamespace, _fname, _inputNames, _outputNames, _outputHops, _opt, et);
		}
		setLineNumbers(fcall);
		setLops(fcall);
		
		//note: no reblock lop because outputs directly bound
		constructAndSetCompressionLopFunctionalIfRequired(et);

		return getLops();
	}

	protected void constructAndSetCompressionLopFunctionalIfRequired(ExecType et) {
		if((requiresCompression()) && ((FunctionCallCP) getLops()).getFunctionName().equalsIgnoreCase(Opcodes.TRANSFORMENCODE.toString())){ // xor
			
			// Lop matrixOut = lop.getFunctionOutputs().get(0);
			Lop compressionInstruction = null;
		
			final int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
			if(_compressedWorkloadTree != null) {
				SingletonLookupHashMap m = SingletonLookupHashMap.getMap();
				int singletonID = m.put(_compressedWorkloadTree);
				compressionInstruction = new Compression(getLops(), DataType.MATRIX, ValueType.FP64, et, singletonID, k);
			}
			else
				compressionInstruction = new Compression(getLops(), DataType.MATRIX, ValueType.FP64, et, 0, k);
			

			setOutputDimensions( compressionInstruction );
			setLineNumbers( compressionInstruction );
			setLops( compressionInstruction );

		}
	}


	@Override
	public String getOpString() {
		return OPCODE + " " + _fnamespace  + " " + _fname;
	}

	@Override
	protected ExecType optFindExecType(boolean transitive) 
	{
		checkAndSetForcedPlatform();
		
		if ( getFunctionType() == FunctionType.MULTIRETURN_BUILTIN ) {
			boolean isBuiltinFunction = isBuiltinFunction();
			// check if there is sufficient memory to execute this function
			double mem = getMemEstimate(); // obtain memory estimates for all (e.g., used in parfor)
			if(isBuiltinFunction && getFunctionName().equalsIgnoreCase(Opcodes.TRANSFORMENCODE.toString()) ) {
				_etype = ((_etypeForced==ExecType.SPARK 
					|| (mem >= OptimizerUtils.getLocalMemBudget() && OptimizerUtils.isSparkExecutionMode())) ? 
					ExecType.SPARK : ExecType.CP);
			}
			else if(isBuiltinFunction && (getFunctionName().equalsIgnoreCase(Opcodes.LSTM.toString()) || getFunctionName().equalsIgnoreCase(Opcodes.LSTM_BACKWARD.toString()))) {
				_etype = DMLScript.USE_ACCELERATOR ? ExecType.GPU : ExecType.CP;
			}
			else if(isBuiltinFunction && (getFunctionName().equalsIgnoreCase(Opcodes.BATCH_NORM2D.toString()) || getFunctionName().equalsIgnoreCase(Opcodes.BATCH_NORM2D_BACKWARD.toString()))) {
				_etype = DMLScript.USE_ACCELERATOR ? ExecType.GPU : ExecType.CP;
			}
			else if(isBuiltinFunction && getFunctionName().equalsIgnoreCase("batch_norm2d_train")) {
				// Only GPU implementation is supported
				_etype = ExecType.GPU;
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
	
	private boolean isBuiltinFunction() {
		return getFunctionNamespace().equals(DMLProgram.INTERNAL_NAMESPACE);
	}

	@Override
	public void refreshSizeInformation() {
		//do nothing
	}

	@Override
	public boolean isMultiThreadedOpType() {
		return isBuiltinFunction();
	}
	
	@Override
	@SuppressWarnings("unchecked")
	public Object clone() throws CloneNotSupportedException {
		FunctionOp ret = new FunctionOp();
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._type = _type;
		ret._fnamespace = _fnamespace;
		ret._fname = _fname;
		ret._opt = _opt;
		ret._inputNames = (_inputNames!=null) ? _inputNames.clone() : null;
		ret._outputNames = _outputNames.clone();
		if( _outputHops != null )
			ret._outputHops = (ArrayList<Hop>) _outputHops.clone();
		
		return ret;
	}
	
	@Override
	public boolean compare(Hop that) {
		return false;
	}

}
