/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import com.ibm.bi.dml.hops.globalopt.CrossBlockOp;
import com.ibm.bi.dml.lops.FunctionCallCP;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.CostEstimatorHops;
import com.ibm.bi.dml.sql.sqllops.SQLLops;

/**
 * This FunctionOp represents the call to a DML-bodied or external function.
 * 
 * Note: Currently, we support expressions in function arguments but no function calls
 * in expressions.
 */
public class FunctionOp extends Hop
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
		super(Kind.FunctionOp, fnamespace + Program.KEY_DELIM + fname, DataType.UNKNOWN, ValueType.UNKNOWN );
		
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
			//System.out.println("QREst " + (_memEstimate/1024/1024));
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
				long outputH = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).get_dim1(), getOutputs().get(0).get_dim2(), 0.5);
				long outputR = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).get_dim1(), getOutputs().get(1).get_dim2(), 0.5);
				//System.out.println("QROut " + (outputH+outputR)/1024/1024);
				return outputH+outputR; 
				
			}
			else if ( getFunctionName().equalsIgnoreCase("lu") ) {
				// upper-triangular and lower-triangular matrices
				long outputP = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).get_dim1(), getOutputs().get(1).get_dim2(), 1.0/getOutputs().get(1).get_dim2());
				long outputL = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).get_dim1(), getOutputs().get(0).get_dim2(), 0.5);
				long outputU = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).get_dim1(), getOutputs().get(1).get_dim2(), 0.5);
				//System.out.println("LUOut " + (outputL+outputU+outputP)/1024/1024);
				return outputL+outputU+outputP; 
				
			}
			else if ( getFunctionName().equalsIgnoreCase("eigen") ) {
				long outputVectors = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(0).get_dim1(), getOutputs().get(0).get_dim2(), 1.0);
				long outputValues = OptimizerUtils.estimateSizeExactSparsity(getOutputs().get(1).get_dim1(), 1, 1.0);
				//System.out.println("EigenOut " + (outputVectors+outputValues)/1024/1024);
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
				double interOutput = OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).get_dim1(), getInput().get(0).get_dim2(), 1.0); 
				//System.out.println("QRInter " + interOutput/1024/1024);
				return interOutput;
			}
			else if ( getFunctionName().equalsIgnoreCase("lu")) {
				// 1D vector 
				double interOutput = OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).get_dim1(), 1, 1.0); 
				//System.out.println("LUInter " + interOutput/1024/1024);
				return interOutput;
			}
			else if ( getFunctionName().equalsIgnoreCase("eigen")) {
				// One matrix of size original input and three 1D vectors (used to represent tridiagonal matrix)
				double interOutput = OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).get_dim1(), getInput().get(0).get_dim2(), 1.0) 
						+ 3*OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).get_dim1(), 1, 1.0); 
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
		if (get_lops() == null) {
			ExecType et = optFindExecType();
			
			if ( et != ExecType.CP ) {
				throw new HopsException("Invalid execution type for function: " + _fname);
			}
			//construct input lops (recursive)
			ArrayList<Lop> tmp = new ArrayList<Lop>();
			for( Hop in : getInput() )
				tmp.add( in.constructLops() );
			
			//construct function call
			FunctionCallCP fcall = new FunctionCallCP( tmp, _fnamespace, _fname, _outputs, _outputHops );
			set_lops( fcall );
		}
		
		return get_lops();
	}

	@Override
	public SQLLops constructSQLLOPs() 
		throws HopsException 
	{
		// TODO MB: @Shirish should we support function for SQL at all?
		return null;
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
		if ( getFunctionType() == FunctionType.MULTIRETURN_BUILTIN ) {
			// Since the memory estimate is only conservative, do not throw
			// exception if the estimated memory is larger than the budget
			// Nevertheless, memory estimates these functions are useful for 
			// other purposes, such as compiling parfor
			return ExecType.CP;
			
			// check if there is sufficient memory to execute this function
			/*if ( getMemEstimate() < OptimizerUtils.getMemBudget(true) ) {
				return ExecType.CP;
			}
			else {
				throw new HopsException("Insufficient memory to execute function: " + getFunctionName());
			}*/
		}
		// the actual function call is always CP
		return ExecType.CP;
	}

	@Override
	public void refreshSizeInformation()
	{
		//do nothing
	}
	
	@Override
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
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		return false;
	}
	
	
	/////////////////////////////////////////////////
	// Extension for Global Data Flow Optimization
	// by Mathias Peters
	///////
	
	/**
	 * FuncOps can have multiple outputs
	 */
	protected Map<String, CrossBlockOp> crossBlockOutputs = new HashMap<String, CrossBlockOp>();
	
	/**
	 * Create a meta edge that connects this hops with the target across statement blocks.
	 * @param target
	 */
	public void append(Hop target) {
		CrossBlockOp crossBlock = new CrossBlockOp(this, target);
		target.prepend(crossBlock);
	}
	
	@Override
	public void setCrossBlockOutput(CrossBlockOp crossBlockOutput) {
		this.crossBlockOutputs.put(crossBlockOutput.get_name(), crossBlockOutput);
		
	}
	
	public Collection<CrossBlockOp> getCrossBlockOutputs() {
		return this.crossBlockOutputs.values();
	}
}
