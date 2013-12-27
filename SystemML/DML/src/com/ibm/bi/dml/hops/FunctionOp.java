/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import java.util.ArrayList;

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
	}
	
	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		throw new RuntimeException("Invalid call of computeOutputMemEstimate in FunctionOp.");
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		throw new RuntimeException("Invalid call of computeIntermediateMemEstimate in FunctionOp.");
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
}
