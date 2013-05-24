package com.ibm.bi.dml.hops;

import java.util.ArrayList;

import com.ibm.bi.dml.lops.FunctionCallCP;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.CostEstimatorHops;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LopsException;

/**
 * This FunctionOp represents the call to a DML-bodied or external function.
 * 
 * Note: Currently, we support expressions in function arguments but no function calls
 * in expressions.
 */
public class FunctionOp extends Hops
{
	public static String OPSTRING = "extfunct";
	
	public enum FunctionType{
		DML,
		EXTERNAL_MEM,
		EXTERNAL_FILE,
		UNKNOWN
	}
	
	private FunctionType _type = null;
	private String _fnamespace = null;
	private String _fname = null; 
	private String[] _outputs = null; 
	
	private FunctionOp() {
		//default constructor for clone
	}
	
	public FunctionOp(FunctionType type, String fnamespace, String fname, ArrayList<Hops> finputs, String[] outputs) 
	{
		super(Kind.FunctionOp, fnamespace + Program.KEY_DELIM + fname, DataType.UNKNOWN, ValueType.UNKNOWN );
		
		_type = type;
		_fnamespace = fnamespace;
		_fname = fname;
		_outputs = outputs;
		
		for( Hops in : finputs )
		{			
			getInput().add(in);
			in.getParent().add(this);
		}
	}
	
	public String getFunctionName()
	{
		return _fname;
	}
	
	public void setFunctionName( String fname )
	{
		_fname = fname;
	}
	

	@Override
	public boolean allowsAllExecTypes() {
		return false;
	}

	@Override
	public double computeMemEstimate() 
	{
		if( _type == FunctionType.DML )
			_memEstimate = 0;
		else if( _type == FunctionType.EXTERNAL_MEM )
			_memEstimate = getInputSize();
		else if(    _type == FunctionType.EXTERNAL_FILE || _type == FunctionType.UNKNOWN )
			_memEstimate = CostEstimatorHops.DEFAULT_MEM_MR;
		
		return _memEstimate;
	}

	@Override
	public Lops constructLops() 
		throws HopsException, LopsException 
	{
		if (get_lops() == null) {
			
			//construct input lops (recursive)
			ArrayList<Lops> tmp = new ArrayList<Lops>();
			for( Hops in : getInput() )
				tmp.add( in.constructLops() );
			
			//construct function call
			FunctionCallCP fcall = new FunctionCallCP( tmp, _fnamespace, _fname, _outputs );
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
}
