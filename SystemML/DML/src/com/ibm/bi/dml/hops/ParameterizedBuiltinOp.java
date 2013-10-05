/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.OptimizerUtils.OptimizationType;
import com.ibm.bi.dml.lops.CombineBinary;
import com.ibm.bi.dml.lops.CombineTertiary;
import com.ibm.bi.dml.lops.GroupedAggregate;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.ParameterizedBuiltin;
import com.ibm.bi.dml.lops.ReBlock;
import com.ibm.bi.dml.lops.CombineBinary.OperationTypes;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.sql.sqllops.SQLLops;


/**
 * Defines the HOP for calling an internal function (with custom parameters) from a DML script. 
 * 
 */
public class ParameterizedBuiltinOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	ParamBuiltinOp _op;

	/**
	 * List of "named" input parameters. They are maintained as a hashmap:
	 * parameter names (String) are mapped as indices (Integer) into getInput()
	 * arraylist.
	 * 
	 * i.e., getInput().get(_paramIndexMap.get(parameterName)) refers to the Hop
	 * that is associated with parameterName.
	 */
	private HashMap<String, Integer> _paramIndexMap = new HashMap<String, Integer>();

	private ParameterizedBuiltinOp() {
		//default constructor for clone
	}
	
	/**
	 * Creates a new HOP for a function call
	 */
	public ParameterizedBuiltinOp(String l, DataType dt, ValueType vt,
			ParamBuiltinOp op, HashMap<String, Hop> inputParameters) {
		super(Hop.Kind.ParameterizedBuiltinOp, l, dt, vt);

		_op = op;

		int index = 0;
		for (String s : inputParameters.keySet()) {
			Hop input = inputParameters.get(s);
			getInput().add(input);
			input.getParent().add(this);

			_paramIndexMap.put(s, index);
			index++;
		}
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	@Override
	public String getOpString() {
		return "" + _op;
	}

	@Override
	public Lop constructLops() throws HopsException, LopsException {
		if (get_lops() == null) {

			// construct lops for all input parameters
			HashMap<String, Lop> inputlops = new HashMap<String, Lop>();
			for (Entry<String, Integer> cur : _paramIndexMap.entrySet()) {
				inputlops.put(cur.getKey(), getInput().get(cur.getValue())
						.constructLops());
			}

			if (   _op == ParamBuiltinOp.CDF ) 
			{
				// simply pass the hashmap of parameters to the lop

				// set the lop for the function call
				
				ParameterizedBuiltin pbilop = new ParameterizedBuiltin(inputlops,
						HopsParameterizedBuiltinLops.get(_op), get_dataType(),
						get_valueType());
				
				pbilop.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
				
				set_lops(pbilop);

				// set the dimesnions for the lop for the output
				get_lops().getOutputParameters().setDimensions(get_dim1(),
						get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
			} 
			else if (_op == ParamBuiltinOp.GROUPEDAGG) {
				
				ExecType et = optFindExecType();
				if ( et == ExecType.MR ) 
				{
					// construct necessary lops: combineBinary/combineTertiary and
					// groupedAgg
	
					boolean isWeighted = (_paramIndexMap.get("weights") != null);
					if (isWeighted) {
						// combineTertiary followed by groupedAgg
						CombineTertiary combine = CombineTertiary
								.constructCombineLop(
										com.ibm.bi.dml.lops.CombineTertiary.OperationTypes.PreGroupedAggWeighted,
										inputlops.get("target"),
										inputlops.get("groups"),
										inputlops.get("weights"),
										DataType.MATRIX, get_valueType());
	
						// the dimensions of "combine" would be same as that of the
						// input data
						combine.getOutputParameters().setDimensions(
								getInput().get(_paramIndexMap.get("target"))
										.get_dim1(),
								getInput().get(_paramIndexMap.get("target"))
										.get_dim2(),		
								getInput().get(_paramIndexMap.get("target"))
										.get_rows_in_block(),
								getInput().get(_paramIndexMap.get("target"))
										.get_cols_in_block(), 
								getInput().get(_paramIndexMap.get("target"))
										.getNnz());
	
						// add the combine lop to parameter list, with a new name
						// "combinedinput"
						inputlops.put("combinedinput", combine);
						inputlops.remove("target");
						inputlops.remove("groups");
						inputlops.remove("weights");
	
					} else {
						// combineBinary followed by groupedAgg
						CombineBinary combine = CombineBinary.constructCombineLop(
								OperationTypes.PreGroupedAggUnweighted,
								inputlops.get("target"), inputlops
										.get("groups"), DataType.MATRIX,
								get_valueType());
	
						// the dimensions of "combine" would be same as that of the
						// input data
						combine.getOutputParameters().setDimensions(
								getInput().get(_paramIndexMap.get("target"))
										.get_dim1(),
								getInput().get(_paramIndexMap.get("target"))
										.get_dim2(),
								getInput().get(_paramIndexMap.get("target"))
										.get_rows_in_block(),
								getInput().get(_paramIndexMap.get("target"))
										.get_cols_in_block(), 
								getInput().get(_paramIndexMap.get("target"))
										.getNnz());
	
						// add the combine lop to parameter list, with a new name
						// "combinedinput"
						inputlops.put("combinedinput", combine);
						inputlops.remove("target");
						inputlops.remove("groups");
	
					}
					GroupedAggregate grp_agg = new GroupedAggregate(inputlops,
							get_dataType(), get_valueType());
					// output dimensions are unknown at compilation time
					grp_agg.getOutputParameters().setDimensions(-1, -1, -1, -1, -1);
					grp_agg.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
					//set_lops(grp_agg);
					
					ReBlock reblock = null;
					try {
						reblock = new ReBlock(
								grp_agg, get_rows_in_block(),
								get_cols_in_block(), get_dataType(),
								get_valueType());
					} catch (Exception e) {
						throw new HopsException(this.printErrorLocation() + "error creating Reblock Lop in ParameterizedBuiltinOp " , e);
					}
					reblock.getOutputParameters().setDimensions(-1, -1, 
							get_rows_in_block(), get_cols_in_block(), -1);
					
					reblock.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
					set_lops(reblock);
				}
				else //CP 
				{
					GroupedAggregate grp_agg = new GroupedAggregate(inputlops,
							get_dataType(), get_valueType(), et);
					// output dimensions are unknown at compilation time
					grp_agg.getOutputParameters().setDimensions(-1, -1, -1, -1, -1);
					grp_agg.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					// introduce a reblock lop only if it is NOT single_node execution
					if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE
						|| et == ExecType.CP ) 
					{
						if( et == ExecType.CP ){
							//force blocked output in CP (see below)
							grp_agg.getOutputParameters().setDimensions(-1, 1, 1000, 1000, -1);
						}
						
						//grouped agg, w/o reblock in CP
						set_lops(grp_agg);
					}
					else 
					{
						//insert reblock binarycell->binaryblock in MR
						ReBlock reblock = null;
						try {
							reblock = new ReBlock(
									grp_agg, get_rows_in_block(),
									get_cols_in_block(), get_dataType(),
									get_valueType());
						} catch (Exception e) {
							throw new HopsException(this.printErrorLocation() + "In ParameterizedBuiltinOp, error creating Reblock Lop " , e);
						}
						reblock.getOutputParameters().setDimensions(-1, -1, 
								get_rows_in_block(), get_cols_in_block(), -1);
						
						reblock.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		
						set_lops(reblock);
					}
				}
			}
			else if(   _op == ParamBuiltinOp.RMEMPTY ) 
			{
				ExecType et = optFindExecType();
				if( et == ExecType.MR ) //no MR version for rmempty
					et = ExecType.CP_FILE; //use file-based function for robustness
				
				ParameterizedBuiltin pbilop = new ParameterizedBuiltin(
						et, inputlops,
						HopsParameterizedBuiltinLops.get(_op), get_dataType(), get_valueType());
				
				pbilop.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
				
				set_lops(pbilop);

				// set the dimesnions for the lop for the output
				get_lops().getOutputParameters().setDimensions(get_dim1(),
						get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
			} 

		}

		return get_lops();
	}

	@Override
	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (get_visited() != VISIT_STATUS.DONE) {
				super.printMe();
				LOG.debug(" " + _op);
			}

			set_visited(VISIT_STATUS.DONE);
		}
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{	
		double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);	
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		return 0;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		//CDF always known because 
		
		long[] ret = null;
	
		Hop input = getInput().get(_paramIndexMap.get("target"));	
		MatrixCharacteristics mc = memo.getAllInputStats(input);

		if (   _op == ParamBuiltinOp.GROUPEDAGG ) { 
			// Output dimensions are completely data dependent. In the worst case, 
			// #groups = #rows in the grouping attribute (e.g., categorical attribute is an ID column, say EmployeeID).
			// In such a case, #rows in the output = #rows in the input. Also, output sparsity is 
			// likely to be 1.0 (e.g., groupedAgg(groups=<a ID column>, fn="count"))
			// get the size of longer dimension
			long m = (mc.get_rows() > 1 ? mc.get_rows() : mc.get_cols()); 
			if ( m > 1 )
			{
				ret = new long[]{m, 1, m};
			}
		}
		else if (   _op == ParamBuiltinOp.RMEMPTY ) 
		{ 
			// similar to groupedagg because in the worst-case ouputsize eq inputsize
			// #nnz is exactly the same as in the input but sparsity can be higher if dimensions.
			// change (denser output).
			if ( mc.dimsKnown() )
				ret= new long[]{mc.get_rows(), mc.get_cols(), mc.getNonZeros()}; 
		}
		
		return ret;
	}
	
	@Override
	public boolean allowsAllExecTypes()
	{
		return false;
	}
	
	@Override
	protected ExecType optFindExecType() 
		throws HopsException 
	{
		checkAndSetForcedPlatform();

		if( _etypeForced != null ) 			
			_etype = _etypeForced;	
		else 
		{
			if ( OptimizerUtils.getOptType() == OptimizationType.MEMORY_BASED ) {
				_etype = findExecTypeByMemEstimate();
			}
			else if ( _op == ParamBuiltinOp.GROUPEDAGG ) {
				if ( this.getInput().get(0).areDimsBelowThreshold() )
					_etype = ExecType.CP;
				else
					_etype = ExecType.MR;
			}
			
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown() && _etype==ExecType.MR )
				setRequiresRecompile();
		}
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		switch( _op )
		{
			case CDF:
				//do nothing; CDF is a scalar
				break;
			
			case GROUPEDAGG:  
				//output dimension dim1 is completely data dependent 
				set_dim2( 1 );
				break;
			
			case RMEMPTY: 
				//one output dimension dim1 or dim2 is completely data dependent 
				Hop target = getInput().get(_paramIndexMap.get("target"));
				String margin = getInput().get(_paramIndexMap.get("margin")).toString();
				if( margin.equals("rows") )
					set_dim2( target.get_dim2() );
				else if (margin.equals("cols"))
					set_dim1( target.get_dim1() );
				setNnz( target.getNnz() );
				break;
		}
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		ParameterizedBuiltinOp ret = new ParameterizedBuiltinOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._op = _op;
		ret._paramIndexMap = (HashMap<String, Integer>) _paramIndexMap.clone();
		//note: no deep cp of params since read-only 
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( that._kind!=Kind.ParameterizedBuiltinOp )
			return false;
		
		ParameterizedBuiltinOp that2 = (ParameterizedBuiltinOp)that;	
		boolean ret = (_op == that2._op
					  && _paramIndexMap!=null && that2._paramIndexMap!=null );
		if( ret )
		{
			for( Entry<String,Integer> e : _paramIndexMap.entrySet() )
			{
				String key1 = e.getKey();
				int pos1 = e.getValue();
				int pos2 = that2._paramIndexMap.get(key1);
				ret &= (   that2.getInput().get(pos2)!=null
					    && getInput().get(pos1) == that2.getInput().get(pos2) );
			}
		}
		
		return ret;
	}
}
