package com.ibm.bi.dml.hops;

import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.lops.CombineBinary;
import com.ibm.bi.dml.lops.CombineTertiary;
import com.ibm.bi.dml.lops.GroupedAggregate;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.ParameterizedBuiltin;
import com.ibm.bi.dml.lops.ReBlock;
import com.ibm.bi.dml.lops.CombineBinary.OperationTypes;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LopsException;


/**
 * Defines the HOP for calling a function from a DML script. Handles both
 * internal and external functions.
 * 
 */
public class ParameterizedBuiltinOp extends Hops {

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

	/**
	 * Creates a new HOP for a function call
	 */
	public ParameterizedBuiltinOp(String l, DataType dt, ValueType vt,
			ParamBuiltinOp op, HashMap<String, Hops> inputParameters) {
		super(Hops.Kind.ParameterizedBuiltinOp, l, dt, vt);

		_op = op;

		int index = 0;
		for (String s : inputParameters.keySet()) {
			Hops input = inputParameters.get(s);
			getInput().add(input);
			input.getParent().add(this);

			_paramIndexMap.put(s, index);
			index++;
		}
	}

	@Override
	public String getOpString() {
		return "" + _op;
	}

	@Override
	public Lops constructLops() throws HopsException, LopsException {
		if (get_lops() == null) {

			// construct lops for all input parameters
			HashMap<String, Lops> inputlops = new HashMap<String, Lops>();
			for (Entry<String, Integer> cur : _paramIndexMap.entrySet()) {
				inputlops.put(cur.getKey(), getInput().get(cur.getValue())
						.constructLops());
			}

			if (_op == ParamBuiltinOp.CDF) {
				// simply pass the hashmap of parameters to the lop

				// set the lop for the function call
				set_lops(new ParameterizedBuiltin(inputlops,
						HopsParameterizedBuiltinLops.get(_op), get_dataType(),
						get_valueType()));

				// set the dimesnions for the lop for the output
				get_lops().getOutputParameters().setDimensions(get_dim1(),
						get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
			} else if (_op == ParamBuiltinOp.GROUPEDAGG) {
				// construct necessary lops: combineBinary/combineTertiary and
				// groupedAgg

				boolean isWeighted = (_paramIndexMap.get("weights") != null);
				if (isWeighted) {
					// combineTertiary followed by groupedAgg
					CombineTertiary combine = CombineTertiary
							.constructCombineLop(
									com.ibm.bi.dml.lops.CombineTertiary.OperationTypes.PreGroupedAggWeighted,
									(Lops) inputlops.get("target"),
									(Lops) inputlops.get("groups"),
									(Lops) inputlops.get("weights"),
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
							(Lops) inputlops.get("target"), (Lops) inputlops
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

				//set_lops(grp_agg);
				
				ReBlock reblock = null;
				try {
					reblock = new ReBlock(
							grp_agg, get_rows_in_block(),
							get_cols_in_block(), get_dataType(),
							get_valueType());
				} catch (Exception e) {
					throw new HopsException(e);
				}
				reblock.getOutputParameters().setDimensions(-1, -1, 
						get_rows_in_block(), get_cols_in_block(), -1);

				set_lops(reblock);
				
			}

		}

		return get_lops();
	}

	@Override
	public void printMe() throws HopsException {
		if (get_visited() != VISIT_STATUS.DONE) {
			super.printMe();
			System.out.println(" " + _op);
		}

		set_visited(VISIT_STATUS.DONE);
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public boolean allowsAllExecTypes()
	{
		return false;
	}

	@Override
	protected ExecType optFindExecType() throws HopsException {
		// TODO Auto-generated method stub
		return null;
	}
}
