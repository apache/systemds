package com.ibm.bi.dml.hops;


import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.hops.OptimizerUtils.OptimizationType;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.Rand;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.RandStatement;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LopsException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;

/**
 * <p>Defines a Rand-HOP.</p>
 * 
 * 
 */
public class RandOp extends Hops
{
	//TODO: MB: potentially move constant and rand seed generation to place in runtime (but currently no central place)
	public static final long UNSPECIFIED_SEED = -1;
	
	/**
	 * List of "named" input parameters. They are maintained as a hashmap:
	 * parameter names (String) are mapped as indices (Integer) into getInput()
	 * arraylist.
	 * 
	 * i.e., getInput().get(_paramIndexMap.get(parameterName)) refers to the Hop
	 * that is associated with parameterName.
	 */
	private HashMap<String, Integer> _paramIndexMap = new HashMap<String, Integer>();
	
	/** target identifier which will hold the random object */
	private DataIdentifier id;
	/** sparsity of the random object */
	/** this is used for mem estimate */
	private double sparsity;
	
	private RandOp() {
		//default constructor for clone
	}
	
	/**
	 * <p>Creates a new Rand HOP.</p>
	 * 
	 * @param id the target identifier
	 * @param inputParameters HashMap of the input parameters for Rand Hop
	 */
	public RandOp(DataIdentifier id, HashMap<String, Hops> inputParameters){
		super(Kind.RandOp, id.getName(), DataType.MATRIX, ValueType.DOUBLE);
		
		this.id = id;
				
		int index = 0;
		for (String s : inputParameters.keySet()) {
			Hops input = inputParameters.get(s);
			getInput().add(input);
			input.getParent().add(this);

			_paramIndexMap.put(s, index);
			index++;
		}
		sparsity = Double.valueOf(((LiteralOp)inputParameters.get(RandStatement.RAND_SPARSITY)).get_name());
	}
	
	@Override
	public String getOpString() {
		return "rand";
	}
	
	@Override
	public Lops constructLops() throws HopsException, LopsException
	{
		if(get_lops() == null)
		{
			ExecType et = optFindExecType();
			
			String scratchSpaceLoc = null;
			try {
				scratchSpaceLoc = ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE);
			} catch (Exception e){
				throw new LopsException("Could not retrieve parameter " + DMLConfig.SCRATCH_SPACE + " from DMLConfig", e);
			}
			
			HashMap<String, Lops> inputLops = new HashMap<String, Lops>();
			for (Entry<String, Integer> cur : _paramIndexMap.entrySet()) {
				inputLops.put(cur.getKey(), getInput().get(cur.getValue())
						.constructLops());
			}
			
			Rand rnd = new Rand(id, inputLops,
					scratchSpaceLoc + Lops.FILE_SEPARATOR + Lops.PROCESS_PREFIX + DMLScript.getUUID() + Lops.FILE_SEPARATOR + 
		   					          Lops.FILE_SEPARATOR + ProgramConverter.CP_ROOT_THREAD_ID + Lops.FILE_SEPARATOR,
					get_dataType(), get_valueType(), et);
			
			rnd.getOutputParameters().setDimensions(
					get_dim1(), get_dim2(),
					get_rows_in_block(), get_cols_in_block(), getNnz());
			
			rnd.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			set_lops(rnd);
		}
		
		return get_lops();
	}
	
	@Override
	public void printMe() throws HopsException
	{	
		if (LOG.isDebugEnabled()){
			if(get_visited() != VISIT_STATUS.DONE)
			{
				super.printMe();
			}

			set_visited(VISIT_STATUS.DONE);
		}
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		if(this.get_sqllops() == null)
		{
			
			SQLLops sqllop = new SQLLops("Random" + this.id.getName(), GENERATES.PROC,
					this.get_valueType(),
					this.get_dataType());

			//TODO extend for seed
			sqllop.set_sql("CALL gensparsematrix('" + sqllop.get_tableName() + "', " + this.get_dim1() + ", "
					+ this.get_dim2() + ");");
			
			sqllop.set_properties(getProperties());
			this.set_sqllops(sqllop);
		}
		return this.get_sqllops();
	}
	
	private SQLLopProperties getProperties()
	{
		SQLLopProperties prop = new SQLLopProperties();
		prop.setJoinType(JOINTYPE.NONE);
		prop.setAggType(AGGREGATIONTYPE.NONE);
		prop.setOpString("Rand");
		return prop;
	}

	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}	
	
	@Override
	public double computeMemEstimate() 
	{	
		if( dimsKnown() )
		{
			Hops min = getInput().get(_paramIndexMap.get("min")); //min 
			Hops max = getInput().get(_paramIndexMap.get("max")); //max
			if(    min instanceof LiteralOp && min.get_name().equals("0")
				&& max instanceof LiteralOp && max.get_name().equals("0"))
			{
				_outputMemEstimate = OptimizerUtils.estimateSizeEmptyBlock(get_dim1(), get_dim2());
			}
			else
				_outputMemEstimate = OptimizerUtils.estimateSizeExactSparsity(get_dim1(), get_dim2(), sparsity);
		}
		else
			_outputMemEstimate = OptimizerUtils.DEFAULT_SIZE;
			
		_memEstimate = getInputOutputSize();
		
		return _memEstimate;
	}
	
	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();

		if( _etypeForced != null ) 			
			_etype = _etypeForced;
		else 
		{
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown() )
				setRequiresRecompile();
			
			if ( OptimizerUtils.getOptType() == OptimizationType.MEMORY_BASED ) {
				_etype = findExecTypeByMemEstimate();
			}
			else if (this.areDimsBelowThreshold() || this.isVector())
				_etype = ExecType.CP;
			else
				_etype = ExecType.MR;
			}
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		Hops input1 = getInput().get(_paramIndexMap.get("rows")); //rows 
		Hops input2 = getInput().get(_paramIndexMap.get("cols")); //cols

		//refresh rows information
		if( input1 instanceof UnaryOp )
		{
			if( ((UnaryOp)input1).get_op() == Hops.OpOp1.NROW )
				set_dim1(input1.getInput().get(0).get_dim1());
			else if ( ((UnaryOp)input1).get_op() == Hops.OpOp1.NCOL )
				set_dim1(input1.getInput().get(0).get_dim2());
		}
		
		//refresh cols information
		if( input2 instanceof UnaryOp )
		{
			if( ((UnaryOp)input2).get_op() == Hops.OpOp1.NROW  )
				set_dim2(input2.getInput().get(0).get_dim1());
			else if( ((UnaryOp)input2).get_op() == Hops.OpOp1.NCOL  )
				set_dim2(input2.getInput().get(0).get_dim2());
		}
	}
	

	public HashMap<String, Integer> getParamIndexMap()
	{
		return _paramIndexMap;
	}
	
	public static long generateRandomSeed()
	{
		return System.nanoTime();
	}

	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		RandOp ret = new RandOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret.id = id;
		ret.sparsity = sparsity;
		ret._paramIndexMap = (HashMap<String, Integer>) _paramIndexMap.clone();
		//note: no deep cp of params since read-only 
		
		return ret;
	}
}
