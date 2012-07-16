package com.ibm.bi.dml.hops;


import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.Rand;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;
import com.ibm.bi.dml.utils.HopsException;
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
	
	/** target identifier which will hold the random object */
	private DataIdentifier id;
	/** minimum of the random values */
	private double minValue;
	/** maximum of the random values */
	private double maxValue;
	/** sparsity of the random object */
	private double sparsity;
	/** fixed seed for all invocations, -1 for random seed on each invocation */
	private long seed;
	/** probability density function which is used to produce the sparsity */
	private String probabilityDensityFunction;

	
	/**
	 * <p>Creates a new Rand-HOP. The target identifier has to hold the dimensions of the new random object.</p>
	 * 
	 * @param id the target identifier
	 * @param minValue minimum of the random values
	 * @param maxValue maximum of the random values
	 * @param sparsity sparsity of the random object
	 * @param probabilityDensityFunction probability density function
	 */
	public RandOp(DataIdentifier id, double minValue, double maxValue, double sparsity, long seed,
			String probabilityDensityFunction)
	{
		super(Kind.RandOp, id.getName(), id.getDataType(), ValueType.DOUBLE);
		this.id = id;
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.sparsity = sparsity;
		this.seed = seed;
		this.probabilityDensityFunction = probabilityDensityFunction;
	}

	@Override
	public String getOpString() {
		return "rand";
	}
	
	@Override
	public Lops constructLops() throws HopsException
	{
		if(get_lops() == null)
		{
			ExecType et = optFindExecType();
			
			String scratchSpaceLoc = null;
			try {
				scratchSpaceLoc = ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE);
			} catch (Exception e){
				System.out.println("ERROR: could not retrieve parameter " + DMLConfig.SCRATCH_SPACE + " from DMLConfig");
			}
			
			Rand rnd = new Rand(id, minValue, maxValue, sparsity, seed,
					probabilityDensityFunction, scratchSpaceLoc
							+ ProgramConverter.CP_ROOT_THREAD_SEPARATOR
							+ ProgramConverter.CP_ROOT_THREAD_ID
							+ ProgramConverter.CP_ROOT_THREAD_SEPARATOR,
					get_dataType(), get_valueType(), et);
			rnd.getOutputParameters().setDimensions(get_dim1(), get_dim2(),
					get_rows_in_block(), get_cols_in_block(), getNnz());
			set_lops(rnd);
		}
		
		return get_lops();
	}
	
	@Override
	public void printMe() throws HopsException
	{
		if(get_visited() != VISIT_STATUS.DONE)
		{
			super.printMe();
			System.out.println();
		}
		
		set_visited(VISIT_STATUS.DONE);
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
					+ this.get_dim2() + ", " + this.minValue + ", " + this.maxValue + ", " + this.sparsity + ");");
			
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
	protected ExecType optFindExecType() throws HopsException {
		if (DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE)
			return ExecType.CP;
		else if ( DMLScript.rtplatform == RUNTIME_PLATFORM.HADOOP )
			return ExecType.MR;

		if( _etype != null ) 			
			return _etype;
		
		if (this.areDimsBelowThreshold() || this.isVector())
			return ExecType.CP;
		else
			return ExecType.MR;
	}
	
	public static long generateRandomSeed()
	{
		return System.nanoTime();
	}
}
