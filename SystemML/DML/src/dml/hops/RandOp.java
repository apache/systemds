package dml.hops;

import dml.api.DMLScript;
import dml.api.DMLScript.RUNTIME_PLATFORM;
import dml.lops.Lops;
import dml.lops.Rand;
import dml.lops.LopProperties.ExecType;
import dml.parser.DataIdentifier;
import dml.parser.Expression.ValueType;
import dml.runtime.controlprogram.parfor.ProgramConverter;
import dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import dml.sql.sqllops.SQLLopProperties;
import dml.sql.sqllops.SQLLops;
import dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import dml.sql.sqllops.SQLLops.GENERATES;
import dml.utils.HopsException;

/**
 * <p>Defines a Rand-HOP.</p>
 * 
 * @author schnetter
 */
public class RandOp extends Hops
{
	/** target identifier which will hold the random object */
	private DataIdentifier id;
	/** minimum of the random values */
	private double minValue;
	/** maximum of the random values */
	private double maxValue;
	/** sparsity of the random object */
	private double sparsity;
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
	public RandOp(DataIdentifier id, double minValue, double maxValue, double sparsity,
			String probabilityDensityFunction)
	{
		super(Kind.RandOp, id.getName(), id.getDataType(), ValueType.DOUBLE);
		this.id = id;
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.sparsity = sparsity;
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
			Rand rnd = new Rand(id, minValue, maxValue, sparsity,
					probabilityDensityFunction, ConfigurationManager
							.getConfig().getTextValue("scratch")
							+ ProgramConverter.CP_ROOT_THREAD_SEPARATOR
							+ ProgramConverter.CP_ROOT_THREAD_ID
							+ ProgramConverter.CP_ROOT_THREAD_SEPARATOR,
					get_dataType(), get_valueType(), et);
			rnd.getOutputParameters().setDimensions(get_dim1(), get_dim2(),
					get_rows_per_block(), get_cols_per_block());
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
	protected ExecType optFindExecType() throws HopsException {
		if (DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE)
			return ExecType.CP;

		if (this.areDimsBelowThreshold() || this.isVector())
			return ExecType.CP;
		else
			return ExecType.MR;
	}
}
