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
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.OptimizerUtils.OptimizationType;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.RandStatement;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;

/**
 * <p>Defines a Rand-HOP.</p>
 * 
 * 
 */
public class DataGenOp extends Hop
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//TODO: MB: potentially move constant and rand seed generation to place in runtime (but currently no central place)
	public static final long UNSPECIFIED_SEED = -1;
	
	// defines the specific data generation method -- random matrix or sequence
	DataGenMethod method;
	
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
	
	private DataGenOp() {
		//default constructor for clone
	}
	
	/**
	 * <p>Creates a new Rand HOP.</p>
	 * 
	 * @param id the target identifier
	 * @param inputParameters HashMap of the input parameters for Rand Hop
	 */
	public DataGenOp(DataGenMethod mthd, DataIdentifier id, HashMap<String, Hop> inputParameters){
		super(Kind.DataGenOp, id.getName(), DataType.MATRIX, ValueType.DOUBLE);
		this.id = id;
		this.method = mthd;
				
		int index = 0;
		for (String s : inputParameters.keySet()) {
			Hop input = inputParameters.get(s);
			getInput().add(input);
			input.getParent().add(this);

			_paramIndexMap.put(s, index);
			index++;
		}
		if ( mthd == DataGenMethod.RAND )
			sparsity = Double.valueOf(((LiteralOp)inputParameters.get(RandStatement.RAND_SPARSITY)).get_name());

		//compute unknown dims and nnz
		refreshSizeInformation();
	}
	
	@Override
	public String getOpString() {
		return "datagen_" + method;
	}
	
	public DataGenMethod getDataGenMethod() {
		return method;
	}
	
	@Override
	public Lop constructLops() throws HopsException, LopsException
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
			
			HashMap<String, Lop> inputLops = new HashMap<String, Lop>();
			for (Entry<String, Integer> cur : _paramIndexMap.entrySet()) {
				inputLops.put(cur.getKey(), getInput().get(cur.getValue())
						.constructLops());
			}
			
			DataGen rnd = new DataGen(method, id, inputLops,
					scratchSpaceLoc + Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID() + Lop.FILE_SEPARATOR + 
		   					          Lop.FILE_SEPARATOR + ProgramConverter.CP_ROOT_THREAD_ID + Lop.FILE_SEPARATOR,
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
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		double ret = 0;
		
		if ( method == DataGenMethod.RAND ) {
			Hop min = getInput().get(_paramIndexMap.get("min")); //min 
			Hop max = getInput().get(_paramIndexMap.get("max")); //max
			if(    min instanceof LiteralOp && min.get_name().equals("0")
				&& max instanceof LiteralOp && max.get_name().equals("0"))
			{
				ret = OptimizerUtils.estimateSizeEmptyBlock(dim1, dim2);
			}
			else
				ret = OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
		}
		else {
			_outputMemEstimate = OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, 1.0);	
		}
		return ret;
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		return 0;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		return null;
	}

	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();

		if( _etypeForced != null ) 			
			_etype = _etypeForced;
		else 
		{
			if ( OptimizerUtils.getOptType() == OptimizationType.MEMORY_BASED ) {
				_etype = findExecTypeByMemEstimate();
			}
			else if (this.areDimsBelowThreshold() || this.isVector())
				_etype = ExecType.CP;
			else
				_etype = ExecType.MR;
			
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown() && _etype==ExecType.MR )
				setRequiresRecompile();
		}
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		
		Hop input1 = null;  
		Hop input2 = null; 
		Hop input3 = null;

		if ( method == DataGenMethod.RAND ) {
			input1 = getInput().get(_paramIndexMap.get("rows")); //rows
			input2 = getInput().get(_paramIndexMap.get("cols")); //cols
			
			//refresh rows information
			refreshRowsParameterInformation(input1);
			
			//refresh cols information
			refreshColsParameterInformation(input2);
		}
		else if (method == DataGenMethod.SEQ ) {
			input1 = getInput().get(_paramIndexMap.get(RandStatement.SEQ_FROM));
			input2 = getInput().get(_paramIndexMap.get(RandStatement.SEQ_TO)); 
			input3 = getInput().get(_paramIndexMap.get(RandStatement.SEQ_INCR)); 

			double from, to, incr;
			from = to = incr = Double.NaN;
			boolean fromKnown, toKnown, incrKnown;
			fromKnown = toKnown = incrKnown = false;
			
			try {
				if ( input1.getKind() == Kind.LiteralOp ) {
					from = ((LiteralOp)input1).getDoubleValue();
					fromKnown = true;
				}
				else if ( input1.getKind() == Kind.UnaryOp ) {
					if( ((UnaryOp)input1).get_op() == Hop.OpOp1.NROW ) {
						from = input1.getInput().get(0).get_dim1();
						fromKnown = true;
					}
					else if ( ((UnaryOp)input1).get_op() == Hop.OpOp1.NCOL ) {
						from = input1.getInput().get(0).get_dim2();
						fromKnown = true;
					}
				}
				
				if ( input2.getKind() == Kind.LiteralOp ) {
					to = ((LiteralOp)input2).getDoubleValue();
					toKnown = true;
				}
				else if ( input2.getKind() == Kind.UnaryOp ) {
					if( ((UnaryOp)input2).get_op() == Hop.OpOp1.NROW ) {
						to = input2.getInput().get(0).get_dim1();
						toKnown = true;
					}
					else if ( ((UnaryOp)input2).get_op() == Hop.OpOp1.NCOL ) {
						to = input2.getInput().get(0).get_dim2();
						toKnown = true;
					}
				}
				
				if ( input3.getKind() == Kind.LiteralOp ) {
					incr = ((LiteralOp)input3).getDoubleValue();
					incrKnown = true;
				}
				else if ( input3.getKind() == Kind.UnaryOp ) {
					if( ((UnaryOp)input3).get_op() == Hop.OpOp1.NROW ) {
						incr = input3.getInput().get(0).get_dim1();
						incrKnown = true;
					}
					else if ( ((UnaryOp)input3).get_op() == Hop.OpOp1.NCOL ) {
						incr = input3.getInput().get(0).get_dim2();
						incrKnown = true;
					}
				}
				else if (input3.getKind() == Kind.BinaryOp && ((BinaryOp)input3).getOp() == Hop.OpOp2.SEQINCR && fromKnown && toKnown) {
					if ( from >= to )
						incr = -1.0;
					else 
						incr = 1.0;
					incrKnown = true;
				}
				
				if ( fromKnown && toKnown && incrKnown ) {
					set_dim1(1 + (long)Math.floor((to-from)/incr));
					set_dim2(1);
				}
			} catch (HopsException e) {
				// TODO Auto-generated catch block
				throw new RuntimeException(e);
			}
		}
		
		//refresh nnz information
		if( dimsKnown() )
			_nnz = (long) (_dim1 * _dim2 * sparsity);
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
		DataGenOp ret = new DataGenOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret.method = method;
		ret.id = id;
		ret.sparsity = sparsity;
		ret._paramIndexMap = (HashMap<String, Integer>) _paramIndexMap.clone();
		//note: no deep cp of params since read-only 
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( that._kind!=Kind.DataGenOp )
			return false;
		
		DataGenOp that2 = (DataGenOp)that;	
		boolean ret = (  method == that2.method
				      && sparsity == that2.sparsity
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
