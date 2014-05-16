/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;


import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.parser.Statement;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;

/**
 * 
 * 
 */
public class DataGenOp extends Hop
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//TODO: MB: potentially move constant and rand seed generation to place in runtime (but currently no central place)
	public static final long UNSPECIFIED_SEED = -1;
	
	
	private DataGenMethod method; // defines the specific data generation method -- random matrix or sequence
	
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
	/** sparsity of the random object, this is used for mem estimate */
	private double sparsity;
	/** base directory for temp file (e.g., input seeds)*/
	private String _baseDir;
	
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
			sparsity = Double.valueOf(((LiteralOp)inputParameters.get(DataExpression.RAND_SPARSITY)).get_name());

		//generate base dir
		String scratch = ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE);
		_baseDir = scratch + Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID() + Lop.FILE_SEPARATOR + 
	               Lop.FILE_SEPARATOR + ProgramConverter.CP_ROOT_THREAD_ID + Lop.FILE_SEPARATOR;
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}
	
	@Override
	public String getOpString() {
		return "dg(" + method +")";
	}
	
	public DataGenMethod getDataGenMethod() {
		return method;
	}
	
	@Override
	public Lop constructLops() 
		throws HopsException, LopsException
	{
		if(get_lops() == null)
		{
			ExecType et = optFindExecType();
			
			HashMap<String, Lop> inputLops = new HashMap<String, Lop>();
			for (Entry<String, Integer> cur : _paramIndexMap.entrySet()) {
				if( cur.getKey().equals(DataExpression.RAND_ROWS) && _dim1>0 )
					inputLops.put(cur.getKey(), new LiteralOp(String.valueOf(_dim1), _dim1).constructLops());
				else if( cur.getKey().equals(DataExpression.RAND_COLS) && _dim2>0 )
					inputLops.put(cur.getKey(), new LiteralOp(String.valueOf(_dim2), _dim2).constructLops());
				else
					inputLops.put(cur.getKey(), getInput().get(cur.getValue()).constructLops());
			}
			
			DataGen rnd = new DataGen(method, id, inputLops,_baseDir,
					get_dataType(), get_valueType(), et);
			
			rnd.getOutputParameters().setDimensions(
					get_dim1(), get_dim2(),
					//robust handling for blocksize (important for -exec singlenode; otherwise incorrect results)
					(get_rows_in_block()>0)?get_rows_in_block():DMLTranslator.DMLBlockSize, 
					(get_cols_in_block()>0)?get_cols_in_block():DMLTranslator.DMLBlockSize,  
					getNnz());
			
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
		
		try
		{
			if ( method == DataGenMethod.RAND ) {
				Hop min = getInput().get(_paramIndexMap.get(DataExpression.RAND_MIN)); //min 
				Hop max = getInput().get(_paramIndexMap.get(DataExpression.RAND_MAX)); //max
				if(    min instanceof LiteralOp && ((LiteralOp)min).getDoubleValue()==0.0
					&& max instanceof LiteralOp && ((LiteralOp)max).getDoubleValue()==0.0 )
				{
					ret = OptimizerUtils.estimateSizeEmptyBlock(dim1, dim2);
				}
				else
				{
					ret = OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, 1.0);
					//disabled sparsity-aware estimation due to dense generation, we can re-enable it once, we have a sparse generation method
					//ret = OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
				}
			}
			else {
				_outputMemEstimate = OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, 1.0);	
			}
		}
		catch(HopsException he)
		{
			throw new RuntimeException(he);
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
		if( method == DataGenMethod.RAND &&
			OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION )
		{
			long dim1 = computeDimParameterInformation(getInput().get(_paramIndexMap.get(DataExpression.RAND_ROWS)), memo);
			long dim2 = computeDimParameterInformation(getInput().get(_paramIndexMap.get(DataExpression.RAND_COLS)), memo);
			long nnz = (long)(sparsity * dim1 * dim2);
			if( dim1>0 && dim2>0 )
				return new long[]{ dim1, dim2, nnz };
		}
		
		return null;
	}

	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();

		if( _etypeForced != null ) 			
			_etype = _etypeForced;
		else 
		{
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) {
				_etype = findExecTypeByMemEstimate();
			}
			else if (this.areDimsBelowThreshold() || this.isVector())
				_etype = ExecType.CP;
			else
				_etype = ExecType.MR;
			
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==ExecType.MR )
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

		if ( method == DataGenMethod.RAND ) 
		{
			input1 = getInput().get(_paramIndexMap.get(DataExpression.RAND_ROWS)); //rows
			input2 = getInput().get(_paramIndexMap.get(DataExpression.RAND_COLS)); //cols
			
			//refresh rows information
			refreshRowsParameterInformation(input1);
			
			//refresh cols information
			refreshColsParameterInformation(input2);
		}
		else if (method == DataGenMethod.SEQ ) 
		{
			input1 = getInput().get(_paramIndexMap.get(Statement.SEQ_FROM));
			input2 = getInput().get(_paramIndexMap.get(Statement.SEQ_TO)); 
			input3 = getInput().get(_paramIndexMap.get(Statement.SEQ_INCR)); 

			double from = computeBoundsInformation(input1);
			boolean fromKnown = (from != Double.MAX_VALUE);
			
			double to = computeBoundsInformation(input2);
			boolean toKnown = (to != Double.MAX_VALUE);
			
			double incr = computeBoundsInformation(input3);
			boolean incrKnown = (incr != Double.MAX_VALUE);
			
			if(  !incrKnown && input3.getKind() == Kind.BinaryOp //special case for incr
			   && ((BinaryOp)input3).getOp() == Hop.OpOp2.SEQINCR && fromKnown && toKnown) 
			{
				incr = ( from >= to ) ? -1 : 1;
				incrKnown = true;
			}
			
			if ( fromKnown && toKnown && incrKnown ) {
				set_dim1(1 + (long)Math.floor(((double)(to-from))/incr));
				set_dim2(1);
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
	
	public int getParamIndex(String key)
	{
		return _paramIndexMap.get(key);
	}
	
	public boolean hasConstantValue()
	{
		Hop min = getInput().get(_paramIndexMap.get(DataExpression.RAND_MIN)); //min 
		Hop max = getInput().get(_paramIndexMap.get(DataExpression.RAND_MAX)); //max
		return min.equals(max);
	}
	
	public static long generateRandomSeed()
	{
		return System.nanoTime();
	}
	

	@SuppressWarnings("unchecked")
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
		ret._baseDir = _baseDir;
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
				      && _baseDir.equals(that2._baseDir)
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
			
			//special case for rand seed (no CSE if unspecified seed because runtime generated)
			//note: if min and max is constant, we can safely merge those hops
			if( method == DataGenMethod.RAND ){
				Hop seed = getInput().get(_paramIndexMap.get(DataExpression.RAND_SEED));
				Hop min = getInput().get(_paramIndexMap.get(DataExpression.RAND_MIN));
				Hop max = getInput().get(_paramIndexMap.get(DataExpression.RAND_MAX));
				if( seed.get_name().equals(String.valueOf(DataGenOp.UNSPECIFIED_SEED)) && min != max )
					ret = false;
			}
		}
		
		return ret;
	}
}
