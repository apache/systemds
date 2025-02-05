/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.hops;


import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.DataGen;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * A DataGenOp can be rand (or matrix constructor), sequence, and sample -
 * these operators have different parameters and use a map of parameter type to hop position.
 */
public class DataGenOp extends MultiThreadedHop
{
	public static final long UNSPECIFIED_SEED = -1;
	
	 // defines the specific data generation method
	private OpOpDG _op;
	
	/**
	 * List of "named" input parameters. They are maintained as a hashmap:
	 * parameter names (String) are mapped as indices (Integer) into getInput()
	 * arraylist.
	 * 
	 * i.e., getInput().get(_paramIndexMap.get(parameterName)) refers to the Hop
	 * that is associated with parameterName.
	 */
	private HashMap<String, Integer> _paramIndexMap = new HashMap<>();
		

	/** target identifier which will hold the random object */
	private DataIdentifier _id;
	
	//Rand-specific attributes
	
	/** sparsity of the random object, this is used for mem estimate */
	private double _sparsity = -1;
	/** base directory for temp file (e.g., input seeds)*/
	private String _baseDir;
	
	//seq-specific attributes (used for recompile/recompile)
	private double _incr = Double.MAX_VALUE; 
	
	
	private DataGenOp() {
		//default constructor for clone
	}
	
	/**
	 * <p>Creates a new Rand HOP.</p>
	 * 
	 * @param mthd data gen method
	 * @param id the target identifier
	 * @param inputParameters HashMap of the input parameters for Rand Hop
	 */
	public DataGenOp(OpOpDG mthd, DataIdentifier id, HashMap<String, Hop> inputParameters) {
		super(id.getName(),
			id.getDataType().isUnknown() ? DataType.MATRIX : id.getDataType(),
			id.getValueType().isUnknown() ? ValueType.FP64 : id.getValueType());
		
		_id = id;
		_op = mthd;
		
		//ensure all parameters existing and consistent with data type
		if( inputParameters.containsKey(DataExpression.RAND_DIMS) )
			setDataType(DataType.TENSOR);
		
		int index = 0;
		for( Entry<String, Hop> e: inputParameters.entrySet() ) {
			String s = e.getKey();
			Hop input = e.getValue();
			getInput().add(input);
			input.getParent().add(this);

			_paramIndexMap.put(s, index);
			index++;
		}
		
		Hop sparsityOp = inputParameters.get(DataExpression.RAND_SPARSITY);
		if ( mthd == OpOpDG.RAND && sparsityOp instanceof LiteralOp)
			_sparsity = HopRewriteUtils.getDoubleValue((LiteralOp)sparsityOp);
		
		//generate base dir
		String scratch = ConfigurationManager.getScratchSpace();
		_baseDir = scratch + Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID() + Lop.FILE_SEPARATOR 
			+ Lop.FILE_SEPARATOR + Lop.CP_ROOT_THREAD_ID + Lop.FILE_SEPARATOR;

		//TODO size information for tensor
		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	public DataGenOp(OpOpDG mthd, DataIdentifier id)
	{
		super(id.getName(), DataType.SCALAR, ValueType.INT64);
		
		_id = id;
		_op = mthd;

		//generate base dir
		String scratch = ConfigurationManager.getScratchSpace();
		_baseDir = scratch + Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID() + Lop.FILE_SEPARATOR 
			+ Lop.FILE_SEPARATOR + Lop.CP_ROOT_THREAD_ID + Lop.FILE_SEPARATOR;
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	@Override
	public String getOpString() {
		return "dg(" + _op.toString().toLowerCase() +")";
	}
	
	public OpOpDG getOp() {
		return _op;
	}
	
	@Override
	public boolean isGPUEnabled() {
		return false;
	}
	
	@Override
	public boolean isMultiThreadedOpType() {
		return _op == OpOpDG.RAND;
	}
	
	@Override
	public Lop constructLops() 
	{
		//return already created lops
		if( getLops() != null )
			return getLops();

		ExecType et = optFindExecType();
		
		HashMap<String, Lop> inputLops = new HashMap<>();
		for (Entry<String, Integer> cur : _paramIndexMap.entrySet()) {
			if( cur.getKey().equals(DataExpression.RAND_ROWS) && rowsKnown() )
				inputLops.put(cur.getKey(), new LiteralOp(getDim1()).constructLops());
			else if( cur.getKey().equals(DataExpression.RAND_COLS) && colsKnown() )
				inputLops.put(cur.getKey(), new LiteralOp(getDim2()).constructLops());
			else
				inputLops.put(cur.getKey(), getInput().get(cur.getValue()).constructLops());
		}
		
		DataGen rnd = new DataGen(_op, _id, inputLops,
			_baseDir, getDataType(), getValueType(), et);
		
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		rnd.setNumThreads(k);
		
		rnd.getOutputParameters().setDimensions(
				getDim1(), getDim2(),
				//robust handling for blocksize (important for -exec singlenode; otherwise incorrect results)
				(getBlocksize()>0)?getBlocksize():ConfigurationManager.getBlocksize(),
				//actual rand nnz might differ (in cp/mr they are corrected after execution)
				(_op==OpOpDG.RAND && et==ExecType.SPARK && getNnz()!=0) ? -1 : getNnz(),
				getUpdateType());
		
		setLineNumbers(rnd);
		setLops(rnd);
		
		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();

		return getLops();
	}

	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}	
	
	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		double ret;
		
		if ( _op == OpOpDG.RAND && _sparsity != -1 ) {
			if( hasConstantValue(0.0) ) { //if empty block
				ret = OptimizerUtils.estimateSizeEmptyBlock(dim1, dim2);
			}
			else {
				//sparsity-aware estimation (dependent on sparse generation approach); for pure dense generation
				//we would need to disable sparsity-awareness and estimate via sparsity=1.0
				ret = OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, _sparsity, getDataType());
			}
		}
		else {
			ret = OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, 1.0, getDataType());
		}
		
		return ret;
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		if ( _op == OpOpDG.RAND && dimsKnown() ) {
			long numBlocks = (long) (Math.ceil((double)dim1/ConfigurationManager.getBlocksize()) 
					* Math.ceil((double)dim2/ConfigurationManager.getBlocksize()));
			return 32 + numBlocks*8.0; // 32 bytes of overhead for an array of long & numBlocks long values.
		}
		else 
			return 0;
	}
	
	@Override
	protected DataCharacteristics inferOutputCharacteristics( MemoTable memo )
	{
		//infer rows and 
		if( (_op == OpOpDG.RAND || _op == OpOpDG.SINIT ) &&
			OptimizerUtils.ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION )
		{
			if (_paramIndexMap.containsKey(DataExpression.RAND_DIMS)) {
				// TODO size information for tensors
				return null;
			}
			else {
				long dim1 = computeDimParameterInformation(getInput().get(_paramIndexMap.get(DataExpression.RAND_ROWS)), memo);
				long dim2 = computeDimParameterInformation(getInput().get(_paramIndexMap.get(DataExpression.RAND_COLS)), memo);
				long nnz = _sparsity >= 0 ? (long) (_sparsity * dim1 * dim2) : -1;
				if( dim1 >= 0 && dim2 >= 0 )
					return new MatrixCharacteristics(dim1, dim2, -1, nnz);
			}
		}
		else if ( _op == OpOpDG.SEQ )
		{
			Hop from = getInput().get(_paramIndexMap.get(Statement.SEQ_FROM));
			Hop to = getInput().get(_paramIndexMap.get(Statement.SEQ_TO));
			Hop incr = getInput().get(_paramIndexMap.get(Statement.SEQ_INCR)); 
			//in order to ensure correctness we also need to know from and incr
			//here, we check for the common case of seq(1,x), i.e. from=1, incr=1
			if(    from instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)from)==1
				&& incr instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)incr)==1 )
			{
				long toVal = computeDimParameterInformation(to, memo);
				if( toVal > 0 )
					return new MatrixCharacteristics(toVal, 1, -1, -1);
			}
			//here, we check for the common case of seq(x,1,-1), i.e. from=x, to=1 incr=-1
			if(    to instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)to)==1
				&& incr instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)incr)==-1 )
			{
				long fromVal = computeDimParameterInformation(from, memo);
				if( fromVal > 0 )
					return new MatrixCharacteristics(fromVal, 1, -1, -1);
			}
		}
		
		return null;
	}

	@Override
	protected ExecType optFindExecType(boolean transitive) {
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
				_etype = ExecType.SPARK;
		
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}

		//mark for recompile (forever)
		setRequiresRecompileIfNecessary();

		//always force string initialization into CP (not supported in MR)
		//similarly, sample is currently not supported in MR either
		if( _op == OpOpDG.SINIT || _op == OpOpDG.TIME ) {
			_etype = ExecType.CP;
		}
		
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		Hop input1;
		Hop input2;
		Hop input3;

		if ( _op == OpOpDG.RAND || _op == OpOpDG.SINIT || _op == OpOpDG.FRAMEINIT )
		{
			if (_dataType != DataType.TENSOR) {
				input1 = getInput().get(_paramIndexMap.get(DataExpression.RAND_ROWS)); //rows
				input2 = getInput().get(_paramIndexMap.get(DataExpression.RAND_COLS)); //cols

				//refresh rows information
				refreshRowsParameterInformation(input1);

				//refresh cols information
				refreshColsParameterInformation(input2);
			}
			// TODO size information for tensor
		}
		else if (_op == OpOpDG.SEQ ) 
		{
			//bounds computation
			input1 = getInput().get(_paramIndexMap.get(Statement.SEQ_FROM));
			input2 = getInput().get(_paramIndexMap.get(Statement.SEQ_TO)); 
			input3 = getInput().get(_paramIndexMap.get(Statement.SEQ_INCR)); 

			double from = computeBoundsInformation(input1);
			boolean fromKnown = (from != Double.MAX_VALUE);
			double to = computeBoundsInformation(input2);
			boolean toKnown = (to != Double.MAX_VALUE);
			double incr = computeBoundsInformation(input3);
			boolean incrKnown = (incr != Double.MAX_VALUE);
			if(  fromKnown && toKnown && incr == 1) {
				incr = ( from >= to ) ? -1 : 1;
			}
			
			if ( fromKnown && toKnown && incrKnown ) {
				//TODO fix parser exception handling and enable check by default
				setDim1(UtilFunctions.getSeqLength(from, to, incr, false));
				setDim2(1);
				_incr = incr;
			}
			
			//leverage high-probability information of output
			if( getDim1() == -1 && getParent().size() == 1 ) {
				Hop p = getParent().get(0);
				p.refreshSizeInformation();
				setDim1((HopRewriteUtils.isTernary(p, OpOp3.CTABLE)
					&& p.getDim1() >= 0 ) ? p.getDim1() : -1);
			}
		}
		else if (_op == OpOpDG.TIME ) {
			setDim1(0);
			setDim2(0);
			_dataType = DataType.SCALAR;
			_valueType = ValueType.INT64;
		}
		
		//refresh nnz information (for seq, sparsity is always -1)
		if( _op == OpOpDG.RAND && hasConstantValue(0.0) )
			setNnz(0);
		else if ( dimsKnown() && _sparsity>=0 ) //general case
			setNnz((long) (_sparsity * getLength()));
		else
			setNnz(-1);
	}
	

	public HashMap<String, Integer> getParamIndexMap() {
		return _paramIndexMap;
	}
	
	public Hop getParam(String key) {
		return getInput().get(getParamIndex(key));
	}
	
	public int getParamIndex(String key) {
		return _paramIndexMap.get(key);
	}
	
	public Hop getInput(String key) {
		return getInput().get(getParamIndex(key));
	}
	
	public void setInput(String key, Hop hop, boolean linkParent) {
		getInput().set(getParamIndex(key), hop);
		if( linkParent )
			hop.getParent().add(this);
	}

	public boolean hasConstantValue() 
	{
		//robustness for other operations, not specifying min/max/sparsity
		if( _op != OpOpDG.RAND )
			return false;
		
		Hop min = getInput().get(_paramIndexMap.get(DataExpression.RAND_MIN)); //min 
		Hop max = getInput().get(_paramIndexMap.get(DataExpression.RAND_MAX)); //max
		Hop sparsity = getInput().get(_paramIndexMap.get(DataExpression.RAND_SPARSITY)); //sparsity
		
		//literal value comparison
		if( min instanceof LiteralOp && max instanceof LiteralOp && sparsity instanceof LiteralOp) {
			try {
				double minVal = HopRewriteUtils.getDoubleValue((LiteralOp)min);
				double maxVal = HopRewriteUtils.getDoubleValue((LiteralOp)max);
				double sp = HopRewriteUtils.getDoubleValue((LiteralOp)sparsity);
				return (sp==1.0 && minVal == maxVal);
			}
			catch(Exception ex) {
				return false;
			}
		}
		//reference comparison (based on common subexpression elimination)
		else if ( min == max && sparsity instanceof LiteralOp ) {
			return (HopRewriteUtils.getDoubleValueSafe((LiteralOp)sparsity)==1);
		}
		
		return false;
	}

	public boolean hasConstantValue(double val) 
	{
		//string initialization does not exhibit constant values
		if( _op != OpOpDG.RAND )
			return false;
		
		boolean ret = false;
		
		Hop min = getInput().get(_paramIndexMap.get(DataExpression.RAND_MIN)); //min 
		Hop max = getInput().get(_paramIndexMap.get(DataExpression.RAND_MAX)); //max
		
		//literal value comparison
		if( min instanceof LiteralOp && max instanceof LiteralOp){
			double minVal = HopRewriteUtils.getDoubleValueSafe((LiteralOp)min);
			double maxVal = HopRewriteUtils.getDoubleValueSafe((LiteralOp)max);
			ret = (minVal == val && maxVal == val);
		}
		
		//sparsity awareness if requires
		if( ret && val != 0 ) {
			Hop sparsity = getInput().get(_paramIndexMap.get(DataExpression.RAND_SPARSITY)); //sparsity
			ret = (sparsity == null || sparsity instanceof LiteralOp
					&& HopRewriteUtils.getDoubleValueSafe((LiteralOp) sparsity) == 1);
		}
		
		return ret;
	}
	
	public boolean hasUnspecifiedSeed() {
		if (_op == OpOpDG.RAND || _op == OpOpDG.SINIT) {
			Hop seed = getInput().get(_paramIndexMap.get(DataExpression.RAND_SEED));
			return seed.getName().equals(String.valueOf(DataGenOp.UNSPECIFIED_SEED));
		}
		return false;
	}
	
	public Hop getConstantValue() {
		return getInput().get(_paramIndexMap.get(DataExpression.RAND_MIN));
	}
	
	public void setIncrementValue(double incr) {
		_incr = incr;
	}
	
	public double getIncrementValue() {
		return _incr;
	}
	
	public static long generateRandomSeed() {
		return System.nanoTime();
	}

	@Override
	@SuppressWarnings("unchecked")
	public Object clone() throws CloneNotSupportedException 
	{
		DataGenOp ret = new DataGenOp();
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._op = _op;
		ret._id = _id;
		ret._sparsity = _sparsity;
		ret._baseDir = _baseDir;
		ret._paramIndexMap = (HashMap<String, Integer>) _paramIndexMap.clone();
		ret._maxNumThreads = _maxNumThreads;
		//note: no deep cp of params since read-only 
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof DataGenOp) )
			return false;
		
		 // NOTE:
		 // This compare() method currently is invoked from Hops RewriteCommonSubexpressionElimination,
		 // which tries to merge two hops if this function returns true. However, two TIME hops should
		 // never be merged, and hence returning false.
		if (_op == OpOpDG.TIME)
			return false;
		
		DataGenOp that2 = (DataGenOp)that;
		boolean ret = (  _op == that2._op
			&& _sparsity == that2._sparsity
			&& _baseDir.equals(that2._baseDir)
			&& _paramIndexMap!=null && that2._paramIndexMap!=null
			&& _maxNumThreads == that2._maxNumThreads );
		
		if( ret ) {
			for( Entry<String,Integer> e : _paramIndexMap.entrySet() ) {
				String key1 = e.getKey();
				int pos1 = e.getValue();
				int pos2 = that2._paramIndexMap.getOrDefault(key1, -1);
				ret &= ( pos2 >=0 && that2.getInput().get(pos2)!=null
					&& getInput().get(pos1) == that2.getInput().get(pos2) );
			}
			
			//special case for rand seed (no CSE if unspecified seed because runtime generated)
			//note: if min and max is constant, we can safely merge those hops
			if( _op == OpOpDG.RAND || _op == OpOpDG.SINIT ){
				Hop seed = getInput().get(_paramIndexMap.get(DataExpression.RAND_SEED));
				Hop min = getInput().get(_paramIndexMap.get(DataExpression.RAND_MIN));
				Hop max = getInput().get(_paramIndexMap.get(DataExpression.RAND_MAX));
				if( seed.getName().equals(String.valueOf(DataGenOp.UNSPECIFIED_SEED)) && min != max )
					ret = false;
			}
		}
		return ret;
	}
}
