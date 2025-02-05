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

import java.util.ArrayList;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.Checkpoint;
import org.apache.sysds.lops.Compression;
import org.apache.sysds.lops.CumulativeOffsetBinary;
import org.apache.sysds.lops.CumulativePartialAggregate;
import org.apache.sysds.lops.Data;
import org.apache.sysds.lops.DeCompression;
import org.apache.sysds.lops.Local;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.PickByCount;
import org.apache.sysds.lops.SortKeys;
import org.apache.sysds.lops.Unary;
import org.apache.sysds.lops.UnaryCP;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.UtilFunctions;



/* Unary (cell operations): e.g, b_ij = round(a_ij)
 * 		Semantic: given a value, perform the operation (independent of other values)
 */

public class UnaryOp extends MultiThreadedHop
{
	private static final boolean ALLOW_CUMAGG_BROADCAST = true;
	private static final boolean ALLOW_CUMAGG_CACHING = false;
	
	private OpOp1 _op = null;
	
	private UnaryOp() {
		//default constructor for clone
	}

	public UnaryOp(String l, DataType dt, ValueType vt, OpOp1 o, Hop inp) {
		super(l, dt, vt);

		getInput().add(inp);
		inp.getParent().add(this);
		_op = o;
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	// this is for OpOp1, e.g. A = -B (0-B); and a=!b
	public OpOp1 getOp() {
		return _op;
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += "u(" + _op.toString().toLowerCase() + ")";
		return s;
	}

	@Override
	public boolean isGPUEnabled() {
		if(!DMLScript.USE_ACCELERATOR)
			return false;
		boolean isScalar = (    getDataType() == DataType.SCALAR //value type casts or matrix to scalar
				|| (_op == OpOp1.CAST_AS_MATRIX && getInput().get(0).getDataType()==DataType.SCALAR)
				|| (_op == OpOp1.CAST_AS_FRAME && getInput().get(0).getDataType()==DataType.SCALAR));
		if(!isScalar) {
			switch(_op) {
				case EXP:case SQRT:case LOG:case ABS:
				case ROUND:case FLOOR:case CEIL:
				case SIN:case COS: case TAN:
				case ASIN:case ACOS:case ATAN:
				case SINH:case COSH: case TANH:
				case SIGN: case SIGMOID:
				case CUMSUM: case CUMPROD:
				case CUMMIN: case CUMMAX:
				case CUMSUMPROD:
					return true;
				default:
					return false;
			}
		}
		else  {
			return false;
		}
	}
	
	@Override
	public boolean isMultiThreadedOpType() {
		return isCumulativeUnaryOperation()
			|| isExpensiveUnaryOperation();
	}
	
	@Override
	public Lop constructLops()
	{
		//reuse existing lop
		if( getLops() != null )
			return getLops();
		final int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		try {
			Hop input = getInput().get(0);
			Lop ret = null;
			switch(_op){
				case COMPRESS:
					ret = new Compression(input.constructLops(), getDataType(), getValueType(), optFindExecType(), 0, k);
					break;
				case DECOMPRESS:
					ret = new DeCompression(input.constructLops(), getDataType(), getValueType(), optFindExecType());
					break;
				case LOCAL:
					ret = new Local(input.constructLops(), getDataType(), getValueType());
					break;
				case _EVICT:
					ret = new UnaryCP(input.constructLops(), _op, getDataType(), getValueType());
					break;
				default:
					final boolean isScalarIn = getInput().get(0).getDataType() == DataType.SCALAR;
					if(getDataType() == DataType.SCALAR // value type casts or matrix to scalar
						|| (_op == OpOp1.CAST_AS_MATRIX && isScalarIn) // cast matrix
						|| (_op == OpOp1.CAST_AS_FRAME && isScalarIn)) { // cast frame
						if(_op == OpOp1.IQM)  // special handling IQM
							ret = constructLopsIQM();
						else if(_op == OpOp1.MEDIAN) 
							ret = constructLopsMedian();
						else  // general case SCALAR/CAST (always in CP) & always single threaded
							ret = new UnaryCP(input.constructLops(), _op, getDataType(), getValueType(), k);
					}
					else { // general case MATRIX
						ExecType et = optFindExecType();
						// special handling cumsum/cumprod/cummin/cumsum
						if(isCumulativeUnaryOperation() && !(et == ExecType.CP || et == ExecType.GPU)) {
							// TODO additional physical operation if offsets fit in memory
							ret = constructLopsSparkCumulativeUnary();
						}
						else if(_op == OpOp1.CAST_AS_FRAME && getInput().size() == 2) {
							throw new NotImplementedException();
						}
						else {// default unary
							final boolean inplace = OptimizerUtils.ALLOW_UNARY_UPDATE_IN_PLACE &&
								input.getParent().size() == 1 && (!(input instanceof DataOp) || !((DataOp) input).isRead());

							ret = new Unary(input.constructLops(), _op, getDataType(), getValueType(), et, k, inplace);
						}
					}
			}
			setOutputDimensions(ret);
			setLineNumbers(ret);
			setLops(ret);
		}
		catch (Exception e) {
			throw new HopsException(this.printErrorLocation() + "error constructing Lops for UnaryOp Hop -- \n " , e);
		}
		
		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();

		return getLops();
	}
	

	private Lop constructLopsMedian() 
	{
		ExecType et = optFindExecType();
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		SortKeys sort = SortKeys.constructSortByValueLop(
							getInput().get(0).constructLops(), 
							SortKeys.OperationTypes.WithoutWeights, 
							DataType.MATRIX, ValueType.FP64, et, k );
		sort.getOutputParameters().setDimensions(
				getInput().get(0).getDim1(),
				getInput().get(0).getDim2(),
				getInput().get(0).getBlocksize(),
				getInput().get(0).getNnz());
		PickByCount pick = new PickByCount(
				sort,
				Data.createLiteralLop(ValueType.FP64, Double.toString(0.5)),
				getDataType(),
				getValueType(),
				PickByCount.OperationTypes.MEDIAN, et, true);

		pick.getOutputParameters().setDimensions(
			getDim1(), getDim2(), getBlocksize(), getNnz());
		setLineNumbers(pick);
		setLops(pick);
		
		return pick;
	}
	
	private Lop constructLopsIQM() 
	{
		ExecType et = optFindExecType();
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		Hop input = getInput().get(0);
		SortKeys sort = SortKeys.constructSortByValueLop(
			input.constructLops(),
			SortKeys.OperationTypes.WithoutWeights,
			DataType.MATRIX, ValueType.FP64, et, k );
		sort.getOutputParameters().setDimensions(
				input.getDim1(),
				input.getDim2(),
				input.getBlocksize(),
				input.getNnz());
		PickByCount pick = new PickByCount(sort, null,
				getDataType(),getValueType(),
				PickByCount.OperationTypes.IQM, et, true);

		pick.getOutputParameters().setDimensions(
			getDim1(), getDim2(), getBlocksize(), getNnz());
		setLineNumbers(pick);
		
		return pick;
	}
	
	@SuppressWarnings("unused")
	private Lop constructLopsSparkCumulativeUnary() 
	{
		Hop input = getInput().get(0);
		long rlen = input.getDim1();
		long clen = input.getDim2();
		long blen = input.getBlocksize();
		boolean force = !dimsKnown() || _etypeForced == ExecType.SPARK;
		AggOp aggtype = getCumulativeAggType();
		Lop X = input.constructLops();
		
		//special case single row block (no offsets needed)
		if( rlen > 0 && clen > 0 && rlen <= blen ) {
			Lop offset = HopRewriteUtils.createDataGenOpByVal(new LiteralOp(1), new LiteralOp(clen),
					null, DataType.MATRIX, ValueType.FP64, getCumulativeInitValue()).constructLops();
			return constructCumOffBinary(X, offset, aggtype, rlen, clen, blen);
		}
		
		Lop TEMP = X;
		ArrayList<Lop> DATA = new ArrayList<>();
		int level = 0;
		
		//recursive preaggregation until aggregates fit into CP memory budget
		while( ((2*OptimizerUtils.estimateSize(TEMP.getOutputParameters().getNumRows(), clen) + OptimizerUtils.estimateSize(1, clen)) 
			> OptimizerUtils.getLocalMemBudget() && TEMP.getOutputParameters().getNumRows()>1) || force )
		{
			//caching within multi-level cascades
			if( ALLOW_CUMAGG_CACHING && level > 0 ) {
				Lop oldTEMP = TEMP;
				TEMP = new Checkpoint(oldTEMP, getDataType(), getValueType(), Checkpoint.getDefaultStorageLevelString());
				TEMP.getOutputParameters().setDimensions(oldTEMP.getOutputParameters());
				setLineNumbers(TEMP);
			}
			DATA.add(TEMP);
	
			//preaggregation per block (for spark, the CumulativePartialAggregate subsumes both
			//the preaggregation and subsequent block aggregation)
			long rlenAgg = (long)Math.ceil((double)TEMP.getOutputParameters().getNumRows()/blen);
			Lop preagg = new CumulativePartialAggregate(TEMP, DataType.MATRIX, ValueType.FP64, aggtype, ExecType.SPARK);
			preagg.getOutputParameters().setDimensions(rlenAgg, clen, blen, -1);
			setLineNumbers(preagg);
			
			TEMP = preagg;
			level++;
			force = false; //in case of unknowns, generate one level
		}
		
		//in-memory cum sum (of partial aggregates)
		//marked for update in-place if there was at least one aggregation level
		if( TEMP.getOutputParameters().getNumRows()!=1 ){
			int k = OptimizerUtils.getConstrainedNumThreads( _maxNumThreads );
			Unary unary1 = new Unary( TEMP, _op, DataType.MATRIX, ValueType.FP64, ExecType.CP, k, TEMP!=X);
			unary1.getOutputParameters().setDimensions(TEMP.getOutputParameters().getNumRows(), clen, blen, -1);
			setLineNumbers(unary1);
			TEMP = unary1;
		}
		
		//split, group and mr cumsum
		while( level-- > 0  ) {
			TEMP = constructCumOffBinary(DATA.get(level),
				TEMP, aggtype, rlen, clen, blen);
		}
		
		return TEMP;
	}
	
	private Lop constructCumOffBinary(Lop data, Lop offset, AggOp aggtype, long rlen, long clen, long blen) {
		//(for spark, the CumulativeOffsetBinary subsumes both the split aggregate and 
		//the subsequent offset binary apply of split aggregates against the original data)
		double initValue = getCumulativeInitValue();
		boolean broadcast = ALLOW_CUMAGG_BROADCAST
			&& OptimizerUtils.checkSparkBroadcastMemoryBudget(OptimizerUtils.estimateSize(
			offset.getOutputParameters().getNumRows(), offset.getOutputParameters().getNumCols()));
		
		CumulativeOffsetBinary binary = new CumulativeOffsetBinary(data, offset, 
				DataType.MATRIX, ValueType.FP64, initValue, broadcast, aggtype, ExecType.SPARK);
		binary.getOutputParameters().setDimensions(rlen, clen, blen, -1);
		setLineNumbers(binary);
		return binary;
	}

	private AggOp getCumulativeAggType() {
		switch( _op ) {
			case CUMSUM:     return AggOp.SUM;
			case CUMPROD:    return AggOp.PROD;
			case CUMMIN:     return AggOp.MIN;
			case CUMMAX:     return AggOp.MAX;
			case CUMSUMPROD: return AggOp.SUM_PROD;
			default:         return null;
		}
	}

	private double getCumulativeInitValue() {
		switch( _op ) {
			case CUMSUMPROD: 
			case CUMSUM:  return 0;
			case CUMPROD: return 1;
			case CUMMIN:  return Double.POSITIVE_INFINITY;
			case CUMMAX:  return Double.NEGATIVE_INFINITY;
			default:      return Double.NaN;
		}
	}
	
		
	@Override
	public void computeMemEstimate(MemoTable memo) {
		//overwrites default hops behavior
		super.computeMemEstimate(memo);

		if( isMetadataOperation() ) {
			_memEstimate = OptimizerUtils.INT_SIZE;
			//_outputMemEstimate = OptimizerUtils.INT_SIZE;
			//_processingMemEstimate = 0;
		}
	}

	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{
		double sparsity = -1;
		if (isGPUEnabled()) {
			sparsity = 1.0; // Output is always dense (for now) on the GPU
		} else {
			sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		}
		return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity, getDataType());
	}
	
	@Override
	protected double computeIntermediateMemEstimate(long dim1, long dim2, long nnz)
	{
		double ret = 0;
		
		if( _op == OpOp1.IQM || _op == OpOp1.MEDIAN ) {
			// buffer (=2*input_size) and output (=input_size) for SORT operation
			// getMemEstimate works for both cases of known dims and worst-case stats
			ret = getInput().get(0).getMemEstimate() * 3; 
		}
		else if( isCumulativeUnaryOperation() ) {
			//account for potential final dense-sparse transformation (worst-case sparse representation)
			ret += MatrixBlock.estimateSizeSparseInMemory(dim1, dim2,
				MatrixBlock.SPARSITY_TURN_POINT - UtilFunctions.DOUBLE_EPS);
		}

		if (isGPUEnabled()) {
			// Intermediate memory required to convert sparse to dense
			ret += OptimizerUtils.estimateSize(dim1, dim2); 
		}
		
		return ret;
	}
	
	@Override
	protected DataCharacteristics inferOutputCharacteristics( MemoTable memo ) {
		DataCharacteristics dc = memo.getAllInputStats(getInput().get(0));
		DataCharacteristics ret = null;
		if( dc.dimsKnown() ) {
			if( _op==OpOp1.ABS || _op==OpOp1.COS || _op==OpOp1.SIN || _op==OpOp1.TAN 
				|| _op==OpOp1.ACOS || _op==OpOp1.ASIN || _op==OpOp1.ATAN  
				|| _op==OpOp1.COSH || _op==OpOp1.SINH || _op==OpOp1.TANH 
				|| _op==OpOp1.SQRT || _op==OpOp1.ROUND  
				|| _op==OpOp1.SPROP || _op== OpOp1.COMPRESS || _op== OpOp1.DECOMPRESS
				|| _op==OpOp1.LOCAL) //sparsity preserving
			{
				ret = new MatrixCharacteristics(dc.getRows(), dc.getCols(), -1, dc.getNonZeros());
			}
			else if( _op==OpOp1.CUMSUMPROD )
				ret = new MatrixCharacteristics(dc.getRows(), 1, -1, -1);
			else 
				ret = new MatrixCharacteristics(dc.getRows(), dc.getCols(), -1, -1);
		}
		
		return ret;
	}

	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}

	private boolean isInMemoryOperation() 
	{
		return ( _op == OpOp1.INVERSE );
	}

	public boolean isCumulativeUnaryOperation()  {
		return (_op == OpOp1.CUMSUM 
			|| _op == OpOp1.CUMPROD
			|| _op == OpOp1.CUMMIN
			|| _op == OpOp1.CUMMAX
			|| _op == OpOp1.CUMSUMPROD);
	}

	public boolean isCastUnaryOperation() {
		return (_op == OpOp1.CAST_AS_MATRIX
			|| _op == OpOp1.CAST_AS_SCALAR
			|| _op == OpOp1.CAST_AS_FRAME
			|| _op == OpOp1.CAST_AS_LIST
			|| _op == OpOp1.CAST_AS_BOOLEAN
			|| _op == OpOp1.CAST_AS_DOUBLE
			|| _op == OpOp1.CAST_AS_INT);
	}
	
	public boolean isExpensiveUnaryOperation() {
		return (_op == OpOp1.EXP || _op == OpOp1.LOG || _op == OpOp1.LOG_NZ
			|| _op == OpOp1.ROUND || _op == OpOp1.FLOOR || _op == OpOp1.CEIL
			|| _op == OpOp1.SIGMOID || _op == OpOp1.SPROP || _op == OpOp1.SOFTMAX
			|| _op == OpOp1.TAN || _op == OpOp1.TANH || _op == OpOp1.ATAN
			|| _op == OpOp1.COMPRESS || _op == OpOp1.DECOMPRESS
			|| _op == OpOp1.MEDIAN || _op == OpOp1.IQM);
	}
	
	public boolean isMetadataOperation() {
		return _op == OpOp1.NROW
			|| _op == OpOp1.NCOL
			|| _op == OpOp1.LENGTH
			|| _op == OpOp1.EXISTS
			|| _op == OpOp1.LINEAGE
			|| _op == OpOp1.CAST_AS_LIST;
	}
	
	@Override
	protected ExecType optFindExecType(boolean transitive) 
	{
		checkAndSetForcedPlatform();
		
		if( _etypeForced != null )
		{
			_etype = _etypeForced;
		}
		else 
		{
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) {
				_etype = findExecTypeByMemEstimate();
			}
			// Choose CP, if the input dimensions are below threshold or if the input is a vector
			// Also, matrix inverse is currently implemented only in CP (through commons math)
			else if ( getInput().get(0).areDimsBelowThreshold() || getInput().get(0).isVector() 
						|| isInMemoryOperation() )
			{
				_etype = ExecType.CP;
			}
			else 
			{
				_etype = ExecType.SPARK;
			}
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}
	
		//spark-specific decision refinement (execute unary w/ spark input and 
		//single parent also in spark because it's likely cheap and reduces intermediates)
		if( _etype == ExecType.CP && _etypeForced != ExecType.CP
			&& getInput().get(0).optFindExecType() == ExecType.SPARK 
			&& getDataType().isMatrix() 
			&& !isCumulativeUnaryOperation() && !isCastUnaryOperation()
			&& _op!=OpOp1.MEDIAN && _op!=OpOp1.IQM
			&& !(getInput().get(0) instanceof DataOp)    //input is not checkpoint
			&& getInput().get(0).getParent().size()==1 ) //unary is only parent
		{
			//pull unary operation into spark 
			_etype = ExecType.SPARK;
		}
		
		//mark for recompile (forever)
		setRequiresRecompileIfNecessary();
		
		//ensure cp exec type for single-node operations
		if( _op == OpOp1.PRINT || _op == OpOp1.ASSERT || _op == OpOp1.STOP || _op == OpOp1.TYPEOF
			|| _op == OpOp1.INVERSE || _op == OpOp1.EIGEN || _op == OpOp1.CHOLESKY || _op == OpOp1.DET
			||_op == OpOp1.SVD || _op == OpOp1.SQRT_MATRIX_JAVA
			|| getInput().get(0).getDataType() == DataType.LIST || isMetadataOperation() )
		{
			_etype = ExecType.CP;
		} else {
			setRequiresRecompileIfNecessary();
		}
		
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		Hop input = getInput().get(0);
		if ( getDataType() == DataType.SCALAR )  {
			//do nothing always known
		}
		else if( (_op == OpOp1.CAST_AS_MATRIX || _op == OpOp1.CAST_AS_FRAME
			|| _op == OpOp1.CAST_AS_SCALAR) && input.getDataType()==DataType.LIST ){
			//handle two cases of list of scalars or list of single matrix
			setDim1( input.getLength() > 1 ? input.getLength() : -1 );
			setDim2( input.getLength() > 1 ? 1 : -1 );
		}
		else if( (_op == OpOp1.CAST_AS_MATRIX || _op == OpOp1.CAST_AS_FRAME)
			&& input.getDataType()==DataType.SCALAR )
		{
			//prevent propagating 0 from scalar (which would be interpreted as unknown)
			setDim1( 1 );
			setDim2( 1 );
		}
		else if( _op == OpOp1.CAST_AS_LIST ) {
			setDim1(-1);
			setDim2(1);
		}
		else if ( _op==OpOp1.CUMSUMPROD ) {
			setDim1(input.getDim1());
			setDim2(1);
		}
		else if(_op == OpOp1.TYPEOF || _op == OpOp1.DETECTSCHEMA || _op == OpOp1.COLNAMES) {
			setDim1(1);
			setDim2(input.getDim2());
		}
		else //general case
		{
			// If output is a Matrix then this operation is of type (B = op(A))
			// Dimensions of B are same as that of A, and sparsity may/maynot change
			// note: round, sin, cos can introduce new zeros for non-zero inputs
			setDim1( input.getDim1() );
			setDim2( input.getDim2() );
			if( _op==OpOp1.ABS || _op==OpOp1.SQRT || _op==OpOp1.SPROP
				|| _op==OpOp1.COMPRESS || _op==OpOp1.DECOMPRESS || _op==OpOp1.LOCAL) //sparsity preserving
			{
				setNnz( input.getNnz() );
			}

			// if the input is compressed then set the output to be compressed as well.
			if(input._compressedOutput && ! (_op==OpOp1.DECOMPRESS)){
				setCompressedOutput(true);
				// Setting the compressed output to be 2 x larger.
				// Just in case we change the compressed structure slightly.
				// this value is overwritten with correct size once the hop is executed
				// TODO handle overlapping state, since some operations would not lead to compressed output.
				setCompressedSize(input.compressedSize() * 2);
			}
		}
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		UnaryOp ret = new UnaryOp();
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._op = _op;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof UnaryOp) )
			return false;
		
		/*
		 * NOTE:
		 * This compare() method currently is invoked from Hops RewriteCommonSubexpressionElimination,
		 * which tries to merge two hops if this function returns true. However, two PRINT hops should
		 * never be merged, and hence returning false.
		 * 
		 * If this method needs to be used elsewhere, then it must be refactored accordingly.
		 */
		if( _op == OpOp1.PRINT )
			return false;
		
		UnaryOp that2 = (UnaryOp)that;
		return (   _op == that2._op
				&& getInput().get(0) == that2.getInput().get(0));
	}
}
