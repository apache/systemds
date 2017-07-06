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

package org.apache.sysml.hops;

import java.util.ArrayList;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.Hop.MultiThreadedHop;
import org.apache.sysml.lops.Aggregate;
import org.apache.sysml.lops.Aggregate.OperationTypes;
import org.apache.sysml.lops.CombineUnary;
import org.apache.sysml.lops.CumulativeOffsetBinary;
import org.apache.sysml.lops.CumulativePartialAggregate;
import org.apache.sysml.lops.CumulativeSplitAggregate;
import org.apache.sysml.lops.Data;
import org.apache.sysml.lops.Group;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.lops.PartialAggregate;
import org.apache.sysml.lops.PartialAggregate.CorrectionLocationType;
import org.apache.sysml.lops.PickByCount;
import org.apache.sysml.lops.SortKeys;
import org.apache.sysml.lops.Unary;
import org.apache.sysml.lops.UnaryCP;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;


/* Unary (cell operations): e.g, b_ij = round(a_ij)
 * 		Semantic: given a value, perform the operation (independent of other values)
 */

public class UnaryOp extends Hop implements MultiThreadedHop
{
	private OpOp1 _op = null;
	
	private int _maxNumThreads = -1; //-1 for unlimited
	
	
	private UnaryOp() {
		//default constructor for clone
	}
	
	public UnaryOp(String l, DataType dt, ValueType vt, OpOp1 o, Hop inp) {
		super(l, dt, vt);

		getInput().add(0, inp);
		inp.getParent().add(this);
		_op = o;
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	@Override
	public void checkArity() throws HopsException {
		HopsException.check(_input.size() == 1, this, "should have arity 1 but has arity %d", _input.size());
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
	public void setMaxNumThreads( int k ) {
		_maxNumThreads = k;
	}
	
	@Override
	public int getMaxNumThreads() {
		return _maxNumThreads;
	}
	
	@Override
	public Lop constructLops()
		throws HopsException, LopsException 
	{		
		//reuse existing lop
		if( getLops() != null )
			return getLops();
		
		try 
		{
			Hop input = getInput().get(0);
			
			if(    getDataType() == DataType.SCALAR //value type casts or matrix to scalar
				|| (_op == OpOp1.CAST_AS_MATRIX && getInput().get(0).getDataType()==DataType.SCALAR)
				|| (_op == OpOp1.CAST_AS_FRAME && getInput().get(0).getDataType()==DataType.SCALAR))
			{
				if (_op == Hop.OpOp1.IQM)  //special handling IQM
				{
					Lop iqmLop = constructLopsIQM();
					setLops(iqmLop);
				} 
				else if(_op == Hop.OpOp1.MEDIAN) {
					Lop medianLop = constructLopsMedian();
					setLops(medianLop);
				}
				else //general case SCALAR/CAST (always in CP)
				{
					UnaryCP.OperationTypes optype = HopsOpOp1LopsUS.get(_op);
					if( optype == null )
						throw new HopsException("Unknown UnaryCP lop type for UnaryOp operation type '"+_op+"'");
					
					UnaryCP unary1 = null;
					if((_op == Hop.OpOp1.NROW || _op == Hop.OpOp1.NCOL || _op == Hop.OpOp1.LENGTH) &&
						input instanceof UnaryOp && ((UnaryOp) input).getOp() == OpOp1.SELP) {
						// Dimensions does not change during sel+ operation.
						// This case is helpful to avoid unnecessary sel+ operation for fused maxpooling.
						unary1 = new UnaryCP(input.getInput().get(0).constructLops(), optype, getDataType(), getValueType());
					}
					else
						unary1 = new UnaryCP(input.constructLops(), optype, getDataType(), getValueType());
					setOutputDimensions(unary1);
					setLineNumbers(unary1);

					setLops(unary1);
				}
			} 
			else //general case MATRIX
			{
				ExecType et = optFindExecType();
				
				//special handling cumsum/cumprod/cummin/cumsum
				if( isCumulativeUnaryOperation() && et != ExecType.CP )  
				{
					//TODO additional physical operation if offsets fit in memory
					Lop cumsumLop = null;
					if( et == ExecType.MR )
						cumsumLop = constructLopsMRCumulativeUnary();
					else
						cumsumLop = constructLopsSparkCumulativeUnary();
					setLops(cumsumLop);
				}
				else //default unary 
				{
					int k = isCumulativeUnaryOperation() ? OptimizerUtils.getConstrainedNumThreads( _maxNumThreads ) : 1;
					switch(_op) {
						case SELP:case EXP:case SQRT:case LOG:case ABS:
						case ROUND:case FLOOR:case CEIL:
						case SIN:case COS: case TAN:case ASIN:case ACOS:case ATAN:
						case SIGN:
							et = findGPUExecTypeByMemEstimate(et);
							break;
						default:
					}
					Unary unary1 = new Unary(input.constructLops(), HopsOpOp1LopsU.get(_op), 
							                 getDataType(), getValueType(), et, k);
					setOutputDimensions(unary1);
					setLineNumbers(unary1);
					setLops(unary1);
				}
			}
		} 
		catch (Exception e) 
		{
			throw new HopsException(this.printErrorLocation() + "error constructing Lops for UnaryOp Hop -- \n " , e);
		}
		
		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();
		
		return getLops();
	}
	

	private Lop constructLopsMedian() 
		throws HopsException, LopsException 
	{
		ExecType et = optFindExecType();

		if ( et == ExecType.MR ) {
			CombineUnary combine = CombineUnary.constructCombineLop(
					getInput().get(0).constructLops(),
					getDataType(), getValueType());

			SortKeys sort = SortKeys.constructSortByValueLop(
					combine, SortKeys.OperationTypes.WithoutWeights,
					DataType.MATRIX, ValueType.DOUBLE, et);

			combine.getOutputParameters().setDimensions(getDim1(),
					getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());

			// Sort dimensions are same as the first input
			sort.getOutputParameters().setDimensions(
					getInput().get(0).getDim1(),
					getInput().get(0).getDim2(),
					getInput().get(0).getRowsInBlock(),
					getInput().get(0).getColsInBlock(), 
					getInput().get(0).getNnz());

			// If only a single quantile is computed, then "pick" operation executes in CP.
			ExecType et_pick = ExecType.CP;
			
			PickByCount pick = new PickByCount(
					sort,
					Data.createLiteralLop(ValueType.DOUBLE, Double.toString(0.5)),
					getDataType(),
					getValueType(),
					PickByCount.OperationTypes.MEDIAN, et_pick, false);

			pick.getOutputParameters().setDimensions(getDim1(),
					getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			setLineNumbers(pick);

			return pick;
		}
		else {
			SortKeys sort = SortKeys.constructSortByValueLop(
								getInput().get(0).constructLops(), 
								SortKeys.OperationTypes.WithoutWeights, 
								DataType.MATRIX, ValueType.DOUBLE, et );
			sort.getOutputParameters().setDimensions(
					getInput().get(0).getDim1(),
					getInput().get(0).getDim2(),
					getInput().get(0).getRowsInBlock(),
					getInput().get(0).getColsInBlock(), 
					getInput().get(0).getNnz());
			PickByCount pick = new PickByCount(
					sort,
					Data.createLiteralLop(ValueType.DOUBLE, Double.toString(0.5)),
					getDataType(),
					getValueType(),
					PickByCount.OperationTypes.MEDIAN, et, true);

			pick.getOutputParameters().setDimensions(getDim1(),
					getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			setLineNumbers(pick);
			setLops(pick);
			
			return pick;
		}
	}
	
	private Lop constructLopsIQM() 
		throws HopsException, LopsException
	{

		ExecType et = optFindExecType();

		Hop input = getInput().get(0);
		if ( et == ExecType.MR ) {
			CombineUnary combine = CombineUnary.constructCombineLop(input.constructLops(),
							                       DataType.MATRIX, getValueType());
			combine.getOutputParameters().setDimensions(
					input.getDim1(),
					input.getDim2(), 
					input.getRowsInBlock(),
					input.getColsInBlock(),
					input.getNnz());

			SortKeys sort = SortKeys.constructSortByValueLop(combine,
							           SortKeys.OperationTypes.WithoutWeights,
							           DataType.MATRIX, ValueType.DOUBLE, ExecType.MR);

			// Sort dimensions are same as the first input
			sort.getOutputParameters().setDimensions(
					input.getDim1(),
					input.getDim2(),
					input.getRowsInBlock(),
					input.getColsInBlock(),
					input.getNnz());

			Data lit = Data.createLiteralLop(ValueType.DOUBLE, Double.toString(0.25));
			
			lit.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
            			
			PickByCount pick = new PickByCount(
					sort, lit, DataType.MATRIX, getValueType(),
					PickByCount.OperationTypes.RANGEPICK);

			pick.getOutputParameters().setDimensions(-1, -1,  
					getRowsInBlock(), getColsInBlock(), -1);
			setLineNumbers(pick);
			
			PartialAggregate pagg = new PartialAggregate(
					pick, HopsAgg2Lops.get(Hop.AggOp.SUM),
					HopsDirection2Lops.get(Hop.Direction.RowCol),
					DataType.MATRIX, getValueType());
			setLineNumbers(pagg);

			// Set the dimensions of PartialAggregate LOP based on the
			// direction in which aggregation is performed
			pagg.setDimensionsBasedOnDirection(getDim1(),
						getDim2(), getRowsInBlock(),
						getColsInBlock());

			Group group1 = new Group(
					pagg, Group.OperationTypes.Sort, DataType.MATRIX,
					getValueType());
			group1.getOutputParameters().setDimensions(getDim1(),
					getDim2(), getRowsInBlock(),
					getColsInBlock(), getNnz());
			setLineNumbers(group1);

			Aggregate agg1 = new Aggregate(
					group1, HopsAgg2Lops.get(Hop.AggOp.SUM),
					DataType.MATRIX, getValueType(), ExecType.MR);
			agg1.getOutputParameters().setDimensions(getDim1(),
					getDim2(), getRowsInBlock(),
					getColsInBlock(), getNnz());
			agg1.setupCorrectionLocation(pagg.getCorrectionLocation());
			setLineNumbers(agg1);
			
			UnaryCP unary1 = new UnaryCP(
					agg1, HopsOpOp1LopsUS.get(OpOp1.CAST_AS_SCALAR),
					getDataType(), getValueType());
			unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			setLineNumbers(unary1);
		
			Unary iqm = new Unary(sort, unary1, Unary.OperationTypes.MR_IQM, DataType.SCALAR, ValueType.DOUBLE, ExecType.CP);
			iqm.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			setLineNumbers(iqm);

			return iqm;
		}
		else {
			SortKeys sort = SortKeys.constructSortByValueLop(
					input.constructLops(), 
					SortKeys.OperationTypes.WithoutWeights, 
					DataType.MATRIX, ValueType.DOUBLE, et );
			sort.getOutputParameters().setDimensions(
					input.getDim1(),
					input.getDim2(),
					input.getRowsInBlock(),
					input.getColsInBlock(),
					input.getNnz());
			PickByCount pick = new PickByCount(sort, null,
					getDataType(),getValueType(),
					PickByCount.OperationTypes.IQM, et, true);

			pick.getOutputParameters().setDimensions(getDim1(),
					getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			setLineNumbers(pick);
			
			return pick;
		}
	}
	
	/**
	 * MR Cumsum is currently based on a multipass algorithm of (1) preaggregation and (2) subsequent offsetting. 
	 * Note that we currently support one robust physical operator but many alternative
	 * realizations are possible for specific scenarios (e.g., when the preaggregated intermediate
	 * fit into the map task memory budget) or by creating custom job types.
	 * 
	 * @return low-level operator
	 * @throws HopsException if HopsException occurs
	 * @throws LopsException if LopsException occurs
	 */
	private Lop constructLopsMRCumulativeUnary() 
		throws HopsException, LopsException 
	{
		Hop input = getInput().get(0);
		long rlen = input.getDim1();
		long clen = input.getDim2();
		long brlen = input.getRowsInBlock();
		long bclen = input.getColsInBlock();
		boolean force = !dimsKnown() || _etypeForced == ExecType.MR;
		OperationTypes aggtype = getCumulativeAggType();
		
		Lop X = input.constructLops();
		Lop TEMP = X;
		ArrayList<Lop> DATA = new ArrayList<Lop>();
		int level = 0;
		
		//recursive preaggregation until aggregates fit into CP memory budget
		while( ((2*OptimizerUtils.estimateSize(TEMP.getOutputParameters().getNumRows(), clen) + OptimizerUtils.estimateSize(1, clen)) 
				 > OptimizerUtils.getLocalMemBudget()
			   && TEMP.getOutputParameters().getNumRows()>1) || force )
		{
			DATA.add(TEMP);
	
			//preaggregation per block
			long rlenAgg = (long)Math.ceil((double)TEMP.getOutputParameters().getNumRows()/brlen);
			Lop preagg = new CumulativePartialAggregate(TEMP, DataType.MATRIX, ValueType.DOUBLE, aggtype, ExecType.MR);
			preagg.getOutputParameters().setDimensions(rlenAgg, clen, brlen, bclen, -1);
			setLineNumbers(preagg);
			
			Group group = new Group( preagg, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE );
			group.getOutputParameters().setDimensions(rlenAgg, clen, brlen, bclen, -1);
			setLineNumbers(group);
			
			Aggregate agg = new Aggregate(group, HopsAgg2Lops.get(AggOp.SUM), getDataType(), getValueType(), ExecType.MR);
			agg.getOutputParameters().setDimensions(rlenAgg, clen, brlen, bclen, -1);
			agg.setupCorrectionLocation(CorrectionLocationType.NONE); // aggregation uses kahanSum but the inputs do not have correction values
			setLineNumbers(agg);
			TEMP = agg;	
			level++;
			force = false; //in case of unknowns, generate one level
		}
		
		//in-memory cum sum (of partial aggregates)
		if( TEMP.getOutputParameters().getNumRows()!=1 ) {
			int k = OptimizerUtils.getConstrainedNumThreads( _maxNumThreads );					
			Unary unary1 = new Unary( TEMP, HopsOpOp1LopsU.get(_op), DataType.MATRIX, ValueType.DOUBLE, ExecType.CP, k);
			unary1.getOutputParameters().setDimensions(TEMP.getOutputParameters().getNumRows(), clen, brlen, bclen, -1);
			setLineNumbers(unary1);
			TEMP = unary1;
		}
		
		//split, group and mr cumsum
		while( level-- > 0  ) {
			double init = getCumulativeInitValue();
			CumulativeSplitAggregate split = new CumulativeSplitAggregate(TEMP, DataType.MATRIX, ValueType.DOUBLE, init);
			split.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, -1);
			setLineNumbers(split);
			
			Group group1 = new Group( DATA.get(level), Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE );
			group1.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, -1);
			setLineNumbers(group1);
			
			Group group2 = new Group( split, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE );
			group2.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, -1);
			setLineNumbers(group2);
			
			CumulativeOffsetBinary binary = new CumulativeOffsetBinary(group1, group2, 
					DataType.MATRIX, ValueType.DOUBLE, aggtype, ExecType.MR);
			binary.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, -1);
			setLineNumbers(binary);
			TEMP = binary;
		}
		
		return TEMP;
	}

	private Lop constructLopsSparkCumulativeUnary() 
		throws HopsException, LopsException 
	{
		Hop input = getInput().get(0);
		long rlen = input.getDim1();
		long clen = input.getDim2();
		long brlen = input.getRowsInBlock();
		long bclen = input.getColsInBlock();
		boolean force = !dimsKnown() || _etypeForced == ExecType.SPARK;
		OperationTypes aggtype = getCumulativeAggType();
		
		Lop X = input.constructLops();
		Lop TEMP = X;
		ArrayList<Lop> DATA = new ArrayList<Lop>();
		int level = 0;
		
		//recursive preaggregation until aggregates fit into CP memory budget
		while( ((2*OptimizerUtils.estimateSize(TEMP.getOutputParameters().getNumRows(), clen) + OptimizerUtils.estimateSize(1, clen)) 
				 > OptimizerUtils.getLocalMemBudget()
			   && TEMP.getOutputParameters().getNumRows()>1) || force )
		{
			DATA.add(TEMP);
	
			//preaggregation per block (for spark, the CumulativePartialAggregate subsumes both
			//the preaggregation and subsequent block aggregation)
			long rlenAgg = (long)Math.ceil((double)TEMP.getOutputParameters().getNumRows()/brlen);
			Lop preagg = new CumulativePartialAggregate(TEMP, DataType.MATRIX, ValueType.DOUBLE, aggtype, ExecType.SPARK);
			preagg.getOutputParameters().setDimensions(rlenAgg, clen, brlen, bclen, -1);
			setLineNumbers(preagg);
			
			TEMP = preagg;	
			level++;
			force = false; //in case of unknowns, generate one level
		}
		
		//in-memory cum sum (of partial aggregates)
		if( TEMP.getOutputParameters().getNumRows()!=1 ){
			int k = OptimizerUtils.getConstrainedNumThreads( _maxNumThreads );					
			Unary unary1 = new Unary( TEMP, HopsOpOp1LopsU.get(_op), DataType.MATRIX, ValueType.DOUBLE, ExecType.CP, k);
			unary1.getOutputParameters().setDimensions(TEMP.getOutputParameters().getNumRows(), clen, brlen, bclen, -1);
			setLineNumbers(unary1);
			TEMP = unary1;
		}
		
		//split, group and mr cumsum
		while( level-- > 0  ) {
			//(for spark, the CumulativeOffsetBinary subsumes both the split aggregate and 
			//the subsequent offset binary apply of split aggregates against the original data)
			double initValue = getCumulativeInitValue();
			CumulativeOffsetBinary binary = new CumulativeOffsetBinary(DATA.get(level), TEMP, 
					DataType.MATRIX, ValueType.DOUBLE, initValue, aggtype, ExecType.SPARK);
			binary.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, -1);
			setLineNumbers(binary);
			TEMP = binary;
		}
		
		return TEMP;
	}

	private OperationTypes getCumulativeAggType()
	{
		switch( _op ) {
			case CUMSUM: 	return OperationTypes.KahanSum;
			case CUMPROD: 	return OperationTypes.Product;
			case CUMMIN: 	return OperationTypes.Min;
			case CUMMAX: 	return OperationTypes.Max;
			default: 		return null;
		}
	}

	private double getCumulativeInitValue()
	{
		switch( _op ) {
			case CUMSUM: 	return 0;
			case CUMPROD: 	return 1;
			case CUMMIN: 	return Double.MAX_VALUE;
			case CUMMAX: 	return -Double.MAX_VALUE;
			default: 		return Double.NaN;
		}
	}
	
		
	@Override
	public void computeMemEstimate(MemoTable memo)
	{
		//overwrites default hops behavior
		super.computeMemEstimate(memo);
		
		if( _op == Hop.OpOp1.NROW || _op == Hop.OpOp1.NCOL ) //specific case for meta data ops
		{
			_memEstimate = OptimizerUtils.INT_SIZE;
			//_outputMemEstimate = OptimizerUtils.INT_SIZE;
			//_processingMemEstimate = 0;
		}
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
		double ret = 0;
		
		if ( _op == OpOp1.IQM  || _op == OpOp1.MEDIAN) {
			// buffer (=2*input_size) and output (=input_size) for SORT operation
			// getMemEstimate works for both cases of known dims and worst-case stats
			ret = getInput().get(0).getMemEstimate() * 3; 
		}
		
		return ret;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		long[] ret = null;
	
		Hop input = getInput().get(0);
		MatrixCharacteristics mc = memo.getAllInputStats(input);
		if( mc.dimsKnown() ) {
			if( _op==OpOp1.ABS || _op==OpOp1.COS || _op==OpOp1.SIN || _op==OpOp1.TAN 
				|| _op==OpOp1.ACOS || _op==OpOp1.ASIN || _op==OpOp1.ATAN  
				|| _op==OpOp1.SQRT || _op==OpOp1.ROUND  
				|| _op==OpOp1.SPROP || _op==OpOp1.SELP ) //sparsity preserving
			{
				ret = new long[]{mc.getRows(), mc.getCols(), mc.getNonZeros()};
			}
			else 
				ret = new long[]{mc.getRows(), mc.getCols(), -1};	
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

	public boolean isCumulativeUnaryOperation() 
	{
		return (   _op == OpOp1.CUMSUM 
				|| _op == OpOp1.CUMPROD
				|| _op == OpOp1.CUMMIN
				|| _op == OpOp1.CUMMAX  );
	}

	public boolean isCastUnaryOperation() 
	{
		return (   _op == OpOp1.CAST_AS_MATRIX
				|| _op == OpOp1.CAST_AS_SCALAR
				|| _op == OpOp1.CAST_AS_FRAME
				|| _op == OpOp1.CAST_AS_BOOLEAN
				|| _op == OpOp1.CAST_AS_DOUBLE
				|| _op == OpOp1.CAST_AS_INT    );
	}
	
	@Override
	protected ExecType optFindExecType() 
		throws HopsException 
	{		
		checkAndSetForcedPlatform();
	
		ExecType REMOTE = OptimizerUtils.isSparkExecutionMode() ? ExecType.SPARK : ExecType.MR;
		
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
				_etype = REMOTE;
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
		if( ConfigurationManager.isDynamicRecompilation() && !dimsKnown(true) && _etype==REMOTE )
			setRequiresRecompile();

		//ensure cp exec type for single-node operations
		if( _op == OpOp1.PRINT || _op == OpOp1.STOP 
			|| _op == OpOp1.INVERSE || _op == OpOp1.EIGEN || _op == OpOp1.CHOLESKY )
		{
			_etype = ExecType.CP;
		}
		
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		if ( getDataType() == DataType.SCALAR ) 
		{
			//do nothing always known
		}
		else if( _op == OpOp1.CAST_AS_MATRIX && getInput().get(0).getDataType()==DataType.SCALAR )
		{
			//prevent propagating 0 from scalar (which would be interpreted as unknown)
			setDim1( 1 );
			setDim2( 1 );
		}
		else //general case
		{
			// If output is a Matrix then this operation is of type (B = op(A))
			// Dimensions of B are same as that of A, and sparsity may/maynot change
			Hop input = getInput().get(0);
			setDim1( input.getDim1() );
			setDim2( input.getDim2() );
			if( _op==OpOp1.ABS || _op==OpOp1.COS || _op==OpOp1.SIN || _op==OpOp1.TAN  
				|| _op==OpOp1.ACOS || _op==OpOp1.ASIN || _op==OpOp1.ATAN
				|| _op==OpOp1.SQRT || _op==OpOp1.ROUND || _op==OpOp1.SPROP ) //sparsity preserving
			{
				setNnz( input.getNnz() );
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
