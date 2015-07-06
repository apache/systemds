/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import java.util.ArrayList;

import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.Aggregate.OperationTypes;
import com.ibm.bi.dml.lops.CombineUnary;
import com.ibm.bi.dml.lops.CumulativeOffsetBinary;
import com.ibm.bi.dml.lops.CumulativePartialAggregate;
import com.ibm.bi.dml.lops.CumulativeSplitAggregate;
import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.PartialAggregate;
import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.lops.PickByCount;
import com.ibm.bi.dml.lops.SortKeys;
import com.ibm.bi.dml.lops.Unary;
import com.ibm.bi.dml.lops.UnaryCP;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;


/* Unary (cell operations): e.g, b_ij = round(a_ij)
 * 		Semantic: given a value, perform the operation (independent of other values)
 */

public class UnaryOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private OpOp1 _op = null;

	
	private UnaryOp() {
		//default constructor for clone
	}
	
	public UnaryOp(String l, DataType dt, ValueType vt, OpOp1 o, Hop inp)
			throws HopsException 
	{
		super(l, dt, vt);

		getInput().add(0, inp);
		inp.getParent().add(this);

		_op = o;
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	// this is for OpOp1, e.g. A = -B (0-B); and a=!b
	public OpOp1 getOp() {
		return _op;
	}
	
	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (getVisited() != VisitStatus.DONE) {
				super.printMe();
				LOG.debug("  Operation: " + _op);
				for (Hop h : getInput()) {
					h.printMe();
				}
			}
			setVisited(VisitStatus.DONE);
		}
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += "u(" + _op.toString().toLowerCase() + ")";
		// s += HopsOpOp1String.get(_op) + ")";
		return s;
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
			
			if (getDataType() == DataType.SCALAR || _op == OpOp1.CAST_AS_MATRIX) 
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
					
					UnaryCP unary1 = new UnaryCP(input.constructLops(), optype, getDataType(), getValueType());
					setOutputDimensions(unary1);
					setLineNumbers(unary1);

					setLops(unary1);
				}
			} 
			else //general case MATRIX
			{
				ExecType et = optFindExecType();
				
				if( isCumulativeUnaryOperation() && et==ExecType.MR )  //special handling MR-cumsum/cumprod/cummin/cumsum
				{
					//TODO additional physical operation if offsets fit in memory
					Lop cumsumLop = constructLopsMRCumulativeUnary();
					setLops(cumsumLop);
				}
				else //default unary 
				{
					Unary unary1 = new Unary(input.constructLops(), HopsOpOp1LopsU.get(_op), 
							                 getDataType(), getValueType(), et);
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
			
			pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

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
			
			pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

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
			
			pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

			PartialAggregate pagg = new PartialAggregate(
					pick, HopsAgg2Lops.get(Hop.AggOp.SUM),
					HopsDirection2Lops.get(Hop.Direction.RowCol),
					DataType.MATRIX, getValueType());
			
			pagg.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

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
			group1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

			Aggregate agg1 = new Aggregate(
					group1, HopsAgg2Lops.get(Hop.AggOp.SUM),
					DataType.MATRIX, getValueType(), ExecType.MR);
			agg1.getOutputParameters().setDimensions(getDim1(),
					getDim2(), getRowsInBlock(),
					getColsInBlock(), getNnz());
			agg1.setupCorrectionLocation(pagg.getCorrectionLocation());
			
			agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			
			UnaryCP unary1 = new UnaryCP(
					agg1, HopsOpOp1LopsUS.get(OpOp1.CAST_AS_SCALAR),
					getDataType(), getValueType());
			unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			unary1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

			Unary iqm = new Unary(sort, unary1, Unary.OperationTypes.MR_IQM, DataType.SCALAR, ValueType.DOUBLE, ExecType.CP);
			iqm.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			iqm.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

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

			pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			
			return pick;
		}
	}
	
	/**
	 * MR Cumsum is currently based on a multipass algorithm of (1) preaggregation and (2) subsequent offsetting. 
	 * Note that we currently support one robust physical operator but many alternative
	 * realizations are possible for specific scenarios (e.g., when the preaggregated intermediate
	 * fit into the map task memory budget) or by creating custom job types.
	 * 
	 * 
	 * 
	 * @return
	 * @throws HopsException
	 * @throws LopsException
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
			Lop preagg = new CumulativePartialAggregate(TEMP, DataType.MATRIX, ValueType.DOUBLE, aggtype);
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
		if( TEMP.getOutputParameters().getNumRows()!=1 ){
			Unary unary1 = new Unary( TEMP, HopsOpOp1LopsU.get(_op), DataType.MATRIX, ValueType.DOUBLE, ExecType.CP);
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
			
			CumulativeOffsetBinary binary = new CumulativeOffsetBinary(group1, group2, DataType.MATRIX, ValueType.DOUBLE, aggtype);
			binary.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, -1);
			setLineNumbers(binary);
			TEMP = binary;
		}
		
		return TEMP;
	}
	
	/**
	 * 
	 * @return
	 */
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
	
	/**
	 * 
	 * @return
	 */
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
				|| _op==OpOp1.SQRT || _op==OpOp1.ROUND ) //sparsity preserving
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
	
	/**
	 * 
	 * @return
	 */
	private boolean isInMemoryOperation() 
	{
		return ( _op == OpOp1.INVERSE );
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isCumulativeUnaryOperation() 
	{
		return (   _op == OpOp1.CUMSUM 
				|| _op == OpOp1.CUMPROD
				|| _op == OpOp1.CUMMIN
				|| _op == OpOp1.CUMMAX  );
	}
	
	@Override
	protected ExecType optFindExecType() 
		throws HopsException 
	{		
		checkAndSetForcedPlatform();
	
		if( _etypeForced != null ) 			
		{
			_etype = _etypeForced;		
		}
		else 
		{
			ExecType REMOTE = OptimizerUtils.isSparkExecutionMode() ? ExecType.SPARK : ExecType.MR;
			
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
			
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==REMOTE )
				setRequiresRecompile();
		}
		
		if( _op == OpOp1.PRINT || _op == OpOp1.STOP || _op == OpOp1.INVERSE )
			_etype = ExecType.CP;
		
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
				|| _op==OpOp1.SQRT || _op==OpOp1.ROUND ) //sparsity preserving
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
		
		UnaryOp that2 = (UnaryOp)that;		
		return (   _op == that2._op
				&& getInput().get(0) == that2.getInput().get(0));
	}
}
