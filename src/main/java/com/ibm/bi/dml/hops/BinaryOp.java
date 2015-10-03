/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.AppendGAlignedSP;
import com.ibm.bi.dml.lops.AppendM;
import com.ibm.bi.dml.lops.AppendCP;
import com.ibm.bi.dml.lops.AppendG;
import com.ibm.bi.dml.lops.AppendR;
import com.ibm.bi.dml.lops.Binary;
import com.ibm.bi.dml.lops.BinaryScalar;
import com.ibm.bi.dml.lops.BinaryM;
import com.ibm.bi.dml.lops.BinaryUAggChain;
import com.ibm.bi.dml.lops.CentralMoment;
import com.ibm.bi.dml.lops.CoVariance;
import com.ibm.bi.dml.lops.CombineBinary;
import com.ibm.bi.dml.lops.CombineUnary;
import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.DataPartition;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.PartialAggregate;
import com.ibm.bi.dml.lops.PickByCount;
import com.ibm.bi.dml.lops.RepMat;
import com.ibm.bi.dml.lops.SortKeys;
import com.ibm.bi.dml.lops.Unary;
import com.ibm.bi.dml.lops.UnaryCP;
import com.ibm.bi.dml.lops.CombineBinary.OperationTypes;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.mapred.DistributedCacheInput;


/* Binary (cell operations): aij + bij
 * 		Properties: 
 * 			Symbol: *, -, +, ...
 * 			2 Operands
 * 		Semantic: align indices (sort), then perform operation
 */

public class BinaryOp extends Hop 
{
	
	//we use the full remote memory budget (but reduced by sort buffer), 
	public static final double APPEND_MEM_MULTIPLIER = 1.0;
	
	private Hop.OpOp2 op;
	private boolean outer = false;
	
	public static AppendMethod FORCED_APPEND_METHOD = null;
	public enum AppendMethod { 
		CP_APPEND, //in-memory general case append
		MR_MAPPEND, //map-only append (rhs must be vector and fit in mapper mem)
		MR_RAPPEND, //reduce-only append (output must have at most one column block)
		MR_GAPPEND, //map-reduce general case append (map-extend, aggregate)
		SP_GAlignedAppend // special case for general case in Spark where left.getCols() % left.getColsPerBlock() == 0
	};
	
	private enum MMBinaryMethod{
		CP_BINARY,
		MR_BINARY_R, //both mm, mv 
		MR_BINARY_M, //only mv (mr/spark)
		MR_BINARY_OUTER_M,
		MR_BINARY_OUTER_R, //only vv 
		MR_BINARY_UAGG_CHAIN, //(mr/spark)
	}
	
	private BinaryOp() {
		//default constructor for clone
	}
	
	public BinaryOp(String l, DataType dt, ValueType vt, Hop.OpOp2 o,
			Hop inp1, Hop inp2) {
		super(l, dt, vt);
		op = o;
		getInput().add(0, inp1);
		getInput().add(1, inp2);

		inp1.getParent().add(this);
		inp2.getParent().add(this);
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	public OpOp2 getOp() {
		return op;
	}
	
	public void setOp(OpOp2 iop) {
		 op = iop;
	}
	
	public void setOuterVectorOperation(boolean flag) {
		outer = flag;
	}
	
	public boolean isOuterVectorOperator(){
		return outer;
	}
	
	@Override
	public Lop constructLops() 
		throws HopsException, LopsException 
	{	
		//return already created lops
		if( getLops() != null )
			return getLops();

		//select the execution type
		ExecType et = optFindExecType();
		
		switch(op) 
		{
			case IQM: {
				constructLopsIQM(et);
				break;
			}
			case CENTRALMOMENT: {
				constructLopsCentralMoment(et);
				break;
			}	
			case COVARIANCE: {
				constructLopsCovariance(et);
				break;
			}
			case QUANTILE:
			case INTERQUANTILE: {
				constructLopsQuantile(et);
				break;
			}
			case MEDIAN: {
				constructLopsMedian(et);
				break;
			}
			case APPEND: {
				constructLopsAppend(et);
				break;
			}
			default:
				constructLopsBinaryDefault();	
		}

		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();
		
		return getLops();
	}
	
	private void constructLopsIQM(ExecType et) throws HopsException, LopsException {
		if ( et == ExecType.MR ) {
			CombineBinary combine = CombineBinary.constructCombineLop(
					OperationTypes.PreSort, (Lop) getInput().get(0)
							.constructLops(), (Lop) getInput().get(1)
							.constructLops(), DataType.MATRIX,
					getValueType());
			combine.getOutputParameters().setDimensions(
					getInput().get(0).getDim1(),
					getInput().get(0).getDim2(),
					getInput().get(0).getRowsInBlock(),
					getInput().get(0).getColsInBlock(), 
					getInput().get(0).getNnz());

			SortKeys sort = SortKeys.constructSortByValueLop(
					combine,
					SortKeys.OperationTypes.WithWeights,
					DataType.MATRIX, ValueType.DOUBLE, ExecType.MR);

			// Sort dimensions are same as the first input
			sort.getOutputParameters().setDimensions(
					getInput().get(0).getDim1(),
					getInput().get(0).getDim2(),
					getInput().get(0).getRowsInBlock(),
					getInput().get(0).getColsInBlock(), 
					getInput().get(0).getNnz());

			Data lit = Data.createLiteralLop(ValueType.DOUBLE, Double.toString(0.25));
			setLineNumbers(lit);
	
			PickByCount pick = new PickByCount(
					sort, lit, DataType.MATRIX, getValueType(),
					PickByCount.OperationTypes.RANGEPICK);

			pick.getOutputParameters().setDimensions(-1, -1, 
					getRowsInBlock(), getColsInBlock(), -1);
			setLineNumbers(pick);
			
			PartialAggregate pagg = new PartialAggregate(pick,
					HopsAgg2Lops.get(Hop.AggOp.SUM),
					HopsDirection2Lops.get(Hop.Direction.RowCol),
					DataType.MATRIX, getValueType());
			setLineNumbers(pagg);
			
			// Set the dimensions of PartialAggregate LOP based on the
			// direction in which aggregation is performed
			pagg.setDimensionsBasedOnDirection(getDim1(), getDim2(),
					getRowsInBlock(), getColsInBlock());

			Group group1 = new Group(pagg, Group.OperationTypes.Sort,
					DataType.MATRIX, getValueType());
			setOutputDimensions(group1);
			setLineNumbers(group1);

			Aggregate agg1 = new Aggregate(group1, HopsAgg2Lops
					.get(Hop.AggOp.SUM), DataType.MATRIX,
					getValueType(), ExecType.MR);
			setOutputDimensions(agg1);
			agg1.setupCorrectionLocation(pagg.getCorrectionLocation());
			setLineNumbers(agg1);

			UnaryCP unary1 = new UnaryCP(agg1, HopsOpOp1LopsUS
					.get(OpOp1.CAST_AS_SCALAR), DataType.SCALAR,
					getValueType());
			unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			setLineNumbers(unary1);
			
			Unary iqm = new Unary(sort, unary1, Unary.OperationTypes.MR_IQM, DataType.SCALAR, ValueType.DOUBLE, ExecType.CP);
			iqm.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			setLineNumbers(iqm);			
			setLops(iqm);
		}
		else 
		{
			SortKeys sort = SortKeys.constructSortByValueLop(
					getInput().get(0).constructLops(), 
					getInput().get(1).constructLops(), 
					SortKeys.OperationTypes.WithWeights, 
					getInput().get(0).getDataType(), getInput().get(0).getValueType(), et);
			sort.getOutputParameters().setDimensions(
					getInput().get(0).getDim1(),
					getInput().get(0).getDim2(), 
					getInput().get(0).getRowsInBlock(), 
					getInput().get(0).getColsInBlock(), 
					getInput().get(0).getNnz());
			PickByCount pick = new PickByCount(
					sort,
					null,
					getDataType(),
					getValueType(),
					PickByCount.OperationTypes.IQM, et, true);
			
			setOutputDimensions(pick);
			setLineNumbers(pick);
			setLops(pick);
		}
	}
	
	private void constructLopsMedian(ExecType et) throws HopsException, LopsException {
		if ( et == ExecType.MR ) {
			CombineBinary combine = CombineBinary
					.constructCombineLop(
							OperationTypes.PreSort,
							getInput().get(0).constructLops(),
							getInput().get(1).constructLops(),
							DataType.MATRIX, getValueType());

			SortKeys sort = SortKeys
					.constructSortByValueLop(
							combine,
							SortKeys.OperationTypes.WithWeights,
							DataType.MATRIX, getValueType(), et);

			combine.getOutputParameters().setDimensions(getDim1(),
					getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());

			// Sort dimensions are same as the first input
			sort.getOutputParameters().setDimensions(
					getInput().get(0).getDim1(),
					getInput().get(0).getDim2(),
					getInput().get(0).getRowsInBlock(),
					getInput().get(0).getColsInBlock(), 
					getInput().get(0).getNnz());

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

			setLops(pick);
		}
		else {
			SortKeys sort = SortKeys.constructSortByValueLop(
					getInput().get(0).constructLops(), 
					getInput().get(1).constructLops(), 
					SortKeys.OperationTypes.WithWeights, 
					getInput().get(0).getDataType(), getInput().get(0).getValueType(), et);
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
		}
	}
	
	private void constructLopsCentralMoment(ExecType et) 
		throws HopsException, LopsException 
	{
		// The output data type is a SCALAR if central moment 
		// gets computed in CP/SPARK, and it will be MATRIX otherwise.
		DataType dt = (et == ExecType.MR ? DataType.MATRIX : DataType.SCALAR );
		CentralMoment cm = new CentralMoment(
				getInput().get(0).constructLops(), 
				getInput().get(1).constructLops(),
				dt, getValueType(), et);

		setLineNumbers(cm);
		
		if ( et == ExecType.MR ) {
			cm.getOutputParameters().setDimensions(1, 1, 0, 0, -1);
			UnaryCP unary1 = new UnaryCP(cm, HopsOpOp1LopsUS
					.get(OpOp1.CAST_AS_SCALAR), getDataType(),
					getValueType());
			unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			setLineNumbers(unary1);
			setLops(unary1);
		}
		else {
			cm.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			setLops(cm);
		}
	}
	
	/**
	 * 
	 * @param et
	 * @throws LopsException
	 * @throws HopsException
	 */
	private void constructLopsCovariance(ExecType et) 
		throws LopsException, HopsException 
	{
		if ( et == ExecType.MR ) {
			// combineBinary -> CoVariance -> CastAsScalar
			CombineBinary combine = CombineBinary.constructCombineLop(
					OperationTypes.PreCovUnweighted, getInput().get(
							0).constructLops(), getInput().get(1)
							.constructLops(), DataType.MATRIX,
					getValueType());

			combine.getOutputParameters().setDimensions(
					getInput().get(0).getDim1(),
					getInput().get(0).getDim2(),
					getInput().get(0).getRowsInBlock(),
					getInput().get(0).getColsInBlock(), 
					getInput().get(0).getNnz());

			CoVariance cov = new CoVariance(combine, DataType.MATRIX,
					getValueType(), et);
			cov.getOutputParameters().setDimensions(1, 1, 0, 0, -1);
			setLineNumbers(cov);

			UnaryCP unary1 = new UnaryCP(cov, HopsOpOp1LopsUS
					.get(OpOp1.CAST_AS_SCALAR), getDataType(),
					getValueType());
			unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			setLineNumbers(unary1);	
			setLops(unary1);
		}
		else //CP/SPARK
		{
			CoVariance cov = new CoVariance(
					getInput().get(0).constructLops(), 
					getInput().get(1).constructLops(), 
					getDataType(), getValueType(), et);
			cov.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			setLineNumbers(cov);
			setLops(cov);
		}
	}
	
	private void constructLopsQuantile(ExecType et) throws HopsException, LopsException {
		// 1st arguments needs to be a 1-dimensional matrix
		// For QUANTILE: 2nd argument is scalar or 1-dimensional matrix
		// For INTERQUANTILE: 2nd argument is always a scalar

		PickByCount.OperationTypes pick_op = null;
		if(op == Hop.OpOp2.QUANTILE)
			pick_op = PickByCount.OperationTypes.VALUEPICK;
		else
			pick_op = PickByCount.OperationTypes.RANGEPICK;

		if ( et == ExecType.MR ) 
		{
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
			ExecType et_pick = (getInput().get(1).getDataType() == DataType.SCALAR ? ExecType.CP : ExecType.MR);
			
			PickByCount pick = new PickByCount(
					sort,
					getInput().get(1).constructLops(),
					getDataType(),
					getValueType(),
					pick_op, et_pick, false);

			pick.getOutputParameters().setDimensions(getDim1(),
					getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
			
			pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

			setLops(pick);
		}
		else //CP/SPARK 
		{
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
			PickByCount pick = new PickByCount( sort, getInput().get(1).constructLops(),
					getDataType(), getValueType(), pick_op, et, true);

			setOutputDimensions(pick);
			setLineNumbers(pick);
			setLops(pick);
		}
	}
	
	/**
	 * 
	 * @param et
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructLopsAppend(ExecType et) 
		throws HopsException, LopsException 
	{
		DataType dt1 = getInput().get(0).getDataType();
		DataType dt2 = getInput().get(1).getDataType();
		ValueType vt1 = getInput().get(0).getValueType();
		ValueType vt2 = getInput().get(1).getValueType();
		
		//sanity check for input data types
		if( !((dt1==DataType.MATRIX && dt2==DataType.MATRIX)
			 ||(dt1==DataType.SCALAR && dt2==DataType.SCALAR
			   && vt1==ValueType.STRING && vt2==ValueType.STRING )) )
		{
			throw new HopsException("Append can only apply to two matrices or two scalar strings!");
		}
				
		Lop append = null;
		if( dt1==DataType.MATRIX && dt2==DataType.MATRIX )
		{
			if( et == ExecType.MR )
			{
				append = constructMRAppendLop(getInput().get(0), getInput().get(1), getDataType(), getValueType(), this);				
			}
			else if(et == ExecType.SPARK) 
			{
				long ncol = (getInput().get(0).dimsKnown() && getInput().get(1).dimsKnown()) ? 
						   getInput().get(0).getDim2()+getInput().get(1).getDim2() : -1; 
				append = constructSPAppendLop(getInput().get(0), getInput().get(1), getDataType(), getValueType(), this);
				append.getOutputParameters().setDimensions(getInput().get(0).getDim1(), ncol, getRowsInBlock(), getColsInBlock(), getNnz());
			}
			else //CP
			{
				Lop offset = createOffsetLop( getInput().get(0), true ); //offset 1st input
				append = new AppendCP(getInput().get(0).constructLops(), getInput().get(1).constructLops(), offset, getDataType(), getValueType());
				append.getOutputParameters().setDimensions(getInput().get(0).getDim1(), getInput().get(0).getDim2()+getInput().get(1).getDim2(), 
							                                getRowsInBlock(), getColsInBlock(), getNnz());
			}
		}
		else //SCALAR-STRING and SCALAR-STRING (always CP)
		{
			append = new AppendCP(getInput().get(0).constructLops(), getInput().get(1).constructLops(), 
				     Data.createLiteralLop(ValueType.INT, "-1"), getDataType(), getValueType());
			append.getOutputParameters().setDimensions(0,0,-1,-1,-1);
		}
		
		setLineNumbers(append);
		setLops(append);
	}
	
	/**
	 * 
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructLopsBinaryDefault() 
		throws HopsException, LopsException 
	{
		/* Default behavior for BinaryOp */
		// it depends on input data types
		DataType dt1 = getInput().get(0).getDataType();
		DataType dt2 = getInput().get(1).getDataType();
		
		if (dt1 == dt2 && dt1 == DataType.SCALAR) {

			// Both operands scalar
			BinaryScalar binScalar1 = new BinaryScalar(getInput().get(0)
					.constructLops(),
					getInput().get(1).constructLops(), HopsOpOp2LopsBS
							.get(op), getDataType(), getValueType());
			binScalar1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			setLineNumbers(binScalar1);
			setLops(binScalar1);

		} 
		else if ((dt1 == DataType.MATRIX && dt2 == DataType.SCALAR)
				   || (dt1 == DataType.SCALAR && dt2 == DataType.MATRIX)) {

			// One operand is Matrix and the other is scalar
			ExecType et = optFindExecType();
			
			//select specific operator implementations
			Unary.OperationTypes ot = null;
			Hop right = getInput().get(1);
			if( op==OpOp2.POW && right instanceof LiteralOp && ((LiteralOp)right).getDoubleValue()==2.0  )
				ot = Unary.OperationTypes.POW2;
			else if( op==OpOp2.MULT && right instanceof LiteralOp && ((LiteralOp)right).getDoubleValue()==2.0  )
				ot = Unary.OperationTypes.MULTIPLY2;
			else //general case
				ot = HopsOpOp2LopsU.get(op);
			
			
			Unary unary1 = new Unary(getInput().get(0).constructLops(),
						   getInput().get(1).constructLops(), ot, getDataType(), getValueType(), et);
		
			setOutputDimensions(unary1);
			setLineNumbers(unary1);
			setLops(unary1);
			
		} 
		else 
		{
			// Both operands are Matrixes
			ExecType et = optFindExecType();
			if ( et == ExecType.CP ) 
			{
				Binary binary = new Binary(getInput().get(0).constructLops(), getInput().get(1).constructLops(), HopsOpOp2LopsB.get(op),
						getDataType(), getValueType(), et);
				
				setOutputDimensions(binary);
				setLineNumbers(binary);
				setLops(binary);
			}
			else if(et == ExecType.SPARK)
			{
				Hop left = getInput().get(0);
				Hop right = getInput().get(1);
				//TODO need to create spark-specific op selection for supporting binarym/binaryr
				MMBinaryMethod mbin = optFindMMBinaryMethodSpark(left, right);
				
				Lop  binary = null;
				if( mbin == MMBinaryMethod.MR_BINARY_UAGG_CHAIN ) {
					AggUnaryOp uRight = (AggUnaryOp)right;
					binary = new BinaryUAggChain(left.constructLops(), HopsOpOp2LopsB.get(op),
							HopsAgg2Lops.get(uRight.getOp()), HopsDirection2Lops.get(uRight.getDirection()),
							getDataType(), getValueType(), et);
				}
				else if (mbin == MMBinaryMethod.MR_BINARY_M) {
					boolean partitioned = false;
					boolean isColVector = (right.getDim2()==1 && left.getDim1()==right.getDim1());
					
					binary = new BinaryM(left.constructLops(), right.constructLops(),
							HopsOpOp2LopsB.get(op), getDataType(), getValueType(), et, partitioned, isColVector); 
				}
				else {
					binary = new Binary(left.constructLops(), right.constructLops(), 
							HopsOpOp2LopsB.get(op), getDataType(), getValueType(), et);
				}
				
				setOutputDimensions(binary);
				setLineNumbers(binary);
				setLops(binary);
			}
			else //MR
			{
				Hop left = getInput().get(0);
				Hop right = getInput().get(1);
				MMBinaryMethod mbin = optFindMMBinaryMethod(left, right);
				
				if( mbin == MMBinaryMethod.MR_BINARY_M )
				{
					boolean needPart = requiresPartitioning(right);
					Lop dcInput = right.constructLops();
					if( needPart ) {
						//right side in distributed cache
						ExecType etPart = (OptimizerUtils.estimateSizeExactSparsity(right.getDim1(), right.getDim2(), OptimizerUtils.getSparsity(right.getDim1(), right.getDim2(), right.getNnz())) 
						          < OptimizerUtils.getLocalMemBudget()) ? ExecType.CP : ExecType.MR; //operator selection
						dcInput = new DataPartition(dcInput, DataType.MATRIX, ValueType.DOUBLE, etPart, (right.getDim2()==1)?PDataPartitionFormat.ROW_BLOCK_WISE_N:PDataPartitionFormat.COLUMN_BLOCK_WISE_N);
						dcInput.getOutputParameters().setDimensions(right.getDim1(), right.getDim2(), right.getRowsInBlock(), right.getColsInBlock(), right.getNnz());
						dcInput.setAllPositions(right.getBeginLine(), right.getBeginColumn(), right.getEndLine(), right.getEndColumn());
					}					
					
					BinaryM binary = new BinaryM(left.constructLops(), dcInput, HopsOpOp2LopsB.get(op),
							getDataType(), getValueType(), ExecType.MR, needPart, (right.getDim2()==1 && left.getDim1()==right.getDim1()));
					setOutputDimensions(binary);
					setLineNumbers(binary);
					setLops(binary);
				}
				else if( mbin == MMBinaryMethod.MR_BINARY_UAGG_CHAIN )
				{
					AggUnaryOp uRight = (AggUnaryOp)right;
					BinaryUAggChain bin = new BinaryUAggChain(left.constructLops(), HopsOpOp2LopsB.get(op),
							HopsAgg2Lops.get(uRight.getOp()), HopsDirection2Lops.get(uRight.getDirection()),
							getDataType(), getValueType(), et);
					setOutputDimensions(bin);
					setLineNumbers(bin);
					setLops(bin);
				}
				else if( mbin == MMBinaryMethod.MR_BINARY_OUTER_R )
				{
					boolean requiresRepLeft = (!right.dimsKnown() || right.getDim2() > right.getColsInBlock());
					boolean requiresRepRight = (!left.dimsKnown() || left.getDim1() > right.getRowsInBlock());
					
					Lop leftLop = left.constructLops();
					Lop rightLop = right.constructLops();
					
					if( requiresRepLeft ) {
						Lop offset = createOffsetLop(right, true); //ncol of right determines rep of left
						leftLop = new RepMat(leftLop, offset, true, left.getDataType(), left.getValueType());
						setOutputDimensions(leftLop);
						setLineNumbers(leftLop);
					}
					
					if( requiresRepRight ) {
						Lop offset = createOffsetLop(left, false); //nrow of right determines rep of right
						rightLop = new RepMat(rightLop, offset, false, right.getDataType(), right.getValueType());
						setOutputDimensions(rightLop);
						setLineNumbers(rightLop);
					}
				
					Group group1 = new Group( leftLop, Group.OperationTypes.Sort, getDataType(), getValueType());
					setLineNumbers(group1);
					setOutputDimensions(group1);
					
					Group group2 = new Group( rightLop, Group.OperationTypes.Sort, getDataType(), getValueType());
					setLineNumbers(group2);
					setOutputDimensions(group2);
					
					Binary binary = new Binary(group1, group2, HopsOpOp2LopsB.get(op), getDataType(), getValueType(), et);
					setOutputDimensions(binary);
					setLineNumbers(binary);
					
					setLops(binary);
				}
				else //MMBinaryMethod.MR_BINARY_R
				{
					boolean requiresRep = requiresReplication(left, right);
					
					Lop rightLop = right.constructLops();
					if( requiresRep ) {
						Lop offset = createOffsetLop(left, (right.getDim2()<=1)); //ncol of left input (determines num replicates)
						rightLop = new RepMat(rightLop, offset, (right.getDim2()<=1), right.getDataType(), right.getValueType());
						setOutputDimensions(rightLop);
						setLineNumbers(rightLop);	
					}
				
					Group group1 = new Group(getInput().get(0).constructLops(), Group.OperationTypes.Sort, getDataType(), getValueType());
					setLineNumbers(group1);
					setOutputDimensions(group1);
				
					Group group2 = new Group( rightLop, Group.OperationTypes.Sort, getDataType(), getValueType());
					setLineNumbers(group2);
					setOutputDimensions(group2);
				
					Binary binary = new Binary(group1, group2, HopsOpOp2LopsB.get(op), getDataType(), getValueType(), et);
					setLineNumbers(binary);
					setOutputDimensions(binary);
					
					setLops(binary);
				}
			}
		}
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += "b(" + HopsOpOp2String.get(op) + ")";
		return s;
	}

	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (getVisited() != VisitStatus.DONE) {
				super.printMe();
				LOG.debug("  Operation: " + op );
				for (Hop h : getInput()) {
					h.printMe();
				}
				;
			}
			setVisited(VisitStatus.DONE);
		}
	}

	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		double ret = 0;
		
		//preprocessing step (recognize unknowns)
		if( dimsKnown() && _nnz<0 ) //never after inference
			nnz = -1; 
		
		if(op==OpOp2.APPEND && !OptimizerUtils.ALLOW_DYN_RECOMPILATION && !(getDataType()==DataType.SCALAR) ) {	
			ret = OptimizerUtils.DEFAULT_SIZE;
		}
		else
		{
			double sparsity = 1.0;
			if( nnz < 0 ){ //check for exactly known nnz
				Hop input1 = getInput().get(0);
				Hop input2 = getInput().get(1);
				if( input1.dimsKnown() && input2.dimsKnown() )
				{
					if( OptimizerUtils.isBinaryOpConditionalSparseSafe(op) && input2 instanceof LiteralOp ) {
						double sp1 = (input1.getNnz()>0 && input1.getDataType()==DataType.MATRIX) ? OptimizerUtils.getSparsity(input1.getDim1(), input1.getDim2(), input1.getNnz()) : 1.0;
						LiteralOp lit = (LiteralOp)input2;
						sparsity = OptimizerUtils.getBinaryOpSparsityConditionalSparseSafe(sp1, op, lit);
					}
					else {
						double sp1 = (input1.getNnz()>0 && input1.getDataType()==DataType.MATRIX) ? OptimizerUtils.getSparsity(input1.getDim1(), input1.getDim2(), input1.getNnz()) : 1.0;
						double sp2 = (input2.getNnz()>0 && input2.getDataType()==DataType.MATRIX) ? OptimizerUtils.getSparsity(input2.getDim1(), input2.getDim2(), input2.getNnz()) : 1.0;
						//sparsity estimates are conservative in terms of the worstcase behavior, however,
						//for outer vector operations the average case is equivalent to the worst case.
						sparsity = OptimizerUtils.getBinaryOpSparsity(sp1, sp2, op, !outer);
					}
				}
			}
			else //e.g., for append,pow or after inference
				sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
			
			ret = OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);	
		}
		
		
		return ret;
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		double ret = 0;
		if ( op == OpOp2.QUANTILE || op == OpOp2.IQM  || op == OpOp2.MEDIAN ) {
			// buffer (=2*input_size) and output (=input_size) for SORT operation 
			// getMemEstimate works for both cases of known dims and worst-case
			ret = getInput().get(0).getMemEstimate() * 3; 
		}
		else if ( op == OpOp2.SOLVE ) {
			// x=solve(A,b) relies on QR decomposition of A, which is done using Apache commons-math
			// matrix of size same as the first input
			double interOutput = OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).getDim1(), getInput().get(0).getDim2(), 1.0); 
			return interOutput;

		}

		return ret;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		long[] ret = null;
		
		MatrixCharacteristics[] mc = memo.getAllInputStats(getInput());
		Hop input1 = getInput().get(0);
		Hop input2 = getInput().get(1);		
		DataType dt1 = input1.getDataType();
		DataType dt2 = input2.getDataType();
		
		if( op== OpOp2.APPEND )
		{
			long ldim1 = -1, ldim2 = -1, lnnz = -1;
			
			if( mc[0].rowsKnown() || mc[1].rowsKnown() )
				ldim1 = mc[0].rowsKnown() ? mc[0].getRows() : mc[1].getRows();
			if( mc[0].colsKnown() && mc[1].colsKnown() )
				ldim2 = mc[0].getCols()+mc[1].getCols();
			if( mc[0].nnzKnown() && mc[1].nnzKnown() )
				lnnz = mc[0].getNonZeros() + mc[1].getNonZeros();
			
			if( ldim1 > 0 || ldim2 > 0 || lnnz >= 0 )
				return new long[]{ldim1, ldim2, lnnz};
		}
		else if ( op == OpOp2.SOLVE ) {
			// Output is a (likely to be dense) vector of size number of columns in the first input
			if ( mc[0].getCols() > 0 ) {
				ret = new long[]{ mc[0].getCols(), 1, mc[0].getCols()};
			}
		}
		else //general case
		{
			long ldim1, ldim2;
			double sp1 = 1.0, sp2 = 1.0;
			
			if( dt1 == DataType.MATRIX && dt2 == DataType.SCALAR && mc[0].dimsKnown() )
			{
				ldim1 = mc[0].getRows();
				ldim2 = mc[0].getCols();
				sp1 = (mc[0].getNonZeros()>0)?OptimizerUtils.getSparsity(ldim1, ldim2, mc[0].getNonZeros()):1.0;	
			}
			else if( dt1 == DataType.SCALAR && dt2 == DataType.MATRIX  ) 
			{
				ldim1 = mc[1].getRows();
				ldim2 = mc[1].getCols();
				sp2 = (mc[1].getNonZeros()>0)?OptimizerUtils.getSparsity(ldim1, ldim2, mc[1].getNonZeros()):1.0;
			}
			else //MATRIX - MATRIX 
			{
				//propagate if either input is known, rows need always be identical,
				//for cols we need to be careful with regard to matrix-vector operations
				if( outer ) //OUTER VECTOR OPERATION
				{
					ldim1 = mc[0].getRows();
					ldim2 = mc[1].getCols();
				}
				else //GENERAL CASE
				{
					ldim1 = (mc[0].getRows()>0) ? mc[0].getRows() : 
					        (mc[1].getRows()>1) ? mc[1].getRows() : -1;
					ldim2 = (mc[0].getCols()>0) ? mc[0].getCols() : 
						    (mc[1].getCols()>1) ? mc[1].getCols() : -1;
				}
				sp1 = (mc[0].getNonZeros()>0)?OptimizerUtils.getSparsity(ldim1, ldim2, mc[0].getNonZeros()):1.0;
				sp2 = (mc[1].getNonZeros()>0)?OptimizerUtils.getSparsity(ldim1, ldim2, mc[1].getNonZeros()):1.0;
			}
			
			if( ldim1>0 && ldim2>0 )
			{
				if( OptimizerUtils.isBinaryOpConditionalSparseSafe(op) && input2 instanceof LiteralOp ) {
					long lnnz = (long) (ldim1*ldim2*OptimizerUtils.getBinaryOpSparsityConditionalSparseSafe(sp1, op,(LiteralOp)input2));
					ret = new long[]{ldim1, ldim2, lnnz};	
				}
				else
				{
					//sparsity estimates are conservative in terms of the worstcase behavior, however,
					//for outer vector operations the average case is equivalent to the worst case.
					long lnnz = (long) (ldim1*ldim2*OptimizerUtils.getBinaryOpSparsity(sp1, sp2, op, !outer));
					ret = new long[]{ldim1, ldim2, lnnz};
				}
			}
		}

		return ret;
	}

	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}
	
	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();
		
		ExecType REMOTE = OptimizerUtils.isSparkExecutionMode() ? ExecType.SPARK : ExecType.MR;
		DataType dt1 = getInput().get(0).getDataType();
		DataType dt2 = getInput().get(1).getDataType();
		
		if( _etypeForced != null ) {		
			_etype = _etypeForced;
		}
		else 
		{
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) 
			{
				_etype = findExecTypeByMemEstimate();
			}
			else
			{
				_etype = null;
				if ( dt1 == DataType.MATRIX && dt2 == DataType.MATRIX ) {
					// choose CP if the dimensions of both inputs are below Hops.CPThreshold 
					// OR if both are vectors
					if ( (getInput().get(0).areDimsBelowThreshold() && getInput().get(1).areDimsBelowThreshold())
							|| (getInput().get(0).isVector() && getInput().get(1).isVector()))
					{
						_etype = ExecType.CP;
					}
				}
				else if ( dt1 == DataType.MATRIX && dt2 == DataType.SCALAR ) {
					if ( getInput().get(0).areDimsBelowThreshold() || getInput().get(0).isVector() )
					{
						_etype = ExecType.CP;
					}
				}
				else if ( dt1 == DataType.SCALAR && dt2 == DataType.MATRIX ) {
					if ( getInput().get(1).areDimsBelowThreshold() || getInput().get(1).isVector() )
					{
						_etype = ExecType.CP;
					}
				}
				else
				{
					_etype = ExecType.CP;
				}
				
				//if no CP condition applied
				if( _etype == null )
					_etype = REMOTE;
			}
		
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}
			
		//spark-specific decision refinement (execute unary scalar w/ spark input and 
		//single parent also in spark because it's likely cheap and reduces intermediates)
		if( _etype == ExecType.CP && _etypeForced != ExecType.CP
			&& getDataType().isMatrix() && (dt1.isScalar() || dt2.isScalar()) 
			&& getInput().get(dt1.isScalar()?1:0).optFindExecType() == ExecType.SPARK 
			&& supportsMatrixScalarOperations()                           //scalar operations
			&& !(getInput().get(dt1.isScalar()?1:0) instanceof DataOp)    //input is not checkpoint
			&& getInput().get(dt1.isScalar()?1:0).getParent().size()==1 ) //unary scalar is only parent
		{
			//pull unary scalar operation into spark 
			_etype = ExecType.SPARK;
		}
		
		//mark for recompile (forever)
		if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && ((!dimsKnown(true)&&_etype==REMOTE) 
			|| (op == OpOp2.APPEND && getDataType()!=DataType.SCALAR) ) )
		{
			setRequiresRecompile();
		}
		
		//ensure cp exec type for single-node operations
		if ( op == OpOp2.SOLVE ) {
			_etype = ExecType.CP;
		}
		
		return _etype;
	}
	
	/**
	 * General case binary append.
	 * 
	 * @param left
	 * @param right
	 * @return
	 * @throws HopsException 
	 * @throws LopsException 
	 */
	public static Lop constructMRAppendLop( Hop left, Hop right, DataType dt, ValueType vt, Hop current ) 
		throws HopsException, LopsException
	{
		Lop ret = null;
		
		long m1_dim1 = left.getDim1();
		long m1_dim2 = left.getDim2();		
		long m2_dim1 = right.getDim1();
		long m2_dim2 = right.getDim2();
		long m3_dim2 = (m1_dim2>0 && m2_dim2>0) ? (m1_dim2 + m2_dim2) : -1; //output cols
		long m3_nnz = (left.getNnz()>0 && right.getNnz()>0) ? (left.getNnz() + right.getNnz()) : -1; //output nnz
		long brlen = left.getRowsInBlock();
		long bclen = left.getColsInBlock();
		
		Lop offset = createOffsetLop( left, true ); //offset 1st input
		AppendMethod am = optFindAppendMethod(m1_dim1, m1_dim2, m2_dim1, m2_dim2, brlen, bclen);
	
		switch( am )
		{
			case MR_MAPPEND: //special case map-only append
			{
				boolean needPart = requiresPartitioning(right);
				//pre partitioning 
				Lop dcInput = right.constructLops();
				if( needPart ) {
					//right side in distributed cache
					ExecType etPart = (OptimizerUtils.estimateSizeExactSparsity(right.getDim1(), right.getDim2(), OptimizerUtils.getSparsity(right.getDim1(), right.getDim2(), right.getNnz())) 
					          < OptimizerUtils.getLocalMemBudget()) ? ExecType.CP : ExecType.MR; //operator selection
					dcInput = new DataPartition(dcInput, DataType.MATRIX, ValueType.DOUBLE, etPart, PDataPartitionFormat.ROW_BLOCK_WISE_N);
					dcInput.getOutputParameters().setDimensions(right.getDim1(), right.getDim2(), right.getRowsInBlock(), right.getColsInBlock(), right.getNnz());
					dcInput.setAllPositions(right.getBeginLine(), right.getBeginColumn(), right.getEndLine(), right.getEndColumn());
				}					
				
				AppendM appM = new AppendM(left.constructLops(), dcInput, offset, dt, vt, needPart, ExecType.MR);
				appM.setAllPositions(current.getBeginLine(), current.getBeginColumn(), current.getEndLine(), current.getEndColumn());
				appM.getOutputParameters().setDimensions(m1_dim1, m3_dim2, brlen, bclen, m3_nnz);
				ret = appM;
				break;
			}
			case MR_RAPPEND: //special case reduce append w/ one column block
			{
				//group
				Group group1 = new Group(left.constructLops(), Group.OperationTypes.Sort, DataType.MATRIX, vt);
				group1.getOutputParameters().setDimensions(m1_dim1, m1_dim2, brlen, bclen, left.getNnz());
				group1.setAllPositions(left.getBeginLine(), left.getBeginColumn(), left.getEndLine(), left.getEndColumn());
				
				Group group2 = new Group(right.constructLops(), Group.OperationTypes.Sort, DataType.MATRIX, vt);
				group1.getOutputParameters().setDimensions(m2_dim1, m2_dim2, brlen, bclen, right.getNnz());
				group1.setAllPositions(right.getBeginLine(), right.getBeginColumn(), right.getEndLine(), right.getEndColumn());
				
				AppendR appR = new AppendR(group1, group2, dt, vt, ExecType.MR);
				appR.getOutputParameters().setDimensions(m1_dim1, m3_dim2, brlen, bclen, m3_nnz);
				appR.setAllPositions(current.getBeginLine(), current.getBeginColumn(), current.getEndLine(), current.getEndColumn());
				
				ret = appR;
				break;
			}	
			case MR_GAPPEND:
			{
				//general case: map expand append, reduce aggregate
				Lop offset2 = createOffsetLop( right, true ); //offset second input
				
				AppendG appG = new AppendG(left.constructLops(), right.constructLops(),	offset, offset2, dt, vt, ExecType.MR);
				appG.getOutputParameters().setDimensions(m1_dim1, m3_dim2, brlen, bclen, m3_nnz);
				appG.setAllPositions(current.getBeginLine(), current.getBeginColumn(), current.getEndLine(), current.getEndColumn());
				
				//group
				Group group1 = new Group(appG, Group.OperationTypes.Sort, DataType.MATRIX, vt);
				group1.getOutputParameters().setDimensions(m1_dim1, m3_dim2, brlen, bclen, m3_nnz);
				group1.setAllPositions(current.getBeginLine(), current.getBeginColumn(), current.getEndLine(), current.getEndColumn());
				
				//aggregate
				Aggregate agg1 = new Aggregate(group1, Aggregate.OperationTypes.Sum, DataType.MATRIX, vt, ExecType.MR);
				agg1.getOutputParameters().setDimensions(m1_dim1, m3_dim2, brlen, bclen, m3_nnz);
				agg1.setAllPositions(current.getBeginLine(), current.getBeginColumn(), current.getEndLine(), current.getEndColumn());
				ret = agg1;
				break;
			}	
			default:
				throw new HopsException("Invalid MR append method: "+am);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param left
	 * @param right
	 * @param dt
	 * @param vt
	 * @param current
	 * @return
	 * @throws HopsException
	 * @throws LopsException
	 */
	public Lop constructSPAppendLop( Hop left, Hop right, DataType dt, ValueType vt, Hop current ) 
		throws HopsException, LopsException
	{
		Lop ret = null;
		
		Lop offset = createOffsetLop( left, true ); //offset 1st input
		AppendMethod am = optFindAppendSPMethod(left.getDim1(), left.getDim2(), right.getDim1(), right.getDim2(), 
				right.getRowsInBlock(), right.getColsInBlock(), right.getNnz());
	
		switch( am )
		{
			case MR_MAPPEND: //special case map-only append
			{
				ret = new AppendM(left.constructLops(), right.constructLops(), offset, getDataType(), getValueType(), false, ExecType.SPARK);
				break;
			}
			case MR_RAPPEND: //special case reduce append w/ one column block
			{
				ret = new AppendR(left.constructLops(), right.constructLops(), getDataType(), getValueType(), ExecType.SPARK);
				break;
			}	
			case MR_GAPPEND:
			{
				Lop offset2 = createOffsetLop( right, true ); //offset second input
				ret = new AppendG(left.constructLops(), right.constructLops(), offset, offset2, getDataType(), getValueType(), ExecType.SPARK);
				break;
			}
			case SP_GAlignedAppend:
			{
				ret = new AppendGAlignedSP(left.constructLops(), right.constructLops(), offset, getDataType(), getValueType());
				break;
			}
			default:
				throw new HopsException("Invalid SP append method: "+am);
		}
		
		ret.setAllPositions(current.getBeginLine(), current.getBeginColumn(), current.getEndLine(), current.getEndColumn());
		
		
		return ret;
	}
	
	/**
	 * Special case tertiary append. Here, we also compile a MR_RAPPEND or MR_GAPPEND
	 * 
	 * @param left
	 * @param right
	 * @param dt
	 * @param vt
	 * @param current
	 * @return
	 * @throws HopsException
	 * @throws LopsException
	 */
	public static Lop constructAppendLopChain( Hop left, Hop right1, Hop right2, DataType dt, ValueType vt, Hop current ) 
		throws HopsException, LopsException
	{
		long m1_dim1 = left.getDim1();
		long m1_dim2 = left.getDim2();		
		long m2_dim1 = right1.getDim1();
		long m2_dim2 = right1.getDim2();
		long m3_dim1 = right2.getDim1();
		long m3_dim2 = right2.getDim2();		
		long m41_dim2 = (m1_dim2>0 && m2_dim2>0) ? (m1_dim2 + m2_dim2) : -1; //output cols
		long m41_nnz = (left.getNnz()>0 && right1.getNnz()>0) ? 
				      (left.getNnz() + right1.getNnz()) : -1; //output nnz
		long m42_dim2 = (m1_dim2>0 && m2_dim2>0 && m3_dim2>0) ? (m1_dim2 + m2_dim2 + m3_dim2) : -1; //output cols
		long m42_nnz = (left.getNnz()>0 && right1.getNnz()>0 && right2.getNnz()>0) ? 
				      (left.getNnz() + right1.getNnz()+ right2.getNnz()) : -1; //output nnz
		long brlen = left.getRowsInBlock();
		long bclen = left.getColsInBlock();
		
		//warn if assumption of blocksize>=3 does not hold
		if( bclen < 3 )
			throw new HopsException("MR_RAPPEND requires a blocksize of >= 3.");
		
		//case MR_RAPPEND:
		//special case reduce append w/ one column block
		
		Group group1 = new Group(left.constructLops(), Group.OperationTypes.Sort, DataType.MATRIX, vt);
		group1.getOutputParameters().setDimensions(m1_dim1, m1_dim2, brlen, bclen, left.getNnz());
		group1.setAllPositions(left.getBeginLine(), left.getBeginColumn(), left.getEndLine(), left.getEndColumn());
		
		Group group2 = new Group(right1.constructLops(), Group.OperationTypes.Sort, DataType.MATRIX, vt);
		group1.getOutputParameters().setDimensions(m2_dim1, m2_dim2, brlen, bclen, right1.getNnz());
		group1.setAllPositions(right1.getBeginLine(), right1.getBeginColumn(), right1.getEndLine(), right1.getEndColumn());
		
		Group group3 = new Group(right2.constructLops(), Group.OperationTypes.Sort, DataType.MATRIX, vt);
		group1.getOutputParameters().setDimensions(m3_dim1, m3_dim2, brlen, bclen, right2.getNnz());
		group1.setAllPositions(right2.getBeginLine(), right2.getBeginColumn(), right2.getEndLine(), right2.getEndColumn());
		
		AppendR appR1 = new AppendR(group1, group2, dt, vt, ExecType.MR);
		appR1.getOutputParameters().setDimensions(m1_dim1, m41_dim2, brlen, bclen, m41_nnz);
		appR1.setAllPositions(current.getBeginLine(), current.getBeginColumn(), current.getEndLine(), current.getEndColumn());
		
		AppendR appR2 = new AppendR(appR1, group3, dt, vt, ExecType.MR);
		appR1.getOutputParameters().setDimensions(m1_dim1, m42_dim2, brlen, bclen, m42_nnz);
		appR1.setAllPositions(current.getBeginLine(), current.getBeginColumn(), current.getEndLine(), current.getEndColumn());
	
		return appR2;
	}
	
	/**
	 * Estimates the memory footprint of MapMult operation depending on which input is put into distributed cache.
	 * This function is called by <code>optFindAppendMethod()</code> to decide the execution strategy, as well as by 
	 * piggybacking to decide the number of Map-side instructions to put into a single GMR job. 
	 */
	public static double footprintInMapper( long m1_dim1, long m1_dim2, long m2_dim1, long m2_dim2, long m1_rpb, long m1_cpb ) {
		double footprint = 0;
		
		// size of left input (matrix block)
		footprint += OptimizerUtils.estimateSize(Math.min(m1_dim1, m1_rpb), Math.min(m1_dim2, m1_cpb));
		
		// size of right input (vector)
		footprint += OptimizerUtils.estimateSize(m2_dim1, m2_dim2);
		
		// size of the output (only boundary block is merged)
		footprint += OptimizerUtils.estimateSize(Math.min(m1_dim1, m1_rpb), Math.min(m1_dim2+m2_dim2, m1_cpb));
		
		return footprint;
	}
	
	/**
	 * 
	 * @param m1_dim1
	 * @param m1_dim2
	 * @param m2_dim1
	 * @param m2_dim2
	 * @return
	 */
	private static AppendMethod optFindAppendMethod( long m1_dim1, long m1_dim2, long m2_dim1, long m2_dim2, long m1_rpb, long m1_cpb )
	{
		if(FORCED_APPEND_METHOD != null) {
			return FORCED_APPEND_METHOD;
		}
		
		//check for best case (map-only)		
		if(    m2_dim1 >= 1 && m2_dim2 >= 1 // rhs dims known 				
			&& m2_dim2 <= m1_cpb  ) //rhs is smaller than column block 
		{
			double footprint = BinaryOp.footprintInMapper(m1_dim1, m1_dim2, m2_dim1, m2_dim2, m1_rpb, m1_cpb);
			if ( footprint < APPEND_MEM_MULTIPLIER * OptimizerUtils.getRemoteMemBudgetMap(true) )
				return AppendMethod.MR_MAPPEND;
		}
		
		//check for in-block append (reduce-only)
		if( m1_dim2 >= 1 && m2_dim2 >= 0 //column dims known
			&& m1_dim2+m2_dim2 <= m1_cpb ) //output has one column block
		{
			return AppendMethod.MR_RAPPEND;
		}
		
		//general case (map and reduce)
		return AppendMethod.MR_GAPPEND; 	
	}
	
	private static AppendMethod optFindAppendSPMethod( long m1_dim1, long m1_dim2, long m2_dim1, long m2_dim2, long m2_rpb, long m2_cpb, long m2_nnz )
	{
		if(FORCED_APPEND_METHOD != null) {
			return FORCED_APPEND_METHOD;
		}
		
		//check for best case (map-only w/o shuffle)		
		if(    m2_dim1 >= 1 && m2_dim2 >= 1 // rhs dims known 				
			&& m2_dim2 <= m2_cpb  ) //rhs is smaller than column block 
		{
			if( OptimizerUtils.checkSparkBroadcastMemoryBudget(m2_dim1, m2_dim2, m2_rpb, m2_cpb, m2_nnz) ) {
				return AppendMethod.MR_MAPPEND;
			}
		}
		
		//check for in-block append (reduce-only)
		if( m1_dim2 >= 1 && m2_dim2 >= 0 //column dims known
			&& m1_dim2+m2_dim2 <= m2_cpb ) //output has one column block
		{
			return AppendMethod.MR_RAPPEND;
		}
		
		// if(mc1.getCols() % mc1.getColsPerBlock() == 0) {
		if(m1_dim2 % m2_cpb == 0) {
			return AppendMethod.SP_GAlignedAppend;
		}
		
		//general case (map and reduce)
		return AppendMethod.MR_GAPPEND; 	
	}

	/**
	 * 
	 * @param rightInput
	 * @return
	 */
	private static boolean requiresPartitioning( Hop rightInput )
	{
		return (   rightInput.dimsKnown() //known input size 
                && rightInput.getDim1()*rightInput.getDim2() > DistributedCacheInput.PARTITION_SIZE);
	}
	
	/**
	 * 
	 * @param left
	 * @param right
	 * @return
	 */
	private static boolean requiresReplication( Hop left, Hop right )
	{
		return (!(left.getDim2()>=1 && right.getDim2()>=1) //cols of any input unknown 
				||(left.getDim2() > 1 && right.getDim2()==1 && left.getDim2()>=left.getColsInBlock() ) //col MV and more than 1 block
				||(left.getDim1() > 1 && right.getDim1()==1 && left.getDim1()>=left.getRowsInBlock() )); //row MV and more than 1 block
	}

	private MMBinaryMethod optFindMMBinaryMethodSpark(Hop left, Hop right) {
		long m1_dim1 = left.getDim1();
		long m1_dim2 = left.getDim2();
		long m2_dim1 =  right.getDim1();
		long m2_dim2 = right.getDim2();
		long m1_rpb = left.getRowsInBlock();
		long m1_cpb = left.getColsInBlock();
		
		//MR_BINARY_UAGG_CHAIN only applied if result is column/row vector of MV binary operation.
		if( right instanceof AggUnaryOp && right.getInput().get(0) == left  //e.g., P / rowSums(P)
			&& ((((AggUnaryOp) right).getDirection() == Direction.Row && m1_dim2 > 1 && m1_dim2 <= m1_cpb ) //single column block
		    ||  (((AggUnaryOp) right).getDirection() == Direction.Col && m1_dim1 > 1 && m1_dim1 <= m1_rpb ))) //single row block
		{
			return MMBinaryMethod.MR_BINARY_UAGG_CHAIN;
		}
		
		//MR_BINARY_M currently only applied for MV because potential partitioning job may cause additional latency for VV.
		if( m2_dim1 >= 1 && m2_dim2 >= 1 // rhs dims known 
			&& ((m1_dim2 >= 1 && m2_dim2 == 1)  //rhs column vector	
			  ||(m1_dim1 >= 1 && m2_dim1 == 1 )) ) //rhs row vector
		{
			double size = OptimizerUtils.estimateSize(m2_dim1, m2_dim2);
			if( OptimizerUtils.checkSparkBroadcastMemoryBudget(size) ) {
				return MMBinaryMethod.MR_BINARY_M;
			}
		}
		
		//MR_BINARY_R as robust fallback strategy
		return MMBinaryMethod.MR_BINARY_R;
	}
	
	/**
	 * 
	 * @param left
	 * @param right
	 * @return
	 */
	private MMBinaryMethod optFindMMBinaryMethod(Hop left, Hop right)
	{
		long m1_dim1 = left.getDim1();
		long m1_dim2 = left.getDim2();
		long m2_dim1 =  right.getDim1();
		long m2_dim2 = right.getDim2();
		long m1_rpb = left.getRowsInBlock();
		long m1_cpb = left.getColsInBlock();
		
		//MR_BINARY_OUTER only applied if outer vector operation 
		if( outer ) {
			return MMBinaryMethod.MR_BINARY_OUTER_R;
		}
		
		//MR_BINARY_UAGG_CHAIN only applied if result is column/row vector of MV binary operation.
		if( right instanceof AggUnaryOp && right.getInput().get(0) == left  //e.g., P / rowSums(P)
			&& ((((AggUnaryOp) right).getDirection() == Direction.Row && m1_dim2 > 1 && m1_dim2 <= m1_cpb ) //single column block
		    ||  (((AggUnaryOp) right).getDirection() == Direction.Col && m1_dim1 > 1 && m1_dim1 <= m1_rpb ))) //single row block
		{
			return MMBinaryMethod.MR_BINARY_UAGG_CHAIN;
		}
		
		//MR_BINARY_M currently only applied for MV because potential partitioning job may cause additional latency for VV.
		if( m2_dim1 >= 1 && m2_dim2 >= 1 // rhs dims known 
			&& ((m1_dim2 >1 && m2_dim2 == 1)  //rhs column vector	
			  ||(m1_dim1 >1 && m2_dim1 == 1 )) ) //rhs row vector
		{
			double footprint = BinaryOp.footprintInMapper(m1_dim1, m1_dim2, m2_dim1, m2_dim2, m1_rpb, m1_cpb);
			if ( footprint < OptimizerUtils.getRemoteMemBudgetMap(true) )
				return MMBinaryMethod.MR_BINARY_M;		
		}
		
		//MR_BINARY_R as robust fallback strategy
		return MMBinaryMethod.MR_BINARY_R;
	}
	
	
	
	@Override
	public void refreshSizeInformation()
	{
		Hop input1 = getInput().get(0);
		Hop input2 = getInput().get(1);		
		DataType dt1 = input1.getDataType();
		DataType dt2 = input2.getDataType();
		
		if ( getDataType() == DataType.SCALAR ) 
		{
			//do nothing always known
			setDim1(0);
			setDim2(0);
		}
		else //MATRIX OUTPUT
		{
			//TODO quantile
			if( op == OpOp2.APPEND )
			{
				setDim1( (input1.getDim1()>0) ? input1.getDim1() : input2.getDim1() );
					
				//ensure both columns are known, otherwise dangerous underestimation due to +(-1)
				if( input1.getDim2()>0 && input2.getDim2()>0 )
					setDim2( input1.getDim2() + input2.getDim2() );
				//ensure both nnz are known, otherwise dangerous underestimation due to +(-1)
				if( input1.getNnz()>0 && input2.getNnz()>0 )
					setNnz( input1.getNnz() + input2.getNnz() );
			}
			else if ( op == OpOp2.SOLVE )
			{
				//normally the second input would be of equal size as the output 
				//however, since we use qr internally, it also supports squared first inputs
				setDim1( input1.getDim2() );
				setDim2( input2.getDim2() );
			}
			else //general case
			{
				long ldim1, ldim2, lnnz1 = -1;
				
				if( dt1 == DataType.MATRIX && dt2 == DataType.SCALAR )
				{
					ldim1 = input1.getDim1();
					ldim2 = input1.getDim2();
					lnnz1 = input1.getNnz();
				}
				else if( dt1 == DataType.SCALAR && dt2 == DataType.MATRIX  ) 
				{
					ldim1 = input2.getDim1();
					ldim2 = input2.getDim2();
				}
				else //MATRIX - MATRIX 
				{
					//propagate if either input is known, rows need always be identical,
					//for cols we need to be careful with regard to matrix-vector operations
					if( outer ) //OUTER VECTOR OPERATION
					{
						ldim1 = input1.getDim1();
						ldim2 = input2.getDim2();
					}
					else //GENERAL CASE
					{
						ldim1 = (input1.getDim1()>0) ? input1.getDim1()
								: ((input2.getDim1()>1)?input2.getDim1():-1);
						ldim2 = (input1.getDim2()>0) ? input1.getDim2() 
								: ((input2.getDim2()>1)?input2.getDim2():-1);
						lnnz1 = input1.getNnz();
					}
				}
				
				setDim1( ldim1 );
				setDim2( ldim2 );
				
				//update nnz only if we can ensure exact results, 
				//otherwise propagated via worst-case estimates
				if(    op == OpOp2.POW 
					|| (input2 instanceof LiteralOp && OptimizerUtils.isBinaryOpConditionalSparseSafeExact(op, (LiteralOp)input2)) ) 
				{
					setNnz( lnnz1 );
				}
			}
		}	
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		BinaryOp ret = new BinaryOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret.op = op;
		ret.outer = outer;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof BinaryOp) )
			return false;
		
		BinaryOp that2 = (BinaryOp)that;
		return (   op == that2.op
				&& outer == that2.outer
				&& getInput().get(0) == that2.getInput().get(0)
				&& getInput().get(1) == that2.getInput().get(1));
	}
	
	/**
	 * 
	 * @param op
	 * @return
	 */
	public boolean supportsMatrixScalarOperations()
	{
		return (   op==OpOp2.PLUS    ||op==OpOp2.MINUS 
		         ||op==OpOp2.MULT    ||op==OpOp2.DIV
		         ||op==OpOp2.MODULUS ||op==OpOp2.INTDIV
		         ||op==OpOp2.LESS    ||op==OpOp2.LESSEQUAL
		         ||op==OpOp2.GREATER ||op==OpOp2.GREATEREQUAL
		         ||op==OpOp2.EQUAL   ||op==OpOp2.NOTEQUAL
		         ||op==OpOp2.MIN     ||op==OpOp2.MAX
		         ||op==OpOp2.AND     ||op==OpOp2.OR
		         ||op==OpOp2.LOG     ||op==OpOp2.POW );
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isPPredOperation()
	{
		return (   op==OpOp2.LESS    ||op==OpOp2.LESSEQUAL
		         ||op==OpOp2.GREATER ||op==OpOp2.GREATEREQUAL
		         ||op==OpOp2.EQUAL   ||op==OpOp2.NOTEQUAL);
	}
}
