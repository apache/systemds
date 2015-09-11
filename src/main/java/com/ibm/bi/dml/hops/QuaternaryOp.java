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

import com.ibm.bi.dml.hops.Hop.MultiThreadedHop;
import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.DataPartition;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.RepMat;
import com.ibm.bi.dml.lops.Transform;
import com.ibm.bi.dml.lops.UnaryCP;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.lops.WeightedCrossEntropy;
import com.ibm.bi.dml.lops.WeightedCrossEntropyR;
import com.ibm.bi.dml.lops.WeightedDivMM;
import com.ibm.bi.dml.lops.WeightedCrossEntropy.WCeMMType;
import com.ibm.bi.dml.lops.WeightedDivMM.WDivMMType;
import com.ibm.bi.dml.lops.WeightedDivMMR;
import com.ibm.bi.dml.lops.WeightedSigmoid;
import com.ibm.bi.dml.lops.WeightedSigmoid.WSigmoidType;
import com.ibm.bi.dml.lops.WeightedSigmoidR;
import com.ibm.bi.dml.lops.WeightedSquaredLoss;
import com.ibm.bi.dml.lops.WeightedSquaredLoss.WeightsType;
import com.ibm.bi.dml.lops.WeightedSquaredLossR;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.mapred.DistributedCacheInput;

/** 
 * Note: this hop should be called AggQuaternaryOp in consistency with AggUnaryOp and AggBinaryOp;
 * however, since there does not exist a real QuaternaryOp yet - we can leave it as is for now. 
 */
public class QuaternaryOp extends Hop implements MultiThreadedHop
{

	//config influencing mr operator selection (for testing purposes only) 
	public static boolean FORCE_REPLICATION = false;
	
	private OpOp4 _op = null;
	private int _maxNumThreads = -1; //-1 for unlimited
	
	//wsloss-specific attributes
	private boolean _postWeights = false;
	
	//wsigmoid-specific attributes
	private boolean _logout = false;
	private boolean _minusin = false;
	
	//wdivmm-specific attributes
	private boolean _left = false;
	
	private QuaternaryOp() {
		//default constructor for clone
	}
	
	/**
	 * Constructor for wsloss.
	 * 
	 * @param l
	 * @param dt
	 * @param vt
	 * @param o
	 * @param inX
	 * @param inU
	 * @param inV
	 * @param inW
	 * @param post
	 */
	public QuaternaryOp(String l, DataType dt, ValueType vt, Hop.OpOp4 o,
			Hop inX, Hop inU, Hop inV, Hop inW, boolean post) 
	{			
		this(l, dt, vt, o, inX, inU, inV);
		getInput().add(3, inW);
		inW.getParent().add(this);
		
		_postWeights = post;
	}
	
	/**
	 * Constructor for wsigmoid.
	 * 
	 * @param l
	 * @param dt
	 * @param vt
	 * @param o
	 * @param inX
	 * @param inU
	 * @param inV
	 * @param logout
	 * @param minusin
	 */
	public QuaternaryOp(String l, DataType dt, ValueType vt, Hop.OpOp4 o,
			Hop inX, Hop inU, Hop inV, boolean logout, boolean minusin) 
	{
		this(l, dt, vt, o, inX, inU, inV);
		
		_logout = logout;
		_minusin = minusin;
	}
	
	/**
	 * Constructor for wdivmm.
	 * 
	 * @param l
	 * @param dt
	 * @param vt
	 * @param o
	 * @param inX
	 * @param inU
	 * @param inV
	 * @param logout
	 * @param minusin
	 */
	public QuaternaryOp(String l, DataType dt, ValueType vt, Hop.OpOp4 o,
			Hop inX, Hop inU, Hop inV, boolean left) 
	{
		this(l, dt, vt, o, inX, inU, inV);
		
		_left = left;
	}
	
	/**
	 * 
	 * @param l
	 * @param dt
	 * @param vt
	 * @param o
	 * @param inX
	 * @param inU
	 * @param inV
	 */
	public QuaternaryOp(String l, DataType dt, ValueType vt, Hop.OpOp4 o, Hop inX, Hop inU, Hop inV) 
	{
		super(l, dt, vt);
		_op = o;
		getInput().add(0, inX);
		getInput().add(1, inU);
		getInput().add(2, inV);
		inX.getParent().add(this);
		inU.getParent().add(this);
		inV.getParent().add(this);
	}
	
	public OpOp4 getOp(){
		return _op;
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
		//return already created lops
		if( getLops() != null )
			return getLops();

		try 
		{
			ExecType et = optFindExecType();
			
			switch( _op ) {
				case WSLOSS: {
					WeightsType wtype = checkWeightsType();
					
					if( et == ExecType.CP )
						constructCPLopsWeightedSquaredLoss(wtype);
					else if( et == ExecType.MR )
						constructMRLopsWeightedSquaredLoss(wtype);
					else if( et == ExecType.SPARK )
						constructSparkLopsWeightedSquaredLoss(wtype);
					else
						throw new HopsException("Unsupported quaternaryop-wsloss exec type: "+et);
					break;
				}
				
				case WSIGMOID:{
					WSigmoidType wtype = checkWSigmoidType();
					
					if( et == ExecType.CP )
						constructCPLopsWeightedSigmoid(wtype);
					else if( et == ExecType.MR )
						constructMRLopsWeightedSigmoid(wtype);
					else if( et == ExecType.SPARK )
						constructSparkLopsWeightedSigmoid(wtype);
					else
						throw new HopsException("Unsupported quaternaryop-wsigmoid exec type: "+et);
					break;
				}
				
				case WDIVMM:{
					WDivMMType wtype = checkWDivMMType();
					
					if( et == ExecType.CP )
						constructCPLopsWeightedDivMM(wtype);
					else if( et == ExecType.MR )
						constructMRLopsWeightedDivMM(wtype);
					else if( et == ExecType.SPARK )
						constructSparkLopsWeightedDivMM(wtype);
					else
						throw new HopsException("Unsupported quaternaryop-wdivmm exec type: "+et);
					break;
				}
				
				case WCEMM:{
					WCeMMType wtype = WCeMMType.BASIC;
					
					if( et == ExecType.CP )
						constructCPLopsWeightedCeMM(wtype);
					else if( et == ExecType.MR )
						constructMRLopsWeightedCeMM(wtype);
					else if( et == ExecType.SPARK )
						constructSparkLopsWeightedCeMM(wtype);
					else
						throw new HopsException("Unsupported quaternaryop-wcemm exec type: "+et);
					break;
				}
				
				default:
					throw new HopsException(this.printErrorLocation() + "Unknown QuaternaryOp (" + _op + ") while constructing Lops");
			}
		} 
		catch(LopsException e) {
			throw new HopsException(this.printErrorLocation() + "error constructing lops for QuaternaryOp." , e);
		}
	
		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();
				
		return getLops();
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += "q(" + HopsOpOp4String.get(_op) + ")";
		return s;
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
	public boolean allowsAllExecTypes()
	{
		return true;
	}

	/**
	 * 
	 * @param wtype 
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructCPLopsWeightedSquaredLoss(WeightsType wtype) 
		throws HopsException, LopsException
	{
		WeightedSquaredLoss wsloss = new WeightedSquaredLoss(
				getInput().get(0).constructLops(),
				getInput().get(1).constructLops(),
				getInput().get(2).constructLops(),
				getInput().get(3).constructLops(),
				getDataType(), getValueType(), wtype, ExecType.CP);
		
		//set degree of parallelism
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		wsloss.setNumThreads(k);
		
		setOutputDimensions( wsloss );
		setLineNumbers( wsloss );
		setLops( wsloss );
	}
	
	/**
	 * 
	 * @param wtype 
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructMRLopsWeightedSquaredLoss(WeightsType wtype) 
		throws HopsException, LopsException
	{
		//NOTE: the common case for wsloss are factors U/V with a rank of 10s to 100s; the current runtime only
		//supports single block outer products (U/V rank <= blocksize, i.e., 1000 by default); we enforce this
		//by applying the hop rewrite for Weighted Squared Loss only if this constraint holds. 
		
		Hop X = getInput().get(0);
		Hop U = getInput().get(1);
		Hop V = getInput().get(2);
		Hop W = getInput().get(3);
		
		//MR operator selection, part1
		double m1Size = OptimizerUtils.estimateSize(U.getDim1(), U.getDim2()); //size U
		double m2Size = OptimizerUtils.estimateSize(V.getDim1(), V.getDim2()); //size V
		boolean isMapWsloss = (wtype == WeightsType.NONE && m1Size+m2Size < OptimizerUtils.getRemoteMemBudgetMap(true)); 
		
		if( !FORCE_REPLICATION && isMapWsloss ) //broadcast
		{
			//partitioning of U
			boolean needPartU = !U.dimsKnown() || U.getDim1() * U.getDim2() > DistributedCacheInput.PARTITION_SIZE;
			Lop lU = U.constructLops();
			if( needPartU ){ //requires partitioning
				lU = new DataPartition(lU, DataType.MATRIX, ValueType.DOUBLE, (m1Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
				lU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), getRowsInBlock(), getColsInBlock(), U.getNnz());
				setLineNumbers(lU);	
			}
			
			//partitioning of V
			boolean needPartV = !V.dimsKnown() || V.getDim1() * V.getDim2() > DistributedCacheInput.PARTITION_SIZE;
			Lop lV = V.constructLops();
			if( needPartV ){ //requires partitioning
				lV = new DataPartition(lV, DataType.MATRIX, ValueType.DOUBLE, (m2Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
				lV.getOutputParameters().setDimensions(V.getDim1(), V.getDim2(), getRowsInBlock(), getColsInBlock(), V.getNnz());
				setLineNumbers(lV);	
			}
			
			//map-side wsloss always with broadcast
			Lop wsloss = new WeightedSquaredLoss( X.constructLops(), lU, lV, W.constructLops(), 
					DataType.MATRIX, ValueType.DOUBLE, wtype, ExecType.MR);
			wsloss.getOutputParameters().setDimensions(1, 1, X.getRowsInBlock(), X.getColsInBlock(), -1);
			setLineNumbers(wsloss);
			
			Group grp = new Group(wsloss, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
			grp.getOutputParameters().setDimensions(1, 1, X.getRowsInBlock(), X.getColsInBlock(), -1);
			setLineNumbers(grp);
			
			Aggregate agg1 = new Aggregate(grp, HopsAgg2Lops.get(AggOp.SUM), DataType.MATRIX, ValueType.DOUBLE, ExecType.MR);
			agg1.setupCorrectionLocation(CorrectionLocationType.NONE); // aggregation uses kahanSum 
			agg1.getOutputParameters().setDimensions(1, 1, X.getRowsInBlock(), X.getColsInBlock(), -1);
			setLineNumbers(agg1);
			
			UnaryCP unary1 = new UnaryCP(agg1, HopsOpOp1LopsUS.get(OpOp1.CAST_AS_SCALAR), getDataType(), getValueType());
			unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			setLineNumbers(unary1);
			setLops(unary1);
		}
		else //general case
		{
			//MR operator selection part 2
			boolean cacheU = !FORCE_REPLICATION && (m1Size < OptimizerUtils.getRemoteMemBudgetReduce());
			boolean cacheV = !FORCE_REPLICATION && ((!cacheU && m2Size < OptimizerUtils.getRemoteMemBudgetReduce()) 
					        || (cacheU && m1Size+m2Size < OptimizerUtils.getRemoteMemBudgetReduce()));
			
			Group grpX = new Group(X.constructLops(), Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
			grpX.getOutputParameters().setDimensions(X.getDim1(), X.getDim2(), X.getRowsInBlock(), X.getColsInBlock(), -1);
			setLineNumbers(grpX);
			
			Lop grpW = W.constructLops();
			if( grpW.getDataType()==DataType.MATRIX ) {
				grpW = new Group(W.constructLops(), Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
				grpW.getOutputParameters().setDimensions(W.getDim1(), W.getDim2(), W.getRowsInBlock(), W.getColsInBlock(), -1);
				setLineNumbers(grpW);
			}
			
			Lop lU = null;
			if( cacheU ) {
				//partitioning of U for read through distributed cache
				boolean needPartU = !U.dimsKnown() || U.getDim1() * U.getDim2() > DistributedCacheInput.PARTITION_SIZE;
				lU = U.constructLops();
				if( needPartU ){ //requires partitioning
					lU = new DataPartition(lU, DataType.MATRIX, ValueType.DOUBLE, (m1Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
					lU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), getRowsInBlock(), getColsInBlock(), U.getNnz());
					setLineNumbers(lU);	
				}
			}
			else {
				//replication of U for shuffle to target block
				Lop offset = createOffsetLop(V, false); //ncol of t(V) -> nrow of V determines num replicates
				lU = new RepMat(U.constructLops(), offset, true, V.getDataType(), V.getValueType());
				lU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), 
						U.getRowsInBlock(), U.getColsInBlock(), U.getNnz());
				setLineNumbers(lU);
				
				Group grpU = new Group(lU, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
				grpU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), U.getRowsInBlock(), U.getColsInBlock(), -1);
				setLineNumbers(grpU);
				lU = grpU;
			}
			
			Lop lV = null;
			if( cacheV ) {
				//partitioning of V for read through distributed cache
				boolean needPartV = !V.dimsKnown() || V.getDim1() * V.getDim2() > DistributedCacheInput.PARTITION_SIZE;
				lV = V.constructLops();
				if( needPartV ){ //requires partitioning
					lV = new DataPartition(lV, DataType.MATRIX, ValueType.DOUBLE, (m2Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
					lV.getOutputParameters().setDimensions(V.getDim1(), V.getDim2(), getRowsInBlock(), getColsInBlock(), V.getNnz());
					setLineNumbers(lV);	
				}
			}
			else {
				//replication of t(V) for shuffle to target block
				Transform ltV = new Transform( V.constructLops(), HopsTransf2Lops.get(ReOrgOp.TRANSPOSE), getDataType(), getValueType(), ExecType.MR);
				ltV.getOutputParameters().setDimensions(V.getDim2(), V.getDim1(), 
						V.getColsInBlock(), V.getRowsInBlock(), V.getNnz());
				setLineNumbers(ltV);
				
				Lop offset = createOffsetLop(U, false); //nrow of U determines num replicates
				lV = new RepMat(ltV, offset, false, V.getDataType(), V.getValueType());
				lV.getOutputParameters().setDimensions(V.getDim2(), V.getDim1(), 
						V.getColsInBlock(), V.getRowsInBlock(), V.getNnz());
				setLineNumbers(lV);
				
				Group grpV = new Group(lV, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
				grpV.getOutputParameters().setDimensions(V.getDim2(), V.getDim1(), V.getColsInBlock(), V.getRowsInBlock(), -1);
				setLineNumbers(grpV);
				lV = grpV;
			}
			
			//reduce-side wsloss w/ or without broadcast
			Lop wsloss = new WeightedSquaredLossR( 
					grpX, lU, lV, grpW, DataType.MATRIX, ValueType.DOUBLE, wtype, cacheU, cacheV, ExecType.MR);
			wsloss.getOutputParameters().setDimensions(1, 1, X.getRowsInBlock(), X.getColsInBlock(), -1);
			setLineNumbers(wsloss);
			
			Group grp = new Group(wsloss, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
			grp.getOutputParameters().setDimensions(1, 1, X.getRowsInBlock(), X.getColsInBlock(), -1);
			setLineNumbers(grp);
			
			Aggregate agg1 = new Aggregate(grp, HopsAgg2Lops.get(AggOp.SUM), DataType.MATRIX, ValueType.DOUBLE, ExecType.MR);
			agg1.setupCorrectionLocation(CorrectionLocationType.NONE); // aggregation uses kahanSum 
			agg1.getOutputParameters().setDimensions(1, 1, X.getRowsInBlock(), X.getColsInBlock(), -1);
			setLineNumbers(agg1);
			
			UnaryCP unary1 = new UnaryCP(agg1, HopsOpOp1LopsUS.get(OpOp1.CAST_AS_SCALAR), getDataType(), getValueType());
			unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			setLineNumbers(unary1);
			setLops(unary1);
		}
	}
	
	/**
	 * 
	 * @param wtype
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructSparkLopsWeightedSquaredLoss(WeightsType wtype) 
		throws HopsException, LopsException
	{
		//NOTE: the common case for wsloss are factors U/V with a rank of 10s to 100s; the current runtime only
		//supports single block outer products (U/V rank <= blocksize, i.e., 1000 by default); we enforce this
		//by applying the hop rewrite for Weighted Squared Loss only if this constraint holds. 
		
		//Notes: Any broadcast needs to fit twice in local memory because we partition the input in cp,
		//and needs to fit once in executor broadcast memory. The 2GB broadcast constraint is no longer
		//required because the max_int byte buffer constraint has been fixed in Spark 1.4 
		double memBudgetExec = SparkExecutionContext.getBroadcastMemoryBudget();
		double memBudgetLocal = OptimizerUtils.getLocalMemBudget();

		Hop X = getInput().get(0);
		Hop U = getInput().get(1);
		Hop V = getInput().get(2);
		Hop W = getInput().get(3);
		
		//MR operator selection, part1
		double m1Size = OptimizerUtils.estimateSize(U.getDim1(), U.getDim2()); //size U
		double m2Size = OptimizerUtils.estimateSize(V.getDim1(), V.getDim2()); //size V
		boolean isMapWsloss = (wtype == WeightsType.NONE && m1Size+m2Size < memBudgetExec
				&& 2*m1Size < memBudgetLocal && 2*m2Size < memBudgetLocal); 
		
		if( !FORCE_REPLICATION && isMapWsloss ) //broadcast
		{
			//map-side wsloss always with broadcast
			Lop wsloss = new WeightedSquaredLoss( X.constructLops(), U.constructLops(), V.constructLops(), W.constructLops(), 
					DataType.MATRIX, ValueType.DOUBLE, wtype, ExecType.SPARK);
			setOutputDimensions(wsloss);
			setLineNumbers(wsloss);
			setLops(wsloss);
		}
		else //general case
		{
			//MR operator selection part 2
			boolean cacheU = !FORCE_REPLICATION && (m1Size < memBudgetExec && 2*m1Size < memBudgetLocal);
			boolean cacheV = !FORCE_REPLICATION && ((!cacheU && m2Size < memBudgetExec ) 
					        || (cacheU && m1Size+m2Size < memBudgetExec)) && 2*m2Size < memBudgetLocal;
			
			//reduce-side wsloss w/ or without broadcast
			Lop wsloss = new WeightedSquaredLossR( 
					X.constructLops(), U.constructLops(), V.constructLops(), W.constructLops(), 
					DataType.MATRIX, ValueType.DOUBLE, wtype, cacheU, cacheV, ExecType.SPARK);
			setOutputDimensions(wsloss);
			setLineNumbers(wsloss);
			setLops(wsloss);
		}
	}

	/**
	 * 
	 * @param wtype 
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructCPLopsWeightedSigmoid(WSigmoidType wtype) 
		throws HopsException, LopsException
	{
		WeightedSigmoid wsig = new WeightedSigmoid(
				getInput().get(0).constructLops(),
				getInput().get(1).constructLops(),
				getInput().get(2).constructLops(),
				getDataType(), getValueType(), wtype, ExecType.CP);
		
		//set degree of parallelism
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		wsig.setNumThreads(k);
		
		setOutputDimensions( wsig );
		setLineNumbers( wsig );
		setLops( wsig );
	}
	
	/**
	 * 
	 * @param wtype
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructMRLopsWeightedSigmoid( WSigmoidType wtype ) 
		throws HopsException, LopsException
	{
		//NOTE: the common case for wsigmoid are factors U/V with a rank of 10s to 100s; the current runtime only
		//supports single block outer products (U/V rank <= blocksize, i.e., 1000 by default); we enforce this
		//by applying the hop rewrite for Weighted Squared Loss only if this constraint holds. 
		
		Hop X = getInput().get(0);
		Hop U = getInput().get(1);
		Hop V = getInput().get(2);
		
		//MR operator selection, part1
		double m1Size = OptimizerUtils.estimateSize(U.getDim1(), U.getDim2()); //size U
		double m2Size = OptimizerUtils.estimateSize(V.getDim1(), V.getDim2()); //size V
		boolean isMapWsloss = (m1Size+m2Size < OptimizerUtils.getRemoteMemBudgetMap(true)); 
		
		if( !FORCE_REPLICATION && isMapWsloss ) //broadcast
		{
			//partitioning of U
			boolean needPartU = !U.dimsKnown() || U.getDim1() * U.getDim2() > DistributedCacheInput.PARTITION_SIZE;
			Lop lU = U.constructLops();
			if( needPartU ){ //requires partitioning
				lU = new DataPartition(lU, DataType.MATRIX, ValueType.DOUBLE, (m1Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
				lU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), getRowsInBlock(), getColsInBlock(), U.getNnz());
				setLineNumbers(lU);	
			}
			
			//partitioning of V
			boolean needPartV = !V.dimsKnown() || V.getDim1() * V.getDim2() > DistributedCacheInput.PARTITION_SIZE;
			Lop lV = V.constructLops();
			if( needPartV ){ //requires partitioning
				lV = new DataPartition(lV, DataType.MATRIX, ValueType.DOUBLE, (m2Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
				lV.getOutputParameters().setDimensions(V.getDim1(), V.getDim2(), getRowsInBlock(), getColsInBlock(), V.getNnz());
				setLineNumbers(lV);	
			}
			
			//map-side wsloss always with broadcast
			Lop wsigmoid = new WeightedSigmoid( X.constructLops(), lU, lV,  
					DataType.MATRIX, ValueType.DOUBLE, wtype, ExecType.MR);
			setOutputDimensions(wsigmoid);
			setLineNumbers(wsigmoid);
			setLops( wsigmoid );
			
			//in contrast to wsloss no aggregation required 
		}
		else //general case
		{
			//MR operator selection part 2
			boolean cacheU = !FORCE_REPLICATION && (m1Size < OptimizerUtils.getRemoteMemBudgetReduce());
			boolean cacheV = !FORCE_REPLICATION && ((!cacheU && m2Size < OptimizerUtils.getRemoteMemBudgetReduce()) 
					        || (cacheU && m1Size+m2Size < OptimizerUtils.getRemoteMemBudgetReduce()));
			
			Group grpX = new Group(X.constructLops(), Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
			grpX.getOutputParameters().setDimensions(X.getDim1(), X.getDim2(), X.getRowsInBlock(), X.getColsInBlock(), X.getNnz());
			setLineNumbers(grpX);
			
			Lop lU = null;
			if( cacheU ) {
				//partitioning of U for read through distributed cache
				boolean needPartU = !U.dimsKnown() || U.getDim1() * U.getDim2() > DistributedCacheInput.PARTITION_SIZE;
				lU = U.constructLops();
				if( needPartU ){ //requires partitioning
					lU = new DataPartition(lU, DataType.MATRIX, ValueType.DOUBLE, (m1Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
					lU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), getRowsInBlock(), getColsInBlock(), U.getNnz());
					setLineNumbers(lU);	
				}
			}
			else {
				//replication of U for shuffle to target block
				Lop offset = createOffsetLop(V, false); //ncol of t(V) -> nrow of V determines num replicates
				lU = new RepMat(U.constructLops(), offset, true, V.getDataType(), V.getValueType());
				lU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), 
						U.getRowsInBlock(), U.getColsInBlock(), U.getNnz());
				setLineNumbers(lU);
				
				Group grpU = new Group(lU, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
				grpU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), U.getRowsInBlock(), U.getColsInBlock(), -1);
				setLineNumbers(grpU);
				lU = grpU;
			}
			
			Lop lV = null;
			if( cacheV ) {
				//partitioning of V for read through distributed cache
				boolean needPartV = !V.dimsKnown() || V.getDim1() * V.getDim2() > DistributedCacheInput.PARTITION_SIZE;
				lV = V.constructLops();
				if( needPartV ){ //requires partitioning
					lV = new DataPartition(lV, DataType.MATRIX, ValueType.DOUBLE, (m2Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
					lV.getOutputParameters().setDimensions(V.getDim1(), V.getDim2(), getRowsInBlock(), getColsInBlock(), V.getNnz());
					setLineNumbers(lV);	
				}
			}
			else {
				//replication of t(V) for shuffle to target block
				Transform ltV = new Transform( V.constructLops(), HopsTransf2Lops.get(ReOrgOp.TRANSPOSE), getDataType(), getValueType(), ExecType.MR);
				ltV.getOutputParameters().setDimensions(V.getDim2(), V.getDim1(), 
						V.getColsInBlock(), V.getRowsInBlock(), V.getNnz());
				setLineNumbers(ltV);
				
				Lop offset = createOffsetLop(U, false); //nrow of U determines num replicates
				lV = new RepMat(ltV, offset, false, V.getDataType(), V.getValueType());
				lV.getOutputParameters().setDimensions(V.getDim2(), V.getDim1(), 
						V.getColsInBlock(), V.getRowsInBlock(), V.getNnz());
				setLineNumbers(lV);
				
				Group grpV = new Group(lV, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
				grpV.getOutputParameters().setDimensions(V.getDim2(), V.getDim1(), V.getColsInBlock(), V.getRowsInBlock(), -1);
				setLineNumbers(grpV);
				lV = grpV;
			}
			
			//reduce-side wsloss w/ or without broadcast
			Lop wsigmoid = new WeightedSigmoidR( 
					grpX, lU, lV, DataType.MATRIX, ValueType.DOUBLE, wtype, cacheU, cacheV, ExecType.MR);
			setOutputDimensions(wsigmoid);
			setLineNumbers(wsigmoid);
			setLops(wsigmoid);
			
			//in contrast to wsloss no aggregation required 	
		}
	}
	
	/**
	 * 
	 * @param wtype
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructSparkLopsWeightedSigmoid( WSigmoidType wtype ) 
		throws HopsException, LopsException
	{
		//NOTE: the common case for wsigmoid are factors U/V with a rank of 10s to 100s; the current runtime only
		//supports single block outer products (U/V rank <= blocksize, i.e., 1000 by default); we enforce this
		//by applying the hop rewrite for Weighted Squared Loss only if this constraint holds. 

		//Notes: Any broadcast needs to fit twice in local memory because we partition the input in cp,
		//and needs to fit once in executor broadcast memory. The 2GB broadcast constraint is no longer
		//required because the max_int byte buffer constraint has been fixed in Spark 1.4 
		double memBudgetExec = SparkExecutionContext.getBroadcastMemoryBudget();
		double memBudgetLocal = OptimizerUtils.getLocalMemBudget();

		Hop X = getInput().get(0);
		Hop U = getInput().get(1);
		Hop V = getInput().get(2);
		
		//MR operator selection, part1
		double m1Size = OptimizerUtils.estimateSize(U.getDim1(), U.getDim2()); //size U
		double m2Size = OptimizerUtils.estimateSize(V.getDim1(), V.getDim2()); //size V
		boolean isMapWsloss = (m1Size+m2Size < memBudgetExec
				&& 2*m1Size<memBudgetLocal && 2*m2Size<memBudgetLocal); 
		
		if( !FORCE_REPLICATION && isMapWsloss ) //broadcast
		{
			//map-side wsloss always with broadcast
			Lop wsigmoid = new WeightedSigmoid( X.constructLops(), U.constructLops(), V.constructLops(),  
					DataType.MATRIX, ValueType.DOUBLE, wtype, ExecType.SPARK);
			setOutputDimensions(wsigmoid);
			setLineNumbers(wsigmoid);
			setLops( wsigmoid );
		}
		else //general case
		{
			//MR operator selection part 2
			boolean cacheU = !FORCE_REPLICATION && (m1Size < memBudgetExec && 2*m1Size < memBudgetLocal);
			boolean cacheV = !FORCE_REPLICATION && ((!cacheU && m2Size < memBudgetExec ) 
					        || (cacheU && m1Size+m2Size < memBudgetExec)) && 2*m2Size < memBudgetLocal;
			
			//reduce-side wsloss w/ or without broadcast
			Lop wsigmoid = new WeightedSigmoidR( 
					X.constructLops(), U.constructLops(), V.constructLops(), 
					DataType.MATRIX, ValueType.DOUBLE, wtype, cacheU, cacheV, ExecType.SPARK);
			setOutputDimensions(wsigmoid);
			setLineNumbers(wsigmoid);
			setLops(wsigmoid);
		}
	}
	
	/**
	 * 
	 * @param wtype
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructCPLopsWeightedDivMM(WDivMMType wtype) 
		throws HopsException, LopsException
	{
		WeightedDivMM wdiv = new WeightedDivMM(
				getInput().get(0).constructLops(),
				getInput().get(1).constructLops(),
				getInput().get(2).constructLops(),
				getDataType(), getValueType(), wtype, ExecType.CP);
		
		//set degree of parallelism
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		wdiv.setNumThreads(k);
		
		setOutputDimensions( wdiv );
		setLineNumbers( wdiv );
		setLops( wdiv );
	}
	
	/**
	 * 
	 * @param wtype
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructMRLopsWeightedDivMM( WDivMMType wtype ) 
		throws HopsException, LopsException
	{
		//NOTE: the common case for wdivmm are factors U/V with a rank of 10s to 100s; the current runtime only
		//supports single block outer products (U/V rank <= blocksize, i.e., 1000 by default); we enforce this
		//by applying the hop rewrite for Weighted Squared Loss only if this constraint holds. 
		
		Hop X = getInput().get(0);
		Hop U = getInput().get(1);
		Hop V = getInput().get(2);
		
		//MR operator selection, part1
		double m1Size = OptimizerUtils.estimateSize(U.getDim1(), U.getDim2()); //size U
		double m2Size = OptimizerUtils.estimateSize(V.getDim1(), V.getDim2()); //size V
		boolean isMapWsloss = (m1Size+m2Size < OptimizerUtils.getRemoteMemBudgetMap(true)); 
		
		if( !FORCE_REPLICATION && isMapWsloss ) //broadcast
		{
			//partitioning of U
			boolean needPartU = !U.dimsKnown() || U.getDim1() * U.getDim2() > DistributedCacheInput.PARTITION_SIZE;
			Lop lU = U.constructLops();
			if( needPartU ){ //requires partitioning
				lU = new DataPartition(lU, DataType.MATRIX, ValueType.DOUBLE, (m1Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
				lU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), getRowsInBlock(), getColsInBlock(), U.getNnz());
				setLineNumbers(lU);	
			}
			
			//partitioning of V
			boolean needPartV = !V.dimsKnown() || V.getDim1() * V.getDim2() > DistributedCacheInput.PARTITION_SIZE;
			Lop lV = V.constructLops();
			if( needPartV ){ //requires partitioning
				lV = new DataPartition(lV, DataType.MATRIX, ValueType.DOUBLE, (m2Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
				lV.getOutputParameters().setDimensions(V.getDim1(), V.getDim2(), getRowsInBlock(), getColsInBlock(), V.getNnz());
				setLineNumbers(lV);	
			}
			
			//map-side wsloss always with broadcast
			Lop wdivmm = new WeightedDivMM( X.constructLops(), lU, lV,  
					DataType.MATRIX, ValueType.DOUBLE, wtype, ExecType.MR);
			setOutputDimensions(wdivmm);
			setLineNumbers(wdivmm);
			setLops(wdivmm); 
		}
		else //general case
		{
			//MR operator selection part 2 (both cannot happen for wdivmm, otherwise mapwdivmm)
			boolean cacheU = !FORCE_REPLICATION && (m1Size < OptimizerUtils.getRemoteMemBudgetReduce());
			boolean cacheV = !FORCE_REPLICATION && ((!cacheU && m2Size < OptimizerUtils.getRemoteMemBudgetReduce()) 
					        || (cacheU && m1Size+m2Size < OptimizerUtils.getRemoteMemBudgetReduce()));
			
			Group grpX = new Group(X.constructLops(), Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
			grpX.getOutputParameters().setDimensions(X.getDim1(), X.getDim2(), X.getRowsInBlock(), X.getColsInBlock(), X.getNnz());
			setLineNumbers(grpX);
			
			Lop lU = null;
			if( cacheU ) {
				//partitioning of U for read through distributed cache
				boolean needPartU = !U.dimsKnown() || U.getDim1() * U.getDim2() > DistributedCacheInput.PARTITION_SIZE;
				lU = U.constructLops();
				if( needPartU ){ //requires partitioning
					lU = new DataPartition(lU, DataType.MATRIX, ValueType.DOUBLE, (m1Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
					lU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), getRowsInBlock(), getColsInBlock(), U.getNnz());
					setLineNumbers(lU);	
				}
			}
			else {
				//replication of U for shuffle to target block
				Lop offset = createOffsetLop(V, false); //ncol of t(V) -> nrow of V determines num replicates
				lU = new RepMat(U.constructLops(), offset, true, V.getDataType(), V.getValueType());
				lU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), 
						U.getRowsInBlock(), U.getColsInBlock(), U.getNnz());
				setLineNumbers(lU);
				
				Group grpU = new Group(lU, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
				grpU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), U.getRowsInBlock(), U.getColsInBlock(), -1);
				setLineNumbers(grpU);
				lU = grpU;
			}
			
			Lop lV = null;
			if( cacheV ) {
				//partitioning of V for read through distributed cache
				boolean needPartV = !V.dimsKnown() || V.getDim1() * V.getDim2() > DistributedCacheInput.PARTITION_SIZE;
				lV = V.constructLops();
				if( needPartV ){ //requires partitioning
					lV = new DataPartition(lV, DataType.MATRIX, ValueType.DOUBLE, (m2Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
					lV.getOutputParameters().setDimensions(V.getDim1(), V.getDim2(), getRowsInBlock(), getColsInBlock(), V.getNnz());
					setLineNumbers(lV);	
				}
			}
			else {
				//replication of t(V) for shuffle to target block
				Transform ltV = new Transform( V.constructLops(), HopsTransf2Lops.get(ReOrgOp.TRANSPOSE), getDataType(), getValueType(), ExecType.MR);
				ltV.getOutputParameters().setDimensions(V.getDim2(), V.getDim1(), 
						V.getColsInBlock(), V.getRowsInBlock(), V.getNnz());
				setLineNumbers(ltV);
				
				Lop offset = createOffsetLop(U, false); //nrow of U determines num replicates
				lV = new RepMat(ltV, offset, false, V.getDataType(), V.getValueType());
				lV.getOutputParameters().setDimensions(V.getDim2(), V.getDim1(), 
						V.getColsInBlock(), V.getRowsInBlock(), V.getNnz());
				setLineNumbers(lV);
				
				Group grpV = new Group(lV, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
				grpV.getOutputParameters().setDimensions(V.getDim2(), V.getDim1(), V.getColsInBlock(), V.getRowsInBlock(), -1);
				setLineNumbers(grpV);
				lV = grpV;
			}
			
			//reduce-side wsloss w/ or without broadcast
			Lop wdivmm = new WeightedDivMMR( 
					grpX, lU, lV, DataType.MATRIX, ValueType.DOUBLE, wtype, cacheU, cacheV, ExecType.MR);
			setOutputDimensions(wdivmm);
			setLineNumbers(wdivmm);
			setLops(wdivmm);
		}
		
		//in contrast to to wsloss/wsigmoid, wdivmm requires partial aggregation (for the final mm)
		Group grp = new Group(getLops(), Group.OperationTypes.Sort, getDataType(), getValueType());
		setOutputDimensions(grp);
		setLineNumbers(grp);
		
		Aggregate agg1 = new Aggregate(grp, HopsAgg2Lops.get(AggOp.SUM), getDataType(), getValueType(), ExecType.MR);
		// aggregation uses kahanSum but the inputs do not have correction values
		agg1.setupCorrectionLocation(CorrectionLocationType.NONE);  
		setOutputDimensions(agg1);
		setLineNumbers(agg1);
		
		setLops(agg1);
	}
	
	/**
	 * 
	 * @param wtype
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructSparkLopsWeightedDivMM( WDivMMType wtype ) 
		throws HopsException, LopsException
	{
		//NOTE: the common case for wdivmm are factors U/V with a rank of 10s to 100s; the current runtime only
		//supports single block outer products (U/V rank <= blocksize, i.e., 1000 by default); we enforce this
		//by applying the hop rewrite for Weighted DivMM only if this constraint holds. 

		//Notes: Any broadcast needs to fit twice in local memory because we partition the input in cp,
		//and needs to fit once in executor broadcast memory. The 2GB broadcast constraint is no longer
		//required because the max_int byte buffer constraint has been fixed in Spark 1.4 
		double memBudgetExec = SparkExecutionContext.getBroadcastMemoryBudget();
		double memBudgetLocal = OptimizerUtils.getLocalMemBudget();

		Hop X = getInput().get(0);
		Hop U = getInput().get(1);
		Hop V = getInput().get(2);
		
		//MR operator selection, part1
		double m1Size = OptimizerUtils.estimateSize(U.getDim1(), U.getDim2()); //size U
		double m2Size = OptimizerUtils.estimateSize(V.getDim1(), V.getDim2()); //size V
		boolean isMapWsloss = (m1Size+m2Size < memBudgetExec
				&& 2*m1Size<memBudgetLocal && 2*m2Size<memBudgetLocal); 
		
		if( !FORCE_REPLICATION && isMapWsloss ) //broadcast
		{
			//map-side wsloss always with broadcast
			Lop wdivmm = new WeightedDivMM( X.constructLops(), U.constructLops(), V.constructLops(),  
					DataType.MATRIX, ValueType.DOUBLE, wtype, ExecType.SPARK);
			setOutputDimensions(wdivmm);
			setLineNumbers(wdivmm);
			setLops( wdivmm );
		}
		else //general case
		{
			//MR operator selection part 2
			boolean cacheU = !FORCE_REPLICATION && (m1Size < memBudgetExec && 2*m1Size < memBudgetLocal);
			boolean cacheV = !FORCE_REPLICATION && ((!cacheU && m2Size < memBudgetExec ) 
					        || (cacheU && m1Size+m2Size < memBudgetExec)) && 2*m2Size < memBudgetLocal;
			
			//reduce-side wsloss w/ or without broadcast
			Lop wdivmm = new WeightedDivMMR( 
					X.constructLops(), U.constructLops(), V.constructLops(), 
					DataType.MATRIX, ValueType.DOUBLE, wtype, cacheU, cacheV, ExecType.SPARK);
			setOutputDimensions(wdivmm);
			setLineNumbers(wdivmm);
			setLops(wdivmm);
		}
	}
	
	/**
	 * 
	 * @param wtype
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructCPLopsWeightedCeMM(WCeMMType wtype) 
		throws HopsException, LopsException
	{
		WeightedCrossEntropy wcemm = new WeightedCrossEntropy(
				getInput().get(0).constructLops(),
				getInput().get(1).constructLops(),
				getInput().get(2).constructLops(),
				getDataType(), getValueType(), wtype, ExecType.CP);
		
		//set degree of parallelism
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		wcemm.setNumThreads(k);
		
		setOutputDimensions( wcemm );
		setLineNumbers( wcemm );
		setLops( wcemm );
	}
	
	/**
	 * 
	 * @param wtype 
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructMRLopsWeightedCeMM(WCeMMType wtype) 
		throws HopsException, LopsException
	{
		//NOTE: the common case for wsloss are factors U/V with a rank of 10s to 100s; the current runtime only
		//supports single block outer products (U/V rank <= blocksize, i.e., 1000 by default); we enforce this
		//by applying the hop rewrite for Weighted Squared Loss only if this constraint holds. 
		
		Hop X = getInput().get(0);
		Hop U = getInput().get(1);
		Hop V = getInput().get(2);
		
		//MR operator selection, part1
		double m1Size = OptimizerUtils.estimateSize(U.getDim1(), U.getDim2()); //size U
		double m2Size = OptimizerUtils.estimateSize(V.getDim1(), V.getDim2()); //size V
		boolean isMapWsloss = (m1Size+m2Size < OptimizerUtils.getRemoteMemBudgetMap(true)); 
		
		if( !FORCE_REPLICATION && isMapWsloss ) //broadcast
		{
			//partitioning of U
			boolean needPartU = !U.dimsKnown() || U.getDim1() * U.getDim2() > DistributedCacheInput.PARTITION_SIZE;
			Lop lU = U.constructLops();
			if( needPartU ){ //requires partitioning
				lU = new DataPartition(lU, DataType.MATRIX, ValueType.DOUBLE, (m1Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
				lU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), getRowsInBlock(), getColsInBlock(), U.getNnz());
				setLineNumbers(lU);	
			}
			
			//partitioning of V
			boolean needPartV = !V.dimsKnown() || V.getDim1() * V.getDim2() > DistributedCacheInput.PARTITION_SIZE;
			Lop lV = V.constructLops();
			if( needPartV ){ //requires partitioning
				lV = new DataPartition(lV, DataType.MATRIX, ValueType.DOUBLE, (m2Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
				lV.getOutputParameters().setDimensions(V.getDim1(), V.getDim2(), getRowsInBlock(), getColsInBlock(), V.getNnz());
				setLineNumbers(lV);	
			}
			
			//map-side wsloss always with broadcast
			Lop wcemm = new WeightedCrossEntropy( X.constructLops(), lU, lV, 
					DataType.MATRIX, ValueType.DOUBLE, wtype, ExecType.MR);
			wcemm.getOutputParameters().setDimensions(1, 1, X.getRowsInBlock(), X.getColsInBlock(), -1);
			setLineNumbers(wcemm);
			
			Group grp = new Group(wcemm, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
			grp.getOutputParameters().setDimensions(1, 1, X.getRowsInBlock(), X.getColsInBlock(), -1);
			setLineNumbers(grp);
			
			Aggregate agg1 = new Aggregate(grp, HopsAgg2Lops.get(AggOp.SUM), DataType.MATRIX, ValueType.DOUBLE, ExecType.MR);
			agg1.setupCorrectionLocation(CorrectionLocationType.NONE); // aggregation uses kahanSum 
			agg1.getOutputParameters().setDimensions(1, 1, X.getRowsInBlock(), X.getColsInBlock(), -1);
			setLineNumbers(agg1);
			
			UnaryCP unary1 = new UnaryCP(agg1, HopsOpOp1LopsUS.get(OpOp1.CAST_AS_SCALAR), getDataType(), getValueType());
			unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			setLineNumbers(unary1);
			setLops(unary1);
		}
		else //general case
		{
			//MR operator selection part 2
			boolean cacheU = !FORCE_REPLICATION && (m1Size < OptimizerUtils.getRemoteMemBudgetReduce());
			boolean cacheV = !FORCE_REPLICATION && ((!cacheU && m2Size < OptimizerUtils.getRemoteMemBudgetReduce()) 
					        || (cacheU && m1Size+m2Size < OptimizerUtils.getRemoteMemBudgetReduce()));
			
			Group grpX = new Group(X.constructLops(), Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
			grpX.getOutputParameters().setDimensions(X.getDim1(), X.getDim2(), X.getRowsInBlock(), X.getColsInBlock(), -1);
			setLineNumbers(grpX);
			
			Lop lU = null;
			if( cacheU ) {
				//partitioning of U for read through distributed cache
				boolean needPartU = !U.dimsKnown() || U.getDim1() * U.getDim2() > DistributedCacheInput.PARTITION_SIZE;
				lU = U.constructLops();
				if( needPartU ){ //requires partitioning
					lU = new DataPartition(lU, DataType.MATRIX, ValueType.DOUBLE, (m1Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
					lU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), getRowsInBlock(), getColsInBlock(), U.getNnz());
					setLineNumbers(lU);	
				}
			}
			else {
				//replication of U for shuffle to target block
				Lop offset = createOffsetLop(V, false); //ncol of t(V) -> nrow of V determines num replicates
				lU = new RepMat(U.constructLops(), offset, true, V.getDataType(), V.getValueType());
				lU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), 
						U.getRowsInBlock(), U.getColsInBlock(), U.getNnz());
				setLineNumbers(lU);
				
				Group grpU = new Group(lU, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
				grpU.getOutputParameters().setDimensions(U.getDim1(), U.getDim2(), U.getRowsInBlock(), U.getColsInBlock(), -1);
				setLineNumbers(grpU);
				lU = grpU;
			}
			
			Lop lV = null;
			if( cacheV ) {
				//partitioning of V for read through distributed cache
				boolean needPartV = !V.dimsKnown() || V.getDim1() * V.getDim2() > DistributedCacheInput.PARTITION_SIZE;
				lV = V.constructLops();
				if( needPartV ){ //requires partitioning
					lV = new DataPartition(lV, DataType.MATRIX, ValueType.DOUBLE, (m2Size>OptimizerUtils.getLocalMemBudget())?ExecType.MR:ExecType.CP, PDataPartitionFormat.ROW_BLOCK_WISE_N);
					lV.getOutputParameters().setDimensions(V.getDim1(), V.getDim2(), getRowsInBlock(), getColsInBlock(), V.getNnz());
					setLineNumbers(lV);	
				}
			}
			else {
				//replication of t(V) for shuffle to target block
				Transform ltV = new Transform( V.constructLops(), HopsTransf2Lops.get(ReOrgOp.TRANSPOSE), getDataType(), getValueType(), ExecType.MR);
				ltV.getOutputParameters().setDimensions(V.getDim2(), V.getDim1(), 
						V.getColsInBlock(), V.getRowsInBlock(), V.getNnz());
				setLineNumbers(ltV);
				
				Lop offset = createOffsetLop(U, false); //nrow of U determines num replicates
				lV = new RepMat(ltV, offset, false, V.getDataType(), V.getValueType());
				lV.getOutputParameters().setDimensions(V.getDim2(), V.getDim1(), 
						V.getColsInBlock(), V.getRowsInBlock(), V.getNnz());
				setLineNumbers(lV);
				
				Group grpV = new Group(lV, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
				grpV.getOutputParameters().setDimensions(V.getDim2(), V.getDim1(), V.getColsInBlock(), V.getRowsInBlock(), -1);
				setLineNumbers(grpV);
				lV = grpV;
			}
			
			//reduce-side wsloss w/ or without broadcast
			Lop wcemm = new WeightedCrossEntropyR( 
					grpX, lU, lV, DataType.MATRIX, ValueType.DOUBLE, wtype, cacheU, cacheV, ExecType.MR);
			wcemm.getOutputParameters().setDimensions(1, 1, X.getRowsInBlock(), X.getColsInBlock(), -1);
			setLineNumbers(wcemm);
			
			Group grp = new Group(wcemm, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE);
			grp.getOutputParameters().setDimensions(1, 1, X.getRowsInBlock(), X.getColsInBlock(), -1);
			setLineNumbers(grp);
			
			Aggregate agg1 = new Aggregate(grp, HopsAgg2Lops.get(AggOp.SUM), DataType.MATRIX, ValueType.DOUBLE, ExecType.MR);
			agg1.setupCorrectionLocation(CorrectionLocationType.NONE); // aggregation uses kahanSum 
			agg1.getOutputParameters().setDimensions(1, 1, X.getRowsInBlock(), X.getColsInBlock(), -1);
			setLineNumbers(agg1);
			
			UnaryCP unary1 = new UnaryCP(agg1, HopsOpOp1LopsUS.get(OpOp1.CAST_AS_SCALAR), getDataType(), getValueType());
			unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			setLineNumbers(unary1);
			setLops(unary1);
		}
	}
	
	/**
	 * 
	 * @param wtype
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructSparkLopsWeightedCeMM(WCeMMType wtype) 
		throws HopsException, LopsException
	{
		//NOTE: the common case for wsloss are factors U/V with a rank of 10s to 100s; the current runtime only
		//supports single block outer products (U/V rank <= blocksize, i.e., 1000 by default); we enforce this
		//by applying the hop rewrite for Weighted Squared Loss only if this constraint holds. 
		
		//Notes: Any broadcast needs to fit twice in local memory because we partition the input in cp,
		//and needs to fit once in executor broadcast memory. The 2GB broadcast constraint is no longer
		//required because the max_int byte buffer constraint has been fixed in Spark 1.4 
		double memBudgetExec = SparkExecutionContext.getBroadcastMemoryBudget();
		double memBudgetLocal = OptimizerUtils.getLocalMemBudget();

		Hop X = getInput().get(0);
		Hop U = getInput().get(1);
		Hop V = getInput().get(2);
		
		//MR operator selection, part1
		double m1Size = OptimizerUtils.estimateSize(U.getDim1(), U.getDim2()); //size U
		double m2Size = OptimizerUtils.estimateSize(V.getDim1(), V.getDim2()); //size V
		boolean isMapWsloss = (m1Size+m2Size < memBudgetExec
				&& 2*m1Size < memBudgetLocal && 2*m2Size < memBudgetLocal); 
		
		if( !FORCE_REPLICATION && isMapWsloss ) //broadcast
		{
			//map-side wsloss always with broadcast
			Lop wsloss = new WeightedCrossEntropy( X.constructLops(), U.constructLops(), V.constructLops(),  
					DataType.SCALAR, ValueType.DOUBLE, wtype, ExecType.SPARK);
			setOutputDimensions(wsloss);
			setLineNumbers(wsloss);
			setLops(wsloss);
		}
		else //general case
		{
			//MR operator selection part 2
			boolean cacheU = !FORCE_REPLICATION && (m1Size < memBudgetExec && 2*m1Size < memBudgetLocal);
			boolean cacheV = !FORCE_REPLICATION && ((!cacheU && m2Size < memBudgetExec ) 
					        || (cacheU && m1Size+m2Size < memBudgetExec)) && 2*m2Size < memBudgetLocal;
			
			//reduce-side wsloss w/ or without broadcast
			Lop wcemm = new WeightedCrossEntropyR( 
					X.constructLops(), U.constructLops(), V.constructLops(), 
					DataType.SCALAR, ValueType.DOUBLE, wtype, cacheU, cacheV, ExecType.SPARK);
			setOutputDimensions(wcemm);
			setLineNumbers(wcemm);
			setLops(wcemm);
		}
	}
	
	/**
	 * 
	 * @return
	 */
	private WeightsType checkWeightsType()
	{
		WeightsType ret = WeightsType.NONE;
		if( !(getInput().get(3) instanceof LiteralOp) ){
			if( _postWeights )
				ret = WeightsType.POST;
			else
				ret = WeightsType.PRE;
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	private WSigmoidType checkWSigmoidType()
	{
		
		if( _logout && _minusin )
			return WSigmoidType.LOG_MINUS;
		else if( _logout )
			return WSigmoidType.LOG;
		else if( _minusin )
			return WSigmoidType.MINUS;
		else
			return WSigmoidType.BASIC;
	}
	
	/**
	 * 
	 * @return
	 */
	private WDivMMType checkWDivMMType()
	{
		if( _left )
			return WDivMMType.LEFT;
		else
			return WDivMMType.RIGHT;
	}
	
	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{
		switch( _op ) {
			case WSLOSS: //always scalar output 
				return OptimizerUtils.DOUBLE_SIZE;
				
			case WSIGMOID: 
			case WDIVMM: 
				double sp = OptimizerUtils.getSparsity(dim1, dim2, nnz);
				return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sp);
				
			default:
				return 0;
		}
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz ) 
	{
		switch( _op ) {
			case WDIVMM: 
				int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
				if( _left && k>1 ){ //require entire output to prevent synchronization
					return OptimizerUtils.estimateSize(dim1, dim2);
				}
				else { //intermediate sparse row (dense as worst-case estimate)
					return Math.max(dim1, dim2) * OptimizerUtils.DOUBLE_SIZE;
				}
				
			default: //no intermediates 
				return 0;
		}
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		long[] ret = null;
		
		switch( _op ) {
			case WSLOSS: //always scalar output
				ret = null;
				break;
			
			case WSIGMOID: {
				MatrixCharacteristics mcW = memo.getAllInputStats(getInput().get(0));
				ret = new long[]{mcW.getRows(), mcW.getCols(), mcW.getNonZeros()};
				break;
			}
			
			case WDIVMM: {
				if( _left ) { //w/ transpose
					MatrixCharacteristics mcV = memo.getAllInputStats(getInput().get(2));
					ret = new long[]{mcV.getRows(), mcV.getCols(), -1};
				}
				else {
					MatrixCharacteristics mcU = memo.getAllInputStats(getInput().get(1));
					ret = new long[]{mcU.getRows(), mcU.getCols(), -1};
				}
				break;
			}
			
			default:
				throw new RuntimeException("Memory for operation (" + _op + ") can not be estimated.");
		}
				
		return ret;
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
			else if ( (getInput().get(0).areDimsBelowThreshold() 
					&& getInput().get(1).areDimsBelowThreshold()
					&& getInput().get(2).areDimsBelowThreshold()
					&& getInput().get(3).areDimsBelowThreshold()) )
				_etype = ExecType.CP;
			else
				_etype = REMOTE;
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}
		
		//mark for recompile (forever)
		if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==REMOTE )
			setRequiresRecompile();
	
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		switch( _op ) {
			case WSLOSS: 
				//do nothing: always scalar
				break;
				
			case WSIGMOID: {
				Hop inW = getInput().get(0);
				setDim1( inW.getDim1() );
				setDim2( inW.getDim2() );
				setNnz( inW.getNnz() );
				break;
			}
			
			case WDIVMM: {
				if( _left ){ //w/ transpose
					Hop inV = getInput().get(2);
					setDim1( inV.getDim1() );
					setDim2( inV.getDim2() );				
				}
				else {
					Hop inU = getInput().get(1);
					setDim1( inU.getDim1() );
					setDim2( inU.getDim2() );	
				}
				break;
			}
			
			default:
				break;
		}	
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		QuaternaryOp ret = new QuaternaryOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._op = _op;
		ret._postWeights = _postWeights;
		ret._logout = _logout;
		ret._minusin = _minusin;
		ret._left = _left;
		ret._maxNumThreads = _maxNumThreads;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof QuaternaryOp) )
			return false;
		
		QuaternaryOp that2 = (QuaternaryOp)that;
		
		//compare basic inputs and weights (always existing)
		boolean ret = (_op == that2._op
				&& getInput().size() == getInput().size()
				&& getInput().get(0) == that2.getInput().get(0)
				&& getInput().get(1) == that2.getInput().get(1)
				&& getInput().get(2) == that2.getInput().get(2) );
	
		//check for 4th argument if same size (see above)
		if( ret && getInput().size()==4 ) 
			ret &= (getInput().get(3) == that2.getInput().get(3));
		
		//compare specific parameters
		ret &= _postWeights == that2._postWeights;
		ret &= _logout      == that2._logout;
		ret &= _minusin 	== that2._minusin;
		ret &= _left        == that2._left;
		ret &= _maxNumThreads == that2._maxNumThreads;
		
		return ret;
	}
}
