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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOp4;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.lops.LopsException;
import org.apache.sysds.lops.WeightedCrossEntropy;
import org.apache.sysds.lops.WeightedCrossEntropy.WCeMMType;
import org.apache.sysds.lops.WeightedCrossEntropyR;
import org.apache.sysds.lops.WeightedDivMM;
import org.apache.sysds.lops.WeightedDivMM.WDivMMType;
import org.apache.sysds.lops.WeightedDivMMR;
import org.apache.sysds.lops.WeightedSigmoid;
import org.apache.sysds.lops.WeightedSigmoid.WSigmoidType;
import org.apache.sysds.lops.WeightedSigmoidR;
import org.apache.sysds.lops.WeightedSquaredLoss;
import org.apache.sysds.lops.WeightedSquaredLoss.WeightsType;
import org.apache.sysds.lops.WeightedSquaredLossR;
import org.apache.sysds.lops.WeightedUnaryMM;
import org.apache.sysds.lops.WeightedUnaryMM.WUMMType;
import org.apache.sysds.lops.WeightedUnaryMMR;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

/** 
 * Note: this hop should be called AggQuaternaryOp in consistency with AggUnaryOp and AggBinaryOp;
 * however, since there does not exist a real QuaternaryOp yet - we can leave it as is for now. 
 */
public class QuaternaryOp extends MultiThreadedHop
{
	//config influencing mr operator selection (for testing purposes only) 
	public static boolean FORCE_REPLICATION = false;
	
	private OpOp4 _op = null;
	
	//wsloss-specific attributes
	private boolean _postWeights = false;
	
	//wsigmoid-specific attributes
	private boolean _logout = false;
	private boolean _minusin = false;
	
	//wdivmm-specific attributes
	private int _baseType = -1;
	private boolean _mult = false;
	private boolean _minus = false;
	
	//wumm-specific attributes
	private boolean _umult = false;
	private OpOp1 _uop = null;
	private OpOp2 _sop = null;
	
	private QuaternaryOp() {
		//default constructor for clone
	}
	
	/**
	 * Constructor for wsloss.
	 * 
	 * @param l ?
	 * @param dt data type
	 * @param vt value type
	 * @param o the Hop.OpOp4
	 * @param inX high-level operator X
	 * @param inU high-level operator U
	 * @param inV high-level operator V
	 * @param inW high-level operator W
	 * @param post post weights
	 */
	public QuaternaryOp(String l, DataType dt, ValueType vt, OpOp4 o,
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
	 * @param l ?
	 * @param dt data type
	 * @param vt value type
	 * @param o the Hop.OpOp4
	 * @param inX high-level operator X
	 * @param inU high-level operator U
	 * @param inV high-level operator V
	 * @param flag1 logout
	 * @param flag2 minusin
	 */
	public QuaternaryOp(String l, DataType dt, ValueType vt, OpOp4 o,
		Hop inX, Hop inU, Hop inV, boolean flag1, boolean flag2) 
	{
		this(l, dt, vt, o, inX, inU, inV);
		_logout = flag1;
		_minusin = flag2;
	}

	public QuaternaryOp(String l, DataType dt, ValueType vt, OpOp4 o,
		Hop inX, Hop inU, Hop inV, Hop inW, int baseType, boolean flag1, boolean flag2) 
	{
		this(l, dt, vt, o, inX, inU, inV);
		if( inW != null ) { //four inputs
			getInput().add(3, inW);
			inW.getParent().add(this);
		}
		_baseType = baseType;
		_mult = flag1;
		_minus = flag2;
	}
	
	public QuaternaryOp(String l, DataType dt, ValueType vt, OpOp4 o,
		Hop inW, Hop inU, Hop inV, boolean umult, OpOp1 uop, OpOp2 sop) 
	{
		this(l, dt, vt, o, inW, inU, inV);
		_umult = umult;
		_uop = uop;
		_sop = sop;
	}

	public QuaternaryOp(String l, DataType dt, ValueType vt, OpOp4 o, Hop inX, Hop inU, Hop inV)  {
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
	public boolean isGPUEnabled() {
		return false;
	}
	
	@Override
	public boolean isMultiThreadedOpType() {
		return true;
	}
	
	@Override
	public Lop constructLops() 
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
					else if( et == ExecType.SPARK )
						constructSparkLopsWeightedDivMM(wtype);
					else
						throw new HopsException("Unsupported quaternaryop-wdivmm exec type: "+et);
					break;
				}
				
				case WCEMM:{
					WCeMMType wtype = checkWCeMMType();
					
					if( et == ExecType.CP )
						constructCPLopsWeightedCeMM(wtype);
					else if( et == ExecType.SPARK )
						constructSparkLopsWeightedCeMM(wtype);
					else
						throw new HopsException("Unsupported quaternaryop-wcemm exec type: "+et);
					break;
				}
				
				case WUMM:{
					WUMMType wtype = _umult ? WUMMType.MULT : WUMMType.DIV;
					
					if( et == ExecType.CP )
						constructCPLopsWeightedUMM(wtype);
					else if( et == ExecType.SPARK )
						constructSparkLopsWeightedUMM(wtype);
					else
						throw new HopsException("Unsupported quaternaryop-wumm exec type: "+et);
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
		return "q(" + _op.toString() + ")";
	}

	@Override
	public boolean allowsAllExecTypes() {
		return true;
	}

	private void constructCPLopsWeightedSquaredLoss(WeightsType wtype) 
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

	private void constructSparkLopsWeightedSquaredLoss(WeightsType wtype) 
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
		boolean isMapWsloss = (!wtype.hasFourInputs() && m1Size+m2Size < memBudgetExec
			&& 2*m1Size < memBudgetLocal && 2*m2Size < memBudgetLocal); 
		
		if( !FORCE_REPLICATION && isMapWsloss ) //broadcast
		{
			//map-side wsloss always with broadcast
			Lop wsloss = new WeightedSquaredLoss( X.constructLops(), U.constructLops(), V.constructLops(),
				W.constructLops(), DataType.SCALAR, ValueType.FP64, wtype, ExecType.SPARK);
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
				DataType.SCALAR, ValueType.FP64, wtype, cacheU, cacheV, ExecType.SPARK);
			setOutputDimensions(wsloss);
			setLineNumbers(wsloss);
			setLops(wsloss);
		}
	}

	private void constructCPLopsWeightedSigmoid(WSigmoidType wtype) {
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

	private void constructSparkLopsWeightedSigmoid( WSigmoidType wtype ) 
	{
		//NOTE: the common case for wsigmoid are factors U/V with a rank of 10s to 100s; the current runtime only
		//supports single block outer products (U/V rank <= blocksize, i.e., 1000 by default); we enforce this
		//by applying the hop rewrite for Weighted Sigmoid only if this constraint holds. 

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
		boolean isMapWsig = (m1Size+m2Size < memBudgetExec
				&& 2*m1Size<memBudgetLocal && 2*m2Size<memBudgetLocal); 
		
		if( !FORCE_REPLICATION && isMapWsig ) //broadcast
		{
			//map-side wsig always with broadcast
			Lop wsigmoid = new WeightedSigmoid( X.constructLops(), U.constructLops(), V.constructLops(),  
					DataType.MATRIX, ValueType.FP64, wtype, ExecType.SPARK);
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
			
			//reduce-side wsig w/ or without broadcast
			Lop wsigmoid = new WeightedSigmoidR( 
				X.constructLops(), U.constructLops(), V.constructLops(), 
				DataType.MATRIX, ValueType.FP64, wtype, cacheU, cacheV, ExecType.SPARK);
			setOutputDimensions(wsigmoid);
			setLineNumbers(wsigmoid);
			setLops(wsigmoid);
		}
	}

	private void constructCPLopsWeightedDivMM(WDivMMType wtype) 
	{
		WeightedDivMM wdiv = new WeightedDivMM(
			getInput().get(0).constructLops(),
			getInput().get(1).constructLops(),
			getInput().get(2).constructLops(),
			getInput().get(3).constructLops(),
			getDataType(), getValueType(), wtype, ExecType.CP);
		
		//set degree of parallelism
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		wdiv.setNumThreads(k);
		
		setOutputDimensions( wdiv );
		setLineNumbers( wdiv );
		setLops( wdiv );
	}

	private void constructSparkLopsWeightedDivMM( WDivMMType wtype ) 
	{
		//NOTE: the common case for wdivmm are factors U/V with a rank of 10s to 100s; the current runtime only
		//supports single block outer products (U/V rank <= blocksize, i.e., 1000 by default); we enforce this
		//by applying the hop rewrite for Weighted DivMM only if this constraint holds. 

		//Notes: Any broadcast needs to fit twice in local memory because we partition the input in cp,
		//and needs to fit once in executor broadcast memory. The 2GB broadcast constraint is no longer
		//required because the max_int byte buffer constraint has been fixed in Spark 1.4 
		double memBudgetExec = SparkExecutionContext.getBroadcastMemoryBudget();
		double memBudgetLocal = OptimizerUtils.getLocalMemBudget();

		Hop W = getInput().get(0);
		Hop U = getInput().get(1);
		Hop V = getInput().get(2);
		Hop X = getInput().get(3);
		
		//MR operator selection, part1
		double m1Size = OptimizerUtils.estimateSize(U.getDim1(), U.getDim2()); //size U
		double m2Size = OptimizerUtils.estimateSize(V.getDim1(), V.getDim2()); //size V
		boolean isMapWdivmm = ((!wtype.hasFourInputs() || wtype.hasScalar()) && m1Size+m2Size < memBudgetExec
				&& 2*m1Size<memBudgetLocal && 2*m2Size<memBudgetLocal); 
		
		if( !FORCE_REPLICATION && isMapWdivmm ) //broadcast
		{
			//map-side wdivmm always with broadcast
			Lop wdivmm = new WeightedDivMM( W.constructLops(), U.constructLops(), V.constructLops(), 
					X.constructLops(), DataType.MATRIX, ValueType.FP64, wtype, ExecType.SPARK);
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
			
			//reduce-side wdivmm w/ or without broadcast
			Lop wdivmm = new WeightedDivMMR( 
				W.constructLops(), U.constructLops(), V.constructLops(), X.constructLops(),
				DataType.MATRIX, ValueType.FP64, wtype, cacheU, cacheV, ExecType.SPARK);
			setOutputDimensions(wdivmm);
			setLineNumbers(wdivmm);
			setLops(wdivmm);
		}
	}

	private void constructCPLopsWeightedCeMM(WCeMMType wtype) 
	{
		WeightedCrossEntropy wcemm = new WeightedCrossEntropy(
			getInput().get(0).constructLops(),
			getInput().get(1).constructLops(),
			getInput().get(2).constructLops(),
			getInput().get(3).constructLops(),
			getDataType(), getValueType(), wtype, ExecType.CP);
		
		//set degree of parallelism
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		wcemm.setNumThreads(k);
		
		setOutputDimensions( wcemm );
		setLineNumbers( wcemm );
		setLops( wcemm );
	}

	private void constructSparkLopsWeightedCeMM(WCeMMType wtype) 
	{
		//NOTE: the common case for wcemm are factors U/V with a rank of 10s to 100s; the current runtime only
		//supports single block outer products (U/V rank <= blocksize, i.e., 1000 by default); we enforce this
		//by applying the hop rewrite for Weighted Cross Entropy only if this constraint holds. 
		
		//Notes: Any broadcast needs to fit twice in local memory because we partition the input in cp,
		//and needs to fit once in executor broadcast memory. The 2GB broadcast constraint is no longer
		//required because the max_int byte buffer constraint has been fixed in Spark 1.4 
		double memBudgetExec = SparkExecutionContext.getBroadcastMemoryBudget();
		double memBudgetLocal = OptimizerUtils.getLocalMemBudget();

		Hop X = getInput().get(0);
		Hop U = getInput().get(1);
		Hop V = getInput().get(2);
		Hop eps = getInput().get(3);
		
		//MR operator selection, part1
		double m1Size = OptimizerUtils.estimateSize(U.getDim1(), U.getDim2()); //size U
		double m2Size = OptimizerUtils.estimateSize(V.getDim1(), V.getDim2()); //size V
		boolean isMapWcemm = (m1Size+m2Size < memBudgetExec
				&& 2*m1Size < memBudgetLocal && 2*m2Size < memBudgetLocal); 
		
		if( !FORCE_REPLICATION && isMapWcemm ) //broadcast
		{
			//map-side wcemm always with broadcast
			Lop wcemm = new WeightedCrossEntropy( X.constructLops(), U.constructLops(), V.constructLops(),
				eps.constructLops(), DataType.SCALAR, ValueType.FP64, wtype, ExecType.SPARK);
			setOutputDimensions(wcemm);
			setLineNumbers(wcemm);
			setLops(wcemm);
		}
		else //general case
		{
			//MR operator selection part 2
			boolean cacheU = !FORCE_REPLICATION && (m1Size < memBudgetExec && 2*m1Size < memBudgetLocal);
			boolean cacheV = !FORCE_REPLICATION && ((!cacheU && m2Size < memBudgetExec ) 
					        || (cacheU && m1Size+m2Size < memBudgetExec)) && 2*m2Size < memBudgetLocal;
			
			//reduce-side wcemm w/ or without broadcast
			Lop wcemm = new WeightedCrossEntropyR( 
				X.constructLops(), U.constructLops(), V.constructLops(), eps.constructLops(),
				DataType.SCALAR, ValueType.FP64, wtype, cacheU, cacheV, ExecType.SPARK);
			setOutputDimensions(wcemm);
			setLineNumbers(wcemm);
			setLops(wcemm);
		}
	}

	private void constructCPLopsWeightedUMM(WUMMType wtype) {
		OpOp1 uop = _uop!=null ? _uop : _sop==OpOp2.POW ? 
			OpOp1.POW2 : OpOp1.MULT2;
		
		WeightedUnaryMM wumm = new WeightedUnaryMM(
			getInput().get(0).constructLops(),
			getInput().get(1).constructLops(),
			getInput().get(2).constructLops(),
			getDataType(), getValueType(), wtype, uop, ExecType.CP);
		
		//set degree of parallelism
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		wumm.setNumThreads(k);
		
		setOutputDimensions( wumm );
		setLineNumbers( wumm );
		setLops( wumm );
	}

	private void constructSparkLopsWeightedUMM( WUMMType wtype ) 
	{
		//NOTE: the common case for wumm are factors U/V with a rank of 10s to 100s; the current runtime only
		//supports single block outer products (U/V rank <= blocksize, i.e., 1000 by default); we enforce this
		//by applying the hop rewrite for Weighted UnaryMM only if this constraint holds. 

		OpOp1 uop = _uop!=null ? _uop : _sop==OpOp2.POW ? 
			OpOp1.POW2 : OpOp1.MULT2;
		
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
			//map-side wumm always with broadcast
			Lop wumm = new WeightedUnaryMM( X.constructLops(), U.constructLops(),
				V.constructLops(), DataType.MATRIX, ValueType.FP64, wtype, uop, ExecType.SPARK);
			setOutputDimensions(wumm);
			setLineNumbers(wumm);
			setLops( wumm );
		}
		else //general case
		{
			//MR operator selection part 2
			boolean cacheU = !FORCE_REPLICATION && (m1Size < memBudgetExec && 2*m1Size < memBudgetLocal);
			boolean cacheV = !FORCE_REPLICATION && ((!cacheU && m2Size < memBudgetExec ) 
					        || (cacheU && m1Size+m2Size < memBudgetExec)) && 2*m2Size < memBudgetLocal;
			
			//reduce-side wumm w/ or without broadcast
			Lop wumm = new WeightedUnaryMMR( 
					X.constructLops(), U.constructLops(), V.constructLops(), 
					DataType.MATRIX, ValueType.FP64, wtype, uop, cacheU, cacheV, ExecType.SPARK);
			setOutputDimensions(wumm);
			setLineNumbers(wumm);
			setLops(wumm);
		}
	}

	private WeightsType checkWeightsType()
	{
		WeightsType ret = WeightsType.NONE;
		if( !(getInput().get(3) instanceof LiteralOp) ){
			if( _postWeights )
				ret = WeightsType.POST;
			else
				ret = WeightsType.PRE;
		}
		else if( _postWeights ){
			ret = WeightsType.POST_NZ;
		}
		
		return ret;
	}

	private WSigmoidType checkWSigmoidType() {
		if( _logout && _minusin )
			return WSigmoidType.LOG_MINUS;
		else if( _logout )
			return WSigmoidType.LOG;
		else if( _minusin )
			return WSigmoidType.MINUS;
		else
			return WSigmoidType.BASIC;
	}

	private WDivMMType checkWDivMMType()
	{
		switch( _baseType )
		{
			case 0: //BASIC
				return WDivMMType.MULT_BASIC;
			case 1: //LEFT
				if( getInput().get(3).getDataType()==DataType.MATRIX )
					return WDivMMType.MULT_MINUS_4_LEFT;
				else if( _minus )
					return WDivMMType.MULT_MINUS_LEFT;
				else
					return _mult ? WDivMMType.MULT_LEFT : WDivMMType.DIV_LEFT;
			case 2: //RIGHT	
				if( getInput().get(3).getDataType()==DataType.MATRIX )
					return WDivMMType.MULT_MINUS_4_RIGHT;
				else if( _minus )
					return WDivMMType.MULT_MINUS_RIGHT;
				else
					return _mult ? WDivMMType.MULT_RIGHT : WDivMMType.DIV_RIGHT;
			case 3: //LEFT w/EPS
				return WDivMMType.DIV_LEFT_EPS;
			case 4: //RIGHT w/EPS
				return WDivMMType.DIV_RIGHT_EPS;
		}
		
		return null;
	}

	private WCeMMType checkWCeMMType() {
		return _baseType == 1 ? WCeMMType.BASIC_EPS : WCeMMType.BASIC;
	}
	
	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{
		switch( _op ) {
			case WSLOSS: //always scalar output 
			case WCEMM:	
				return OptimizerUtils.DOUBLE_SIZE;
				
			case WSIGMOID: 
			case WDIVMM: 
			case WUMM:	
				double sp = OptimizerUtils.getSparsity(dim1, dim2, nnz);
				return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sp);
				
			default:
				return 0;
		}
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz ) {
		//no intermediates 
		return 0;
	}
	
	@Override
	protected DataCharacteristics inferOutputCharacteristics( MemoTable memo )
	{
		DataCharacteristics ret = null;
		
		switch( _op ) {
			case WSLOSS: //always scalar output
				break; //null
			
			case WSIGMOID: 
			case WUMM: {
				DataCharacteristics mcW = memo.getAllInputStats(getInput().get(0));
				ret = new MatrixCharacteristics(mcW.getRows(), mcW.getCols(), -1, mcW.getNonZeros());
				break;
			}
			
			case WDIVMM: {
				if( _baseType == 0 ){ //basic
					ret = memo.getAllInputStats(getInput().get(0));
				}
				else if( _baseType == 1 || _baseType == 3 ) { //left (w/ transpose or w/ epsilon)
					DataCharacteristics mcV = memo.getAllInputStats(getInput().get(2));
					ret = mcV.setNonZeros(-1);
				}
				else { //right
					DataCharacteristics mcU = memo.getAllInputStats(getInput().get(1));
					ret = mcU.setNonZeros(-1);
				}
				break;
			}
			
			default:
				throw new RuntimeException("Memory for operation (" + _op + ") can not be estimated.");
		}
		
		return ret;
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
			else if ( (getInput().get(0).areDimsBelowThreshold() 
					&& getInput().get(1).areDimsBelowThreshold()
					&& getInput().get(2).areDimsBelowThreshold()
					&& getInput().get(3).areDimsBelowThreshold()) )
				_etype = ExecType.CP;
			else
				_etype = ExecType.SPARK;
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}
		
		//mark for recompile (forever)
		setRequiresRecompileIfNecessary();
		
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		switch( _op ) {
			case WSLOSS: 
				//do nothing: always scalar
				break;
				
			case WSIGMOID:
			case WUMM: {
				Hop inW = getInput().get(0);
				setDim1( inW.getDim1() );
				setDim2( inW.getDim2() );
				setNnz( inW.getNnz() );
				break;
			}
			
			case WDIVMM: {
				if( _baseType == 0 ) { //basic
					Hop inW = getInput().get(0);
					setDim1( inW.getDim1() );
					setDim2( inW.getDim2() );
					setNnz( inW.getNnz() );
				}
				else if( _baseType == 1 || _baseType == 3 ){ //left (w/ transpose or w/ epsilon)
					Hop inV = getInput().get(2);
					setDim1( inV.getDim1() );
					setDim2( inV.getDim2() );
					setNnz( -1 ); //reset
				}
				else { //right
					Hop inU = getInput().get(1);
					setDim1( inU.getDim1() );
					setDim2( inU.getDim2() );
					setNnz( -1 ); //reset
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
		ret._op            = _op;
		ret._postWeights   = _postWeights;
		ret._logout        = _logout;
		ret._minusin       = _minusin;
		ret._baseType      = _baseType;
		ret._mult          = _mult;
		ret._minus         = _minus;
		ret._umult         = _umult;
		ret._uop           = _uop;
		ret._sop           = _sop;		
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
				&& getInput().size() == that2.getInput().size()
				&& getInput().get(0) == that2.getInput().get(0)
				&& getInput().get(1) == that2.getInput().get(1)
				&& getInput().get(2) == that2.getInput().get(2) );
	
		//check for 4th argument if same size (see above)
		if( ret && getInput().size()==4 ) 
			ret &= (getInput().get(3) == that2.getInput().get(3));
		
		//compare specific parameters
		ret &= _postWeights   == that2._postWeights;
		ret &= _logout        == that2._logout;
		ret &= _minusin 	  == that2._minusin;
		ret &= _baseType      == that2._baseType;
		ret &= _mult          == that2._mult;
		ret &= _minus         == that2._minus;
		ret &= _umult         == that2._umult;
		ret &= _uop           == that2._uop;
		ret &= _sop           == that2._sop;
		ret &= _maxNumThreads == that2._maxNumThreads;
		
		return ret;
	}
}
