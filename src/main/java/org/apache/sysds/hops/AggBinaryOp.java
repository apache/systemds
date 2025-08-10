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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.MMCJ;
import org.apache.sysds.lops.MMRJ;
import org.apache.sysds.lops.MMTSJ;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.lops.MMZip;
import org.apache.sysds.lops.MapMult;
import org.apache.sysds.lops.MapMultChain;
import org.apache.sysds.lops.MapMultChain.ChainType;
import org.apache.sysds.lops.MatMultCP;
import org.apache.sysds.lops.PMMJ;
import org.apache.sysds.lops.PMapMult;
import org.apache.sysds.lops.Transform;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction.FederatedOutput;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.UtilFunctions;


/* Aggregate binary (cell operations): Sum (aij + bij)
 * 		Properties: 
 * 			Inner Symbol: *, -, +, ...
 * 			Outer Symbol: +, min, max, ...
 * 			2 Operands
 * 	
 * 		Semantic: generate indices, align, cross-operate, generate indices, align, aggregate
 */
public class AggBinaryOp extends MultiThreadedHop {
	// private static final Log LOG =  LogFactory.getLog(BinaryOp.class.getName());

	public static final double MAPMULT_MEM_MULTIPLIER = 1.0;
	public static MMultMethod FORCED_MMULT_METHOD = null;

	public enum MMultMethod {
		CPMM,     //cross-product matrix multiplication (mr)
		RMM,      //replication matrix multiplication (mr)
		MAPMM_L,  //map-side matrix-matrix multiplication using distributed cache (mr/sp)
		MAPMM_R,  //map-side matrix-matrix multiplication using distributed cache (mr/sp)
		MAPMM_CHAIN, //map-side matrix-matrix-matrix multiplication using distributed cache, for right input (cp/mr/sp)
		PMAPMM,   //partitioned map-side matrix-matrix multiplication (sp)
		PMM,      //permutation matrix multiplication using distributed cache, for left input (mr/cp)
		TSMM,     //transpose-self matrix multiplication (cp/mr/sp)
		TSMM2,    //transpose-self matrix multiplication, 2-pass w/o shuffle (sp)
		ZIPMM,    //zip matrix multiplication (sp)
		MM        //in-memory matrix multiplication (cp)
	}

	public enum SparkAggType {
		NONE,
		SINGLE_BLOCK,
		MULTI_BLOCK,
	}

	private OpOp2 innerOp;
	private AggOp outerOp;

	private MMultMethod _method = null;

	//hints set by previous to operator selection
	private boolean _hasLeftPMInput = false; //left input is permutation matrix

	private AggBinaryOp() {
		//default constructor for clone
	}

	public AggBinaryOp(String l, DataType dt, ValueType vt, OpOp2 innOp,
					   AggOp outOp, Hop in1, Hop in2) {
		super(l, dt, vt);
		innerOp = innOp;
		outerOp = outOp;
		getInput().add(0, in1);
		getInput().add(1, in2);
		in1.getParent().add(this);
		in2.getParent().add(this);

		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	public void setHasLeftPMInput(boolean flag) {
		_hasLeftPMInput = flag;
	}

	public boolean hasLeftPMInput() {
		return _hasLeftPMInput;
	}

	public MMultMethod getMMultMethod() {
		return _method;
	}

	@Override
	public boolean isGPUEnabled() {
		if (!DMLScript.USE_ACCELERATOR)
			return false;

		Hop input1 = getInput().get(0);
		Hop input2 = getInput().get(1);
		//matrix mult operation selection part 2 (specific pattern)
		MMTSJType mmtsj = checkTransposeSelf(); //determine tsmm pattern
		ChainType chain = checkMapMultChain(); //determine mmchain pattern

		_method = optFindMMultMethodCP(input1.getDim1(), input1.getDim2(),
				input2.getDim1(), input2.getDim2(), mmtsj, chain, _hasLeftPMInput);
		switch (_method) {
			case TSMM:
				//return false; // TODO: Disabling any fused transa optimization in 1.0 release.
				return true;
			case MAPMM_CHAIN:
				return false;
			case PMM:
				return false;
			case MM:
				return true;
			default:
				throw new RuntimeException("Unsupported method:" + _method);
		}
	}

	/**
	 * NOTE: overestimated mem in case of transpose-identity matmult, but 3/2 at worst
	 * and existing mem estimate advantageous in terms of consistency hops/lops,
	 * and some special cases internally materialize the transpose for better cache locality
	 */
	@Override
	public Lop constructLops() {
		//return already created lops
		if (getLops() != null)
			return getLops();

		//construct matrix mult lops (currently only supported aggbinary)
		if (isMatrixMultiply()) {
			Hop input1 = getInput().get(0);
			Hop input2 = getInput().get(1);

			//matrix mult operation selection part 1 (CP vs MR vs Spark)
			ExecType et = optFindExecType();

			//matrix mult operation selection part 2 (specific pattern)
			MMTSJType mmtsj = checkTransposeSelf(); //determine tsmm pattern
			ChainType chain = checkMapMultChain(); //determine mmchain pattern

			if (mmtsj == MMTSJType.LEFT && input2.isCompressedOutput()) {
				// if tsmm and input is compressed. (using input2, since input1 is transposed and therefore not compressed.)
				et = ExecType.CP;
			}

			if (et == ExecType.CP || et == ExecType.GPU || et == ExecType.FED) {
				//matrix mult operation selection part 3 (CP type)
				_method = optFindMMultMethodCP(input1.getDim1(), input1.getDim2(),
						input2.getDim1(), input2.getDim2(), mmtsj, chain, _hasLeftPMInput);

				//dispatch CP lops construction 
				switch (_method) {
					case TSMM:
						constructCPLopsTSMM(mmtsj, et);
						break;
					case MAPMM_CHAIN:
						constructCPLopsMMChain(chain);
						break;
					case PMM:
						constructCPLopsPMM();
						break;
					case MM:
						constructCPLopsMM(et);
						break;
					default:
						throw new HopsException(this.printErrorLocation() + "Invalid Matrix Mult Method (" + _method + ") while constructing CP lops.");
				}
			} else if (et == ExecType.SPARK) {
				//matrix mult operation selection part 3 (SPARK type)
				boolean tmmRewrite = HopRewriteUtils.isTransposeOperation(input1);
				_method = optFindMMultMethodSpark(
						input1.getDim1(), input1.getDim2(), input1.getBlocksize(), input1.getNnz(),
						input2.getDim1(), input2.getDim2(), input2.getBlocksize(), input2.getNnz(),
						mmtsj, chain, _hasLeftPMInput, tmmRewrite);
				//dispatch SPARK lops construction
				switch (_method) {
					case TSMM:
					case TSMM2:
						constructSparkLopsTSMM(mmtsj, _method == MMultMethod.TSMM2);
						break;
					case MAPMM_L:
					case MAPMM_R:
						constructSparkLopsMapMM(_method);
						break;
					case MAPMM_CHAIN:
						constructSparkLopsMapMMChain(chain);
						break;
					case PMAPMM:
						constructSparkLopsPMapMM();
						break;
					case CPMM:
						constructSparkLopsCPMM();
						break;
					case RMM:
						constructSparkLopsRMM();
						break;
					case PMM:
						constructSparkLopsPMM();
						break;
					case ZIPMM:
						constructSparkLopsZIPMM();
						break;

					default:
						throw new HopsException(this.printErrorLocation() + "Invalid Matrix Mult Method (" + _method + ") while constructing SPARK lops.");
				}
			} else if (et == ExecType.OOC) {
				Lop in1 = getInput().get(0).constructLops();
				Lop in2 = getInput().get(1).constructLops();
				MatMultCP matmult = new MatMultCP(in1, in2, getDataType(), getValueType(),
					et, OptimizerUtils.getConstrainedNumThreads(_maxNumThreads));
				setOutputDimensions(matmult);
				setLineNumbers(matmult);
				setLops(matmult);
			}
		} else
			throw new HopsException(this.printErrorLocation() + "Invalid operation in AggBinary Hop, aggBin(" + innerOp + "," + outerOp + ") while constructing lops.");

		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();

		return getLops();
	}

	@Override
	public String getOpString() {
		//ba - binary aggregate, for consistency with runtime 
		return "ba(" + outerOp.toString() + innerOp.toString() + ")";
	}

	@Override
	public void computeMemEstimate(MemoTable memo) {
		//extension of default compute memory estimate in order to 
		//account for smaller tsmm memory requirements.
		super.computeMemEstimate(memo);

		//tsmm left is guaranteed to require only X but not t(X), while
		//tsmm right might have additional requirements to transpose X if sparse
		//NOTE: as a heuristic this correction is only applied if not a column vector because
		//most other vector operations require memory for at least two vectors (we aim for 
		//consistency in order to prevent anomalies in parfor opt leading to small degree of par)
		MMTSJType mmtsj = checkTransposeSelf();
		if (mmtsj.isLeft() && getInput().get(1).dimsKnown() && getInput().get(1).getDim2() > 1) {
			_memEstimate = _memEstimate - getInput().get(0)._outputMemEstimate;
		}
	}

	@Override
	protected double computeOutputMemEstimate(long dim1, long dim2, long nnz) {
		//NOTES:  
		// * The estimate for transpose-self is the same as for normal matrix multiplications
		//   because (1) this decouples the decision of TSMM over default MM and (2) some cases
		//   of TSMM internally materialize the transpose for efficiency.
		// * All matrix multiplications internally use dense output representations for efficiency.
		//   This is reflected in our conservative memory estimate. However, we additionally need 
		//   to account for potential final dense/sparse transformations via processing mem estimates.
		double sparsity = (nnz == 0) ? 0 : 1;
		double ret;
		/*
		if( isMatrixMultiply() ) {	
			if( nnz < 0 ){
				Hops input1 = getInput().get(0);
				Hops input2 = getInput().get(1);
				if( input1.dimsKnown() && input2.dimsKnown() )
				{
					double sp1 = (input1.getNnz()>0) ? OptimizerUtils.getSparsity(input1.getDim1(), input1.getDim2(), input1.getNnz()) : 1.0;
					double sp2 = (input2.getNnz()>0) ? OptimizerUtils.getSparsity(input2.getDim1(), input2.getDim2(), input2.getNnz()) : 1.0;
					sparsity = OptimizerUtils.getMatMultSparsity(sp1, sp2, input1.getDim1(), input1.getDim2(), input2.getDim2(), true);	
				}
			}
			else //sparsity known (e.g., inferred from worst case estimates)
				sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		}
		*/
		// estimated as dense in order to account for dense intermediate without unnecessary overestimation
		ret = OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);

		return ret;
	}

	@Override
	protected double computeIntermediateMemEstimate(long dim1, long dim2, long nnz) {
		double ret = 0;

		if (isGPUEnabled()) {
			Hop in1 = _input.get(0);
			Hop in2 = _input.get(1);
			double in1Sparsity = OptimizerUtils.getSparsity(in1.getDim1(), in1.getDim2(), in1.getNnz());
			double in2Sparsity = OptimizerUtils.getSparsity(in2.getDim1(), in2.getDim2(), in2.getNnz());
			boolean in1Sparse = in1Sparsity < MatrixBlock.SPARSITY_TURN_POINT;
			boolean in2Sparse = in2Sparsity < MatrixBlock.SPARSITY_TURN_POINT;
			if (in1Sparse && !in2Sparse) {
				// Only in sparse-dense cases, we need additional memory budget for GPU
				ret += OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, 1.0);
			}
		}

		//account for potential final dense-sparse transformation (worst-case sparse representation)
		if (dim2 >= 2 && nnz != 0) //vectors always dense
			ret += MatrixBlock.estimateSizeSparseInMemory(dim1, dim2,
					MatrixBlock.SPARSITY_TURN_POINT - UtilFunctions.DOUBLE_EPS);

		return ret;
	}

	@Override
	protected DataCharacteristics inferOutputCharacteristics(MemoTable memo) {
		DataCharacteristics[] dc = memo.getAllInputStats(getInput());
		DataCharacteristics ret = null;
		if (dc[0].rowsKnown() && dc[1].colsKnown()) {
			ret = new MatrixCharacteristics(dc[0].getRows(), dc[1].getCols());
			double sp1 = (dc[0].getNonZeros() > 0) ? OptimizerUtils.getSparsity(dc[0].getRows(), dc[0].getCols(), dc[0].getNonZeros()) : 1.0;
			double sp2 = (dc[1].getNonZeros() > 0) ? OptimizerUtils.getSparsity(dc[1].getRows(), dc[1].getCols(), dc[1].getNonZeros()) : 1.0;
			ret.setNonZeros((long) (ret.getLength() * OptimizerUtils.getMatMultSparsity(sp1, sp2, ret.getRows(), dc[0].getCols(), ret.getCols(), true)));
		}
		return ret;
	}


	public boolean isMatrixMultiply() {
		return (this.innerOp == OpOp2.MULT && this.outerOp == AggOp.SUM);
	}

	private boolean isOuterProduct() {
		return (getInput().get(0).isVector() && getInput().get(1).isVector())
				&& (getInput().get(0).getDim1() == 1 && getInput().get(0).getDim1() > 1
				&& getInput().get(1).getDim1() > 1 && getInput().get(1).getDim2() == 1);
	}

	@Override
	public boolean isMultiThreadedOpType() {
		return isMatrixMultiply();
	}

	@Override
	public boolean allowsAllExecTypes() {
		return true;
	}

	@Override
	protected ExecType optFindExecType(boolean transitive) {
		checkAndSetForcedPlatform();

		if (_etypeForced != null) {
			setExecType(_etypeForced);
		} else {
			if (OptimizerUtils.isMemoryBasedOptLevel()) {
				setExecType(findExecTypeByMemEstimate());
			}
			// choose CP if the dimensions of both inputs are below Hops.CPThreshold 
			// OR if it is vector-vector inner product
			else if ((getInput().get(0).areDimsBelowThreshold() && getInput().get(1).areDimsBelowThreshold())
					|| (getInput().get(0).isVector() && getInput().get(1).isVector() && !isOuterProduct())) {
				setExecType(ExecType.CP);
			} else {
				setExecType(ExecType.SPARK);
			}

			//check for valid CP mmchain, send invalid memory requirements to remote
			if (_etype == ExecType.CP
					&& checkMapMultChain() != ChainType.NONE
					&& OptimizerUtils.getLocalMemBudget() <
					getInput().get(0).getInput().get(0).getOutputMemEstimate()) {
				setExecType(ExecType.SPARK);
			}

			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}

		//spark-specific decision refinement (execute binary aggregate w/ left or right spark input and 
		//single parent also in spark because it's likely cheap and reduces data transfer)
		MMTSJType mmtsj = checkTransposeSelf(); //determine tsmm pattern
		if (transitive && _etype == ExecType.CP && _etypeForced != ExecType.CP
				&& ((!mmtsj.isLeft() && isApplicableForTransitiveSparkExecType(true))
				|| (!mmtsj.isRight() && isApplicableForTransitiveSparkExecType(false)))) {
			//pull binary aggregate into spark 
			setExecType(ExecType.SPARK);
		}

		//mark for recompile (forever)
		setRequiresRecompileIfNecessary();

		return _etype;
	}

	private boolean isApplicableForTransitiveSparkExecType(boolean left) {
		int index = left ? 0 : 1;
		return !(getInput(index) instanceof DataOp && ((DataOp) getInput(index)).requiresCheckpoint())
				&& (!HopRewriteUtils.isTransposeOperation(getInput(index))
				|| (left && !isLeftTransposeRewriteApplicable(true)))
				&& getInput(index).getParent().size() == 1 //bagg is only parent
				&& !getInput(index).areDimsBelowThreshold()
				&& (getInput(index).optFindExecType() == ExecType.SPARK
				|| (getInput(index) instanceof DataOp && ((DataOp) getInput(index)).hasOnlyRDD()))
				&& getInput(index).getOutputMemEstimate() > getOutputMemEstimate();
	}

	/**
	 * TSMM: Determine if XtX pattern applies for this aggbinary and if yes
	 * which type.
	 *
	 * @return MMTSJType
	 */
	public MMTSJType checkTransposeSelf() {
		MMTSJType ret = MMTSJType.NONE;

		Hop in1 = getInput().get(0);
		Hop in2 = getInput().get(1);

		if (HopRewriteUtils.isTransposeOperation(in1)
				&& in1.getInput().get(0) == in2) {
			ret = MMTSJType.LEFT;
		}

		if (HopRewriteUtils.isTransposeOperation(in2)
				&& in2.getInput().get(0) == in1) {
			ret = MMTSJType.RIGHT;
		}

		return ret;
	}

	/**
	 * MapMultChain: Determine if XtwXv/XtXv pattern applies for this aggbinary
	 * and if yes which type.
	 *
	 * @return ChainType
	 */
	public ChainType checkMapMultChain() {
		ChainType chainType = ChainType.NONE;

		Hop in1 = getInput().get(0);
		Hop in2 = getInput().get(1);

		//check for transpose left input (both chain types)
		if (HopRewriteUtils.isTransposeOperation(in1)) {
			Hop X = in1.getInput().get(0);

			//check mapmultchain patterns
			//t(X)%*%(w*(X%*%v))
			if (in2 instanceof BinaryOp && ((BinaryOp) in2).getOp() == OpOp2.MULT) {
				Hop in3b = in2.getInput().get(1);
				if (in3b instanceof AggBinaryOp) {
					Hop in4 = in3b.getInput().get(0);
					if (X == in4) //common input
						chainType = ChainType.XtwXv;
				}
			}
			//t(X)%*%((X%*%v)-y)
			else if (in2 instanceof BinaryOp && ((BinaryOp) in2).getOp() == OpOp2.MINUS) {
				Hop in3a = in2.getInput().get(0);
				Hop in3b = in2.getInput().get(1);
				if (in3a instanceof AggBinaryOp && in3b.getDataType() == DataType.MATRIX) {
					Hop in4 = in3a.getInput().get(0);
					if (X == in4) //common input
						chainType = ChainType.XtXvy;
				}
			}
			//t(X)%*%(X%*%v)
			else if (in2 instanceof AggBinaryOp) {
				Hop in3 = in2.getInput().get(0);
				if (X == in3) //common input
					chainType = ChainType.XtXv;
			}
		}

		return chainType;
	}

	//////////////////////////
	// CP Lops generation

	/// //////////////////////

	private void constructCPLopsTSMM(MMTSJType mmtsj, ExecType et) {
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		Lop matmultCP = new MMTSJ(getInput().get(mmtsj.isLeft() ? 1 : 0).constructLops(),
				getDataType(), getValueType(), et, mmtsj, false, k);
		matmultCP.getOutputParameters().setDimensions(getDim1(), getDim2(), getBlocksize(), getNnz());
		setLineNumbers(matmultCP);
		setLops(matmultCP);
	}

	private void constructCPLopsMMChain(ChainType chain) {
		MapMultChain mapmmchain = null;
		if (chain == ChainType.XtXv) {
			Hop hX = getInput().get(0).getInput().get(0);
			Hop hv = getInput().get(1).getInput().get(1);
			mapmmchain = new MapMultChain(hX.constructLops(), hv.constructLops(), getDataType(), getValueType(), ExecType.CP);
		} else { //ChainType.XtwXv / ChainType.XtwXvy
			int wix = (chain == ChainType.XtwXv) ? 0 : 1;
			int vix = (chain == ChainType.XtwXv) ? 1 : 0;
			Hop hX = getInput().get(0).getInput().get(0);
			Hop hw = getInput().get(1).getInput().get(wix);
			Hop hv = getInput().get(1).getInput().get(vix).getInput().get(1);
			mapmmchain = new MapMultChain(hX.constructLops(), hv.constructLops(), hw.constructLops(), chain, getDataType(), getValueType(), ExecType.CP);
		}

		//set degree of parallelism
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		mapmmchain.setNumThreads(k);

		//set basic lop properties
		setOutputDimensions(mapmmchain);
		setLineNumbers(mapmmchain);
		setLops(mapmmchain);
	}

	/**
	 * NOTE: exists for consistency since removeEmtpy might be scheduled to MR
	 * but matrix mult on small output might be scheduled to CP. Hence, we
	 * need to handle directly passed selection vectors in CP as well.
	 */
	private void constructCPLopsPMM() {
		Hop pmInput = getInput().get(0);
		Hop rightInput = getInput().get(1);

		Hop nrow = HopRewriteUtils.createValueHop(pmInput, true); //NROW
		nrow.setBlocksize(0);
		nrow.setForcedExecType(ExecType.CP);
		HopRewriteUtils.copyLineNumbers(this, nrow);
		Lop lnrow = nrow.constructLops();

		PMMJ pmm = new PMMJ(pmInput.constructLops(), rightInput.constructLops(), lnrow, getDataType(), getValueType(), false, false, ExecType.CP);

		//set degree of parallelism
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		pmm.setNumThreads(k);

		pmm.getOutputParameters().setDimensions(getDim1(), getDim2(), getBlocksize(), getNnz());
		setLineNumbers(pmm);

		setLops(pmm);

		HopRewriteUtils.removeChildReference(pmInput, nrow);
	}

	private void constructCPLopsMM(ExecType et) {
		Lop matmultCP = null;
		String cla = ConfigurationManager.getDMLConfig().getTextValue("sysds.compressed.linalg");
		if (et == ExecType.GPU) {
			Hop h1 = getInput().get(0);
			Hop h2 = getInput().get(1);
			// Since GPU backend is in experimental mode, rewrite optimization can be skipped.
			// CuSPARSE's cusparsecsrmm2 fails with only following parameters, but passes for all other settings:
			// transa=1 transb=1 m=300 n=300 k=300 ldb=300 ldc=300
			// Hence, we disable hope rewrite optimization.
			boolean leftTrans = false; // HopRewriteUtils.isTransposeOperation(h1);
			boolean rightTrans = false; // HopRewriteUtils.isTransposeOperation(h2);
			Lop left = !leftTrans ? h1.constructLops() :
					h1.getInput().get(0).constructLops();
			Lop right = !rightTrans ? h2.constructLops() :
					h2.getInput().get(0).constructLops();
			matmultCP = new MatMultCP(left, right, getDataType(), getValueType(), et, leftTrans, rightTrans);
			setOutputDimensions(matmultCP);
		} else if (cla.equals("true") || cla.equals("cost")) {
			Hop h1 = getInput().get(0);
			Hop h2 = getInput().get(1);
			int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
			boolean leftTrans = HopRewriteUtils.isTransposeOperation(h1);
			boolean rightTrans = HopRewriteUtils.isTransposeOperation(h2);
			Lop left = !leftTrans ? h1.constructLops() :
					h1.getInput().get(0).constructLops();
			Lop right = !rightTrans ? h2.constructLops() :
					h2.getInput().get(0).constructLops();
			matmultCP = new MatMultCP(left, right, getDataType(), getValueType(), et, k, leftTrans, rightTrans);
		} else {
			if (isLeftTransposeRewriteApplicable(true)) {
				matmultCP = constructCPLopsMMWithLeftTransposeRewrite(et);
			} else {
				int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
				matmultCP = new MatMultCP(getInput().get(0).constructLops(),
						getInput().get(1).constructLops(), getDataType(), getValueType(), et, k);
				updateLopFedOut(matmultCP);
			}
			setOutputDimensions(matmultCP);
		}

		setLineNumbers(matmultCP);
		setLops(matmultCP);
	}

	private Lop constructCPLopsMMWithLeftTransposeRewrite(ExecType et) {
		Hop X = getInput().get(0).getInput().get(0); // guaranteed to exist
		Hop Y = getInput().get(1);
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);

		//Check if X is already a transpose operation
		boolean isXTransposed = X instanceof ReorgOp && ((ReorgOp)X).getOp() == ReOrgOp.TRANS;
		Hop actualX = isXTransposed ? X.getInput().get(0) : X;

		//Check if Y is a transpose operation
		boolean isYTransposed = Y instanceof ReorgOp && ((ReorgOp)Y).getOp() == ReOrgOp.TRANS;
		Hop actualY = isYTransposed ? Y.getInput().get(0) : Y;

		//Handle Y or actualY for transpose
		Lop yLop = isYTransposed ? actualY.constructLops() : Y.constructLops();
		ExecType inputReorgExecType = (Y.hasFederatedOutput()) ? ExecType.FED : ExecType.CP;

		//right vector transpose
		Lop tY = (yLop instanceof Transform && ((Transform)yLop).getOp() == ReOrgOp.TRANS) ?
				yLop.getInputs().get(0) : //if input is already a transpose, avoid redundant transpose ops
				new Transform(yLop, ReOrgOp.TRANS, getDataType(), getValueType(), inputReorgExecType, k);

		//Set dimensions for tY
		long tYRows = isYTransposed ? actualY.getDim1() : Y.getDim2();
		long tYCols = isYTransposed ? actualY.getDim2() : Y.getDim1();
		tY.getOutputParameters().setDimensions(tYRows, tYCols, getBlocksize(), Y.getNnz());
		setLineNumbers(tY);
		if (Y.hasFederatedOutput())
			updateLopFedOut(tY);

		//Construct X lops for matrix multiplication
		Lop xLop = isXTransposed ? actualX.constructLops() : X.constructLops();

		//matrix mult
		Lop mult = new MatMultCP(tY, xLop, getDataType(), getValueType(), et, k);
		mult.getOutputParameters().setDimensions(tYRows, isXTransposed ? actualX.getDim1() : X.getDim2(), getBlocksize(), getNnz());
		mult.setFederatedOutput(_federatedOutput);
		setLineNumbers(mult);

		//result transpose (dimensions set outside)
		ExecType outTransposeExecType = (_federatedOutput == FederatedOutput.FOUT) ?
				ExecType.FED : ExecType.CP;
		Lop out = new Transform(mult, ReOrgOp.TRANS, getDataType(), getValueType(), outTransposeExecType, k);

		return out;
	}

	//////////////////////////
	// Spark Lops generation
	/////////////////////////

	private void constructSparkLopsTSMM(MMTSJType mmtsj, boolean multiPass) {
		Hop input = getInput().get(mmtsj.isLeft()?1:0);
		MMTSJ tsmm = new MMTSJ(input.constructLops(), getDataType(), 
				getValueType(), ExecType.SPARK, mmtsj, multiPass);
		setOutputDimensions(tsmm);
		setLineNumbers(tsmm);
		setLops(tsmm);
	}

	private void constructSparkLopsMapMM(MMultMethod method)
	{
		Lop mapmult = null;
		if( isLeftTransposeRewriteApplicable(false) ) 
		{
			mapmult = constructSparkLopsMapMMWithLeftTransposeRewrite();
		}
		else
		{
			// If number of columns is smaller than block size then explicit aggregation is not required.
			// i.e., entire matrix multiplication can be performed in the mappers.
			boolean needAgg = requiresAggregation(method); 
			SparkAggType aggtype = getSparkMMAggregationType(needAgg);
			_outputEmptyBlocks = !OptimizerUtils.allowsToFilterEmptyBlockOutputs(this); 
			
			//core matrix mult
			mapmult = new MapMult( getInput().get(0).constructLops(), getInput().get(1).constructLops(), 
				getDataType(), getValueType(), (method==MMultMethod.MAPMM_R), false, 
				_outputEmptyBlocks, aggtype);
		}
		setOutputDimensions(mapmult);
		setLineNumbers(mapmult);
		setLops(mapmult);	
	}

	private Lop constructSparkLopsMapMMWithLeftTransposeRewrite()
	{
		Hop X = getInput().get(0).getInput().get(0); //guaranteed to exists
		Hop Y = getInput().get(1);

		//right vector transpose
		Lop tY = new Transform(Y.constructLops(), ReOrgOp.TRANS, getDataType(), getValueType(), ExecType.CP);
		tY.getOutputParameters().setDimensions(Y.getDim2(), Y.getDim1(), getBlocksize(), Y.getNnz());
		setLineNumbers(tY);

		//matrix mult spark
		boolean needAgg = requiresAggregation(MMultMethod.MAPMM_R);
		SparkAggType aggtype = getSparkMMAggregationType(needAgg);
		_outputEmptyBlocks = !OptimizerUtils.allowsToFilterEmptyBlockOutputs(this);

		Lop mult = new MapMult( tY, X.constructLops(), getDataType(), getValueType(),
				      false, false, _outputEmptyBlocks, aggtype);
		mult.getOutputParameters().setDimensions(Y.getDim2(), X.getDim2(), getBlocksize(), getNnz());
		setLineNumbers(mult);

		//result transpose (dimensions set outside)
		Lop out = new Transform(mult, ReOrgOp.TRANS, getDataType(), getValueType(), ExecType.CP);

		return out;
	}

	private void constructSparkLopsMapMMChain(ChainType chain) {
		MapMultChain mapmmchain = null;
		if( chain == ChainType.XtXv ) {
			Hop hX = getInput().get(0).getInput().get(0);
			Hop hv = getInput().get(1).getInput().get(1);
			mapmmchain = new MapMultChain( hX.constructLops(), hv.constructLops(), getDataType(), getValueType(), ExecType.SPARK);
		}
		else { //ChainType.XtwXv / ChainType.XtXvy
			int wix = (chain == ChainType.XtwXv) ? 0 : 1;
			int vix = (chain == ChainType.XtwXv) ? 1 : 0;
			Hop hX = getInput().get(0).getInput().get(0);
			Hop hw = getInput().get(1).getInput().get(wix);
			Hop hv = getInput().get(1).getInput().get(vix).getInput().get(1);
			mapmmchain = new MapMultChain( hX.constructLops(), hv.constructLops(), hw.constructLops(), chain, getDataType(), getValueType(), ExecType.SPARK);
		}
		setOutputDimensions(mapmmchain);
		setLineNumbers(mapmmchain);
		setLops(mapmmchain);
	}

	private void constructSparkLopsPMapMM() {
		PMapMult pmapmult = new PMapMult( 
				getInput().get(0).constructLops(),
				getInput().get(1).constructLops(), 
				getDataType(), getValueType() );
		setOutputDimensions(pmapmult);
		setLineNumbers(pmapmult);
		setLops(pmapmult);
	}

	private void constructSparkLopsCPMM() {
		if( isLeftTransposeRewriteApplicable(false) ) {
			setLops( constructSparkLopsCPMMWithLeftTransposeRewrite() );
		} 
		else {
			SparkAggType aggtype = getSparkMMAggregationType(true);
			_outputEmptyBlocks = !OptimizerUtils.allowsToFilterEmptyBlockOutputs(this);
			Lop cpmm = new MMCJ(getInput().get(0).constructLops(), getInput().get(1).constructLops(),
				getDataType(), getValueType(), _outputEmptyBlocks, aggtype, ExecType.SPARK);
			setOutputDimensions( cpmm );
			setLineNumbers( cpmm );
			setLops( cpmm );
		}
	}

	private Lop constructSparkLopsCPMMWithLeftTransposeRewrite() {
		SparkAggType aggtype = getSparkMMAggregationType(true);
		
		Hop X = getInput().get(0).getInput().get(0); //guaranteed to exists
		Hop Y = getInput().get(1);
		
		//right vector transpose CP
		Lop tY = new Transform(Y.constructLops(), ReOrgOp.TRANS, getDataType(), getValueType(), ExecType.CP);
		tY.getOutputParameters().setDimensions(Y.getDim2(), Y.getDim1(), Y.getBlocksize(), Y.getNnz());
		setLineNumbers(tY);
		
		//matrix multiply
		_outputEmptyBlocks = !OptimizerUtils.allowsToFilterEmptyBlockOutputs(this); 
		MMCJ mmcj = new MMCJ(tY, X.constructLops(), getDataType(), getValueType(), _outputEmptyBlocks, aggtype, ExecType.SPARK);
		mmcj.getOutputParameters().setDimensions(getDim1(), getDim2(), getBlocksize(), getNnz());
		setLineNumbers(mmcj);

		//result transpose CP 
		Lop out = new Transform(mmcj, ReOrgOp.TRANS, getDataType(), getValueType(), ExecType.CP);
		out.getOutputParameters().setDimensions(X.getDim2(), Y.getDim2(), getBlocksize(), getNnz());
		
		return out;
	}

	private void constructSparkLopsRMM() {
		Lop rmm = new MMRJ(getInput().get(0).constructLops(),getInput().get(1).constructLops(), 
			getDataType(), getValueType(), ExecType.SPARK);
		setOutputDimensions(rmm);
		//setMarkForLineageCaching(rmm);
		setLineNumbers( rmm );
		setLops(rmm);
	}

	private void constructSparkLopsPMM() {
		//PMM has two potential modes (a) w/ full permutation matrix input, and 
		//(b) w/ already condensed input vector of target row positions.
		
		Hop pmInput = getInput().get(0);
		Hop rightInput = getInput().get(1);
		
		Lop lpmInput = pmInput.constructLops();
		Hop nrow = null;
		double mestPM = OptimizerUtils.estimateSize(pmInput.getDim1(), 1);
		ExecType etVect = (mestPM>OptimizerUtils.getLocalMemBudget())?ExecType.SPARK:ExecType.CP;
		
		//a) full permutation matrix input (potentially without empty block materialized)
		if( pmInput.getDim2() != 1 ) //not a vector
		{
			//compute condensed permutation matrix vector input			
			//v = rowMaxIndex(t(pm)) * rowMax(t(pm)) 
			ReorgOp transpose = HopRewriteUtils.createTranspose(pmInput);
			transpose.setForcedExecType(ExecType.SPARK);
			
			AggUnaryOp agg1 = HopRewriteUtils.createAggUnaryOp(transpose, AggOp.MAXINDEX, Direction.Row);
			agg1.setForcedExecType(ExecType.SPARK);
			
			AggUnaryOp agg2 = HopRewriteUtils.createAggUnaryOp(transpose, AggOp.MAX, Direction.Row);
			agg2.setForcedExecType(ExecType.SPARK);
			
			BinaryOp mult = HopRewriteUtils.createBinary(agg1, agg2, OpOp2.MULT);
			mult.setForcedExecType(ExecType.SPARK);
			
			//compute NROW target via nrow(m)
			nrow = HopRewriteUtils.createValueHop(pmInput, true);
			nrow.setBlocksize(0);
			nrow.setForcedExecType(ExecType.CP);
			HopRewriteUtils.copyLineNumbers(this, nrow);
			
			lpmInput = mult.constructLops();
			HopRewriteUtils.removeChildReference(pmInput, transpose);
		}
		else //input vector
		{
			//compute NROW target via max(v)
			nrow = HopRewriteUtils.createAggUnaryOp(pmInput, AggOp.MAX, Direction.RowCol); 
			nrow.setBlocksize(0);
			nrow.setForcedExecType(etVect);
			HopRewriteUtils.copyLineNumbers(this, nrow);
		}
		
		//b) condensed permutation matrix vector input (target rows)
		_outputEmptyBlocks = !OptimizerUtils.allowsToFilterEmptyBlockOutputs(this); 
		PMMJ pmm = new PMMJ(lpmInput, rightInput.constructLops(), nrow.constructLops(), 
				getDataType(), getValueType(), false, _outputEmptyBlocks, ExecType.SPARK);
		setOutputDimensions(pmm);
		setLineNumbers(pmm);
		setLops(pmm);
		
		HopRewriteUtils.removeChildReference(pmInput, nrow);
	} 

	private void constructSparkLopsZIPMM() {
		//zipmm applies to t(X)%*%y if ncol(X)<=blocksize and it prevents 
		//unnecessary reshuffling by keeping the original indexes (and partitioning) 
		//joining the datasets, and internally doing the necessary transpose operations
		
		Hop left = getInput().get(0).getInput().get(0); //x out of t(X)
		Hop right = getInput().get(1); //y

		//determine left-transpose rewrite beneficial
		boolean tRewrite = (left.getDim1()*left.getDim2() >= right.getDim1()*right.getDim2());
		
		Lop zipmm = new MMZip(left.constructLops(), right.constructLops(), getDataType(), getValueType(), tRewrite, ExecType.SPARK);
		setOutputDimensions(zipmm);
		setLineNumbers( zipmm );
		setLops(zipmm);
	}

	/**
	 * Determines if the rewrite t(X)%*%Y -> t(t(Y)%*%X) is applicable
	 * and cost effective. Whenever X is a wide matrix and Y is a vector
	 * this has huge impact, because the transpose of X would dominate
	 * the entire operation costs.
	 *
	 * @param CP true if CP
	 * @return true if left transpose rewrite applicable
	 */
	private boolean isLeftTransposeRewriteApplicable(boolean CP)
	{
		//check for forced MR or Spark execution modes, which prevent the introduction of
		//additional CP operations and hence the rewrite application
		if( DMLScript.getGlobalExecMode() == ExecMode.SPARK ) //not HYBRID
		{
			return false;
		}

		boolean ret = false;
		Hop h1 = getInput().get(0);
		Hop h2 = getInput().get(1);

		//check for known dimensions and cost for t(X) vs t(v) + t(tvX)
		//(for both CP/MR, we explicitly check that new transposes fit in memory,
		//even a ba in CP does not imply that both transposes can be executed in CP)
		if( CP ) //in-memory ba
		{
			if( HopRewriteUtils.isTransposeOperation(h1) )
			{
				long m = h1.getDim1();
				long cd = h1.getDim2();
				long n = h2.getDim2();

				//check for known dimensions (necessary condition for subsequent checks)
				ret = (m>0 && cd>0 && n>0);

				//check operation memory with changed transpose (this is important if we have
				//e.g., t(X) %*% v, where X is sparse and tX fits in memory but X does not
				double memX = h1.getInput().get(0).getOutputMemEstimate();
				double memtv = OptimizerUtils.estimateSizeExactSparsity(n, cd, 1.0);
				double memtXv = OptimizerUtils.estimateSizeExactSparsity(n, m, 1.0);
				double newMemEstimate = memtv + memX + memtXv;
				ret &= ( newMemEstimate < OptimizerUtils.getLocalMemBudget() );

				//check for cost benefit of t(X) vs t(v) + t(tvX) and memory of additional transpose ops
				ret &= ( m*cd > (cd*n + m*n) &&
						2 * OptimizerUtils.estimateSizeExactSparsity(cd, n, 1.0) < OptimizerUtils.getLocalMemBudget() &&
						2 * OptimizerUtils.estimateSizeExactSparsity(m, n, 1.0) < OptimizerUtils.getLocalMemBudget() );

				//update operation memory estimate (e.g., for parfor optimizer)
				if( ret )
					_memEstimate = newMemEstimate;
			}
		}
		else //MR
		{
			if( HopRewriteUtils.isTransposeOperation(h1) ) {
				long m = h1.getDim1();
				long cd = h1.getDim2();
				long n = h2.getDim2();
				//note: output size constraint for mapmult already checked by optfindmmultmethod
				if( m>0 && cd>0 && n>0 && (m*cd > (cd*n + m*n)) &&
						2 * OptimizerUtils.estimateSizeExactSparsity(cd, n, 1.0) <  OptimizerUtils.getLocalMemBudget() &&
						2 * OptimizerUtils.estimateSizeExactSparsity(m, n, 1.0) <  OptimizerUtils.getLocalMemBudget() )
				{
					ret = true;
				}
			}
		}

		return ret;
	}

	private SparkAggType getSparkMMAggregationType( boolean agg ) {
		if( !agg )
			return SparkAggType.NONE;
		if( dimsKnown() && getDim1()<=getBlocksize() && getDim2()<=getBlocksize() )
			return SparkAggType.SINGLE_BLOCK;
		return SparkAggType.MULTI_BLOCK;
	}

	private boolean requiresAggregation(MMultMethod method) 
	{
		//worst-case assumption (for plan correctness)
		boolean ret = true;
		
		//right side cached (no agg if left has just one column block)
		if(  method == MMultMethod.MAPMM_R && getInput().get(0).getDim2() >= 0 //known num columns
			&& getInput().get(0).getDim2() <= getInput().get(0).getBlocksize() ) 
		{
			ret = false;
		}

		//left side cached (no agg if right has just one row block)
		if(  method == MMultMethod.MAPMM_L && getInput().get(1).getDim1() >= 0 //known num rows
			&& getInput().get(1).getDim1() <= getInput().get(1).getBlocksize() ) 
		{
			ret = false;
		}

		return ret;
	}
	
	/**
	 * Estimates the memory footprint of MapMult operation depending on which input is put into distributed cache.
	 * This function is called by <code>optFindMMultMethod()</code> to decide the execution strategy, as well as by 
	 * piggybacking to decide the number of Map-side instructions to put into a single GMR job.
	 * 
	 * @param m1_rows m1 rows
	 * @param m1_cols m1 cols
	 * @param m1_blen m1 rows/cols per block
	 * @param m1_nnz m1 num non-zeros
	 * @param m2_rows m2 rows
	 * @param m2_cols m2 cols
	 * @param m2_blen m2 rows/cols per block
	 * @param m2_nnz m2 num non-zeros
	 * @param cachedInputIndex true if cached input index
	 * @param pmm true if permutation matrix multiply
	 * @return map mm memory estimate
	 */
	public static double getMapmmMemEstimate(long m1_rows, long m1_cols, long m1_blen, long m1_nnz,
			long m2_rows, long m2_cols, long m2_blen, long m2_nnz, int cachedInputIndex, boolean pmm) 
	{
		// If the size of one input is small, choose a method that uses distributed cache
		// NOTE: be aware of output size because one input block might generate many output blocks
		double m1SizeP = OptimizerUtils.estimatePartitionedSizeExactSparsity(m1_rows, m1_cols, m1_blen, m1_nnz); //m1 partitioned 
		double m2SizeP = OptimizerUtils.estimatePartitionedSizeExactSparsity(m2_rows, m2_cols, m2_blen, m2_nnz); //m2 partitioned
		
		double m1BlockSize = OptimizerUtils.estimateSize(Math.min(m1_rows, m1_blen), Math.min(m1_cols, m1_blen));
		double m2BlockSize = OptimizerUtils.estimateSize(Math.min(m2_rows, m2_blen), Math.min(m2_cols, m2_blen));
		double m3m1OutSize = OptimizerUtils.estimateSize(Math.min(m1_rows, m1_blen), m2_cols); //output per m1 block if m2 in cache
		double m3m2OutSize = OptimizerUtils.estimateSize(m1_rows, Math.min(m2_cols, m2_blen)); //output per m2 block if m1 in cache
	
		double footprint = 0;
		if( pmm )
		{
			//permutation matrix multiply 
			//(one input block -> at most two output blocks)
			footprint = m1SizeP + 3*m2BlockSize; //in+2*out
		}
		else
		{
			//generic matrix multiply
			if ( cachedInputIndex == 1 ) {
				// left input (m1) is in cache
				footprint = m1SizeP+m2BlockSize+m3m2OutSize;
			}
			else {
				// right input (m2) is in cache
				footprint = m1BlockSize+m2SizeP+m3m1OutSize;
			}	
		}
		
		return footprint;
	}

	private static MMultMethod optFindMMultMethodCP( long m1_rows, long m1_cols, long m2_rows, long m2_cols, MMTSJType mmtsj, ChainType chainType, boolean leftPM ) 
	{	
		//step 1: check for TSMM pattern
		if( mmtsj != MMTSJType.NONE )
			return MMultMethod.TSMM;
		
		//step 2: check for MMChain pattern
		if( chainType != ChainType.NONE && OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES && m2_cols==1 )
			return MMultMethod.MAPMM_CHAIN;
		
		//step 3: check for PMM
		if( leftPM && m1_cols==1 && m2_rows!=1 )
			return MMultMethod.PMM;
		
		//step 4: general purpose MM
		return MMultMethod.MM; 
	}
	
	private MMultMethod optFindMMultMethodSpark( long m1_rows, long m1_cols, long m1_blen, long m1_nnz, 
		long m2_rows, long m2_cols, long m2_blen, long m2_nnz,
		MMTSJType mmtsj, ChainType chainType, boolean leftPMInput, boolean tmmRewrite ) 
	{
		//Notes: Any broadcast needs to fit twice in local memory because we partition the input in cp,
		//and needs to fit once in executor broadcast memory. The 2GB broadcast constraint is no longer
		//required because the max_int byte buffer constraint has been fixed in Spark 1.4 
		double memBudgetExec = MAPMULT_MEM_MULTIPLIER * SparkExecutionContext.getBroadcastMemoryBudget();
		double memBudgetLocal = OptimizerUtils.getLocalMemBudget();

		//reset spark broadcast memory information (for concurrent parfor jobs, awareness of additional 
		//cp memory requirements on spark rdd operations with broadcasts)
		_spBroadcastMemEstimate = 0;
		
		// Step 0: check for forced mmultmethod
		if( FORCED_MMULT_METHOD !=null )
			return FORCED_MMULT_METHOD;
		
		// Step 1: check TSMM
		// If transpose self pattern and result is single block:
		// use specialized TSMM method (always better than generic jobs)
		if(    ( mmtsj == MMTSJType.LEFT && m2_cols>=0 && m2_cols <= m2_blen )
			|| ( mmtsj == MMTSJType.RIGHT && m1_rows>=0 && m1_rows <= m1_blen ) )
		{
			return MMultMethod.TSMM;
		}
		
		// Step 2: check MapMMChain
		// If mapmultchain pattern and result is a single block:
		// use specialized mapmult method
		if( OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES )
		{
			//matmultchain if dim2(X)<=blocksize and all vectors fit in mappers
			//(X: m1_cols x m1_rows, v: m1_rows x m2_cols, w: m1_cols x m2_cols) 
			//NOTE: generalization possibe: m2_cols>=0 && m2_cols<=m2_cpb
			if( chainType!=ChainType.NONE && m1_rows >=0 && m1_rows <= m1_blen && m2_cols==1 )
			{
				if( chainType==ChainType.XtXv && m1_rows>=0 && m2_cols>=0 
					&& OptimizerUtils.estimateSize(m1_rows, m2_cols ) < memBudgetExec )
				{
					return MMultMethod.MAPMM_CHAIN;
				}
				else if( (chainType==ChainType.XtwXv || chainType==ChainType.XtXvy ) 
					&& m1_rows>=0 && m2_cols>=0 && m1_cols>=0
					&&   OptimizerUtils.estimateSize(m1_rows, m2_cols) 
					   + OptimizerUtils.estimateSize(m1_cols, m2_cols) < memBudgetExec
					&& 2*(OptimizerUtils.estimateSize(m1_rows, m2_cols) 
					   + OptimizerUtils.estimateSize(m1_cols, m2_cols)) < memBudgetLocal )
				{
					_spBroadcastMemEstimate = 2*(OptimizerUtils.estimateSize(m1_rows, m2_cols) 
						+ OptimizerUtils.estimateSize(m1_cols, m2_cols));
					return MMultMethod.MAPMM_CHAIN;
				}
			}
		}
		
		// Step 3: check for PMM (permutation matrix needs to fit into mapper memory)
		// (needs to be checked before mapmult for consistency with removeEmpty compilation 
		double footprintPM1 = getMapmmMemEstimate(m1_rows, 1, m1_blen, m1_nnz, m2_rows, m2_cols, m2_blen, m2_nnz, 1, true);
		double footprintPM2 = getMapmmMemEstimate(m2_rows, 1, m1_blen, m1_nnz, m2_rows, m2_cols, m2_blen, m2_nnz, 1, true);
		if( (footprintPM1 < memBudgetExec && m1_rows>=0 || footprintPM2 < memBudgetExec && m2_rows>=0)
			&& 2*OptimizerUtils.estimateSize(m1_rows, 1) < memBudgetLocal
			&& leftPMInput ) 
		{
			_spBroadcastMemEstimate = 2*OptimizerUtils.estimateSize(m1_rows, 1);
			return MMultMethod.PMM;
		}
		
		// Step 4: check MapMM
		// If the size of one input is small, choose a method that uses broadcast variables to prevent shuffle
		
		//memory estimates for local partitioning (mb -> partitioned mb)
		double m1Size = OptimizerUtils.estimateSizeExactSparsity(m1_rows, m1_cols, m1_nnz); //m1 single block
		double m2Size = OptimizerUtils.estimateSizeExactSparsity(m2_rows, m2_cols, m2_nnz); //m2 single block
		double m1SizeP = OptimizerUtils.estimatePartitionedSizeExactSparsity(m1_rows, m1_cols, m1_blen, m1_nnz); //m1 partitioned 
		double m2SizeP = OptimizerUtils.estimatePartitionedSizeExactSparsity(m2_rows, m2_cols, m2_blen, m2_nnz); //m2 partitioned
		
		//memory estimates for remote execution (broadcast and outputs)
		double footprint1 = getMapmmMemEstimate(m1_rows, m1_cols, m1_blen, m1_nnz, m2_rows, m2_cols, m2_blen, m2_nnz, 1, false);
		double footprint2 = getMapmmMemEstimate(m1_rows, m1_cols, m1_blen, m1_nnz, m2_rows, m2_cols, m2_blen, m2_nnz, 2, false);		
		
		if (   (footprint1 < memBudgetExec && m1Size+m1SizeP < memBudgetLocal && m1_rows>=0 && m1_cols>=0)
			|| (footprint2 < memBudgetExec && m2Size+m2SizeP < memBudgetLocal && m2_rows>=0 && m2_cols>=0) ) 
		{
			//apply map mult if one side fits in remote task memory 
			//(if so pick smaller input for distributed cache)
			//TODO relax requirement of valid CP dimensions once we support broadcast creation from files/RDDs
			double em1Size = getInput().get(0).getOutputMemEstimate(); //w/ worst-case estimate
			double em2Size = getInput().get(1).getOutputMemEstimate(); //w/ worst-case estimate
			if( (m1SizeP < m2SizeP || (m1SizeP==m2SizeP && em1Size<em2Size) )
				&& m1_rows>=0 && m1_cols>=0
				&& OptimizerUtils.isValidCPDimensions(m1_rows, m1_cols) ) {
				_spBroadcastMemEstimate = m1Size+m1SizeP;
				return MMultMethod.MAPMM_L;
			}
			else if( OptimizerUtils.isValidCPDimensions(m2_rows, m2_cols) ) {
				_spBroadcastMemEstimate = m2Size+m2SizeP;
				return MMultMethod.MAPMM_R;
			}
		}
		
		// Step 5: check for TSMM2 (2 pass w/o suffle, preferred over CPMM/RMM)
		if( mmtsj != MMTSJType.NONE && m1_rows >=0 && m1_cols>=0 
			&& m2_rows >= 0 && m2_cols>=0 )
		{
			double mSize = (mmtsj == MMTSJType.LEFT) ? 
					OptimizerUtils.estimateSizeExactSparsity(m2_rows, m2_cols-m2_blen, 1.0) : 
					OptimizerUtils.estimateSizeExactSparsity(m1_rows-m1_blen, m1_cols, 1.0);
			double mSizeP = (mmtsj == MMTSJType.LEFT) ? 
					OptimizerUtils.estimatePartitionedSizeExactSparsity(m2_rows, m2_cols-m2_blen, m2_blen, 1.0) : 
					OptimizerUtils.estimatePartitionedSizeExactSparsity(m1_rows-m1_blen, m1_cols, m1_blen, 1.0); 
			if( mSizeP < memBudgetExec && mSize+mSizeP < memBudgetLocal 
				&& ((mmtsj == MMTSJType.LEFT) ? m2_cols<=2*m2_blen : m1_rows<=2*m1_blen) //4 output blocks
				&& mSizeP < 2L*1024*1024*1024) { //2GB limitation as single broadcast
				return MMultMethod.TSMM2;
			}
		}
		
		// Step 6: check for unknowns
		// If the dimensions are unknown at compilation time, simply assume 
		// the worst-case scenario and produce the most robust plan -- which is CPMM
		if ( m1_rows == -1 || m1_cols == -1 || m2_rows == -1 || m2_cols == -1 )
			return MMultMethod.CPMM;

		// Step 7: check for ZIPMM
		// If t(X)%*%y -> t(t(y)%*%X) rewrite and ncol(X)<blocksize
		if( tmmRewrite && m1_rows >= 0 && m1_rows <= m1_blen  //blocksize constraint left
			&& m2_cols >= 0 && m2_cols <= m2_blen )           //blocksize constraint right
		{
			return MMultMethod.ZIPMM;
		}
		
		// Step 8: Decide CPMM vs RMM based on io costs
		//estimate shuffle costs weighted by parallelism
		//TODO currently we reuse the mr estimates, these need to be fine-tune for our spark operators
		double rmm_costs = getRMMCostEstimate(m1_rows, m1_cols, m1_blen, m2_rows, m2_cols, m2_blen);
		double cpmm_costs = getCPMMCostEstimate(m1_rows, m1_cols, m1_blen, m2_rows, m2_cols, m2_blen);
		
		//final mmult method decision 
		if ( cpmm_costs < rmm_costs ) 
			return MMultMethod.CPMM;
		return MMultMethod.RMM;
	}

	private static double getRMMCostEstimate( long m1_rows, long m1_cols, long m1_blen, 
			long m2_rows, long m2_cols, long m2_blen )
	{
		long m1_nrb = (long) Math.ceil((double)m1_rows/m1_blen); // number of row blocks in m1
		long m2_ncb = (long) Math.ceil((double)m2_cols/m2_blen); // number of column blocks in m2

		// TODO: we must factor in the "sparsity"
		double m1_size = m1_rows * m1_cols;
		double m2_size = m2_rows * m2_cols;
		double result_size = m1_rows * m2_cols;

		int numReducersRMM = OptimizerUtils.getNumTasks();
		
		// Estimate the cost of RMM
		// RMM phase 1
		double rmm_shuffle = (m2_ncb*m1_size) + (m1_nrb*m2_size);
		double rmm_io = m1_size + m2_size + result_size;
		double rmm_nred = Math.min( m1_nrb * m2_ncb, //max used reducers 
				                    numReducersRMM); //available reducers
		// RMM total costs
		double rmm_costs = (rmm_shuffle + rmm_io) / rmm_nred;
		
		// return total costs
		return rmm_costs;
	}

	private static double getCPMMCostEstimate( long m1_rows, long m1_cols, long m1_blen, 
		long m2_rows, long m2_cols, long m2_blen )
	{
		long m1_nrb = (long) Math.ceil((double)m1_rows/m1_blen); // number of row blocks in m1
		long m1_ncb = (long) Math.ceil((double)m1_cols/m1_blen); // number of column blocks in m1
		long m2_ncb = (long) Math.ceil((double)m2_cols/m2_blen); // number of column blocks in m2

		// TODO: we must factor in the "sparsity"
		double m1_size = m1_rows * m1_cols;
		double m2_size = m2_rows * m2_cols;
		double result_size = m1_rows * m2_cols;

		int numReducersCPMM = OptimizerUtils.getNumTasks();
		
		// Estimate the cost of CPMM
		// CPMM phase 1
		double cpmm_shuffle1 = m1_size + m2_size;
		double cpmm_nred1 = Math.min( m1_ncb, //max used reducers 
			numReducersCPMM); //available reducer
		double cpmm_io1 = m1_size + m2_size + cpmm_nred1 * result_size;
		// CPMM phase 2
		double cpmm_shuffle2 = cpmm_nred1 * result_size;
		double cpmm_io2 = cpmm_nred1 * result_size + result_size;
		double cpmm_nred2 = Math.min( m1_nrb * m2_ncb, //max used reducers 
			numReducersCPMM); //available reducers
		// CPMM total costs
		double cpmm_costs =  (cpmm_shuffle1+cpmm_io1)/cpmm_nred1  //cpmm phase1
			+(cpmm_shuffle2+cpmm_io2)/cpmm_nred2; //cpmm phase2
		
		//return total costs
		return cpmm_costs;
	}
	
	@Override
	public void refreshSizeInformation() {
		Hop input1 = getInput().get(0);
		Hop input2 = getInput().get(1);
		if( isMatrixMultiply() ) {
			setDim1(input1.getDim1());
			setDim2(input2.getDim2());
			setNnz(-1); // for reset on recompile w/ unknowns 
			if( input1.getNnz() == 0 || input2.getNnz() == 0 )
				setNnz(0);
			if(hasCompressedInput() && !isRequiredDecompression() && input1.isCompressedOutput()) {

				// right matrix multiplication ... compressed output
				setCompressedOutput(true);
				// conservatively set the size to a multiplication of the compression size
				// multiplied with the number of columns.
				setCompressedSize(input1._compressedSize * input2.getDim2());
			}
		}
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		AggBinaryOp ret = new AggBinaryOp();
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret.innerOp = innerOp;
		ret.outerOp = outerOp;
		ret._hasLeftPMInput = _hasLeftPMInput;
		ret._maxNumThreads = _maxNumThreads;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof AggBinaryOp) )
			return false;
		
		AggBinaryOp that2 = (AggBinaryOp)that;
		return (   innerOp == that2.innerOp
				&& outerOp == that2.outerOp
				&& getInput().get(0) == that2.getInput().get(0)
				&& getInput().get(1) == that2.getInput().get(1)
				&& _hasLeftPMInput == that2._hasLeftPMInput
				&& _maxNumThreads == that2._maxNumThreads);
	}
}
