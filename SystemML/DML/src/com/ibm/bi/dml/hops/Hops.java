package com.ibm.bi.dml.hops;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.OptimizerUtils.OptimizationType;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LopsException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;


abstract public class Hops {
	protected static final Log LOG =  LogFactory.getLog(Hops.class.getName());
	
	public static boolean BREAKONSCALARS = false;
	public static boolean SPLITLARGEMATRIXMULT = true;
	public static long CPThreshold = 2000;
	
	public enum Kind {
		UnaryOp, BinaryOp, AggUnaryOp, AggBinaryOp, ReorgOp, Reblock, DataOp, LiteralOp, PartitionOp, CrossvalOp, RandOp, GenericFunctionOp, 
		TertiaryOp, ParameterizedBuiltinOp, Indexing, FunctionOp
	};

	public enum VISIT_STATUS {
		DONE, VISITING, NOTVISITED
	}

	// static variable to assign an unique ID to every hop that is created
	private static IDSequence UniqueHopID = new IDSequence();
	
	protected long ID;
	protected Kind _kind;
	protected String _name;
	protected DataType _dataType;
	protected ValueType _valueType;
	protected VISIT_STATUS _visited = VISIT_STATUS.NOTVISITED;
	protected long _dim1 = -1;
	protected long _dim2 = -1;
	protected long _rows_in_block = -1;
	protected long _cols_in_block = -1;
	protected long _nnz = -1;

	protected ArrayList<Hops> _parent = new ArrayList<Hops>();
	protected ArrayList<Hops> _input = new ArrayList<Hops>();

	private Lops _lops = null;
	private SQLLops _sqllops = null;

	protected ExecType _etype = null; //currently used exec type
	protected ExecType _etypeForced = null; //exec type forced via platform or external optimizer
	
	// Estimated size for the output produced from this Hop
	protected double _outputMemEstimate = OptimizerUtils.INVALID_SIZE;
	
	// Estimated size for the entire operation represented by this Hop
	// It includes the memory required for all inputs as well as the output 
	protected double _memEstimate = OptimizerUtils.INVALID_SIZE;
	
	protected double _processingMemEstimate = 0;
	
	// indicates if there are unknowns during compilation 
	// (in that case re-complication ensures robustness and efficiency)
	protected boolean _requiresRecompile = false;
	
	protected Hops(){
		//default constructor for clone
	}
	
	
	private static long getNextHopID() {
		return UniqueHopID.getNextID();
	}
	
	public long getHopID() {
		return ID;
	}
	
	public ExecType getExecType()
	{
		return _etype;
	}
	
	/**
	 * 
	 * @return
	 */
	public ExecType getForcedExecType()
	{
		return _etypeForced;
	}
	
	/**
	 * 
	 * @param etype
	 */
	public void setForcedExecType(ExecType etype)
	{
		_etypeForced = etype;
	}
	
	/**
	 * 
	 * @return
	 */
	public abstract boolean allowsAllExecTypes();
	
	/**
	 * 
	 */
	public void checkAndSetForcedPlatform()
	{
		if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE )
			_etypeForced = ExecType.CP;
		else if ( DMLScript.rtplatform == RUNTIME_PLATFORM.HADOOP )
			_etypeForced = ExecType.MR;
	}
	
	
	/**
	 * Returns the memory estimate for the output produced from this Hop.
	 * It must be invoked only within Hops. From outside Hops, one must 
	 * only use getMemEstimate(), which gives memory required to store 
	 * all inputs and the output.
	 * 
	 * @return
	 */
	protected double getOutputSize() {
		return _outputMemEstimate;
	}
	
	protected double getInputOutputSize() {
		double sum = this._outputMemEstimate;
		sum += this._processingMemEstimate;
		for(Hops h : _input ) {
			sum += h._outputMemEstimate;
		}
		return sum;
	}
	
	protected double getInputSize() {
		double sum = 0;
		for(Hops h : _input ) {
			sum += h._outputMemEstimate;
		}
		return sum;
	}
	
	/**
	 * 
	 * @param pos
	 * @return
	 */
	protected double getInputSize( int pos ){
		double ret = 0;
		if( _input.size()>pos )
			ret = _input.get(pos)._outputMemEstimate;
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	public double getMemEstimate()
	{
		if ( OptimizerUtils.getOptType() == OptimizationType.MEMORY_BASED ) {
			if ( ! isMemEstimated() ) {
				computeMemEstimate();
			}
			return _memEstimate;
		}
		else {
			return OptimizerUtils.INVALID_SIZE;
		}
	}
	
	/**
	 * Returns memory estimate in bytes
	 * 
	 * @param mem
	 */
	public void setMemEstimate( double mem )
	{
		_memEstimate = mem;
	}

	public boolean isMemEstimated() {
		return (_memEstimate != OptimizerUtils.INVALID_SIZE);
	}

	/**
	 * Computes the estimate of memory required to store the output of this hop in memory. 
	 * Note that it DOES NOT include the memory needed for its inputs.
	 * 
	 * @return computed estimate
	 */
	public abstract double computeMemEstimate();

	
	/**
	 * This function is used only for sanity check.
	 * Returns true if estimates for all the hops in the DAG rooted at the current 
	 * hop are computed. Returns false if any of the hops have INVALID estimate.
	 * 
	 * @return
	 */
	public boolean checkEstimates() {
		boolean childStatus = true;
		for (Hops h : this.getInput())
			childStatus = childStatus && h.checkEstimates();
		return childStatus && (_memEstimate != OptimizerUtils.INVALID_SIZE);
	}
	
	/**
	 * Recursively computes memory estimates for all the Hops in the DAG rooted at the 
	 * current hop pointed by <code>this</code>.
	 * 
	 */
	public void refreshMemEstimates() {
		if (get_visited() == VISIT_STATUS.DONE)
			return;
		for (Hops h : this.getInput())
			h.refreshMemEstimates();
		this.computeMemEstimate();
		this.set_visited(VISIT_STATUS.DONE);
	}

	/**
	 * This method determines the execution type (CP, MR) based ONLY on the 
	 * estimated memory footprint required for this operation, which includes 
	 * memory for all inputs and the output represented by this Hop.
	 * 
	 * It is used when <code>OptimizationType = MEMORY_BASED</code>.
	 * This optimization schedules an operation to CP whenever inputs+output 
	 * fit in memory -- note that this decision MAY NOT be optimal in terms of 
	 * execution time.
	 * 
	 * @return
	 */
	protected ExecType findExecTypeByMemEstimate() {
		ExecType et = null;
		char c = ' ';
		if ( getMemEstimate() < OptimizerUtils.getMemBudget(true) ) {
			et = ExecType.CP;
		}
		else {
			et = ExecType.MR;
			c = '*';
		}
		
		if (LOG.isDebugEnabled()){
			String s = String.format("  %c %-5s %-8s (%s,%s)  %s", c, getHopID(), getOpString(), OptimizerUtils.toMB(_outputMemEstimate), OptimizerUtils.toMB(_memEstimate), et);
			//System.out.println(s);
			LOG.debug(s);
		}
		// This is the old format for reference
		// %c %-5s %-8s (%s,%s)  %s\n", c, getHopID(), getOpString(), OptimizerUtils.toMB(_outputMemEstimate), OptimizerUtils.toMB(_memEstimate), et);
		
		return et;
	}

	public ArrayList<Hops> getParent() {
		return _parent;
	}

	public ArrayList<Hops> getInput() {
		return _input;
	}

	public Hops(Kind k, String l, DataType dt, ValueType vt) {
		_kind = k;
		set_name(l);
		_dataType = dt;
		_valueType = vt;
		ID = getNextHopID();
	}

	public long get_rows_in_block() {
		return _rows_in_block;
	}

	public void set_rows_in_block(long rowsInBlock) {
		_rows_in_block = rowsInBlock;
	}

	public long get_cols_in_block() {
		return _cols_in_block;
	}

	public void set_cols_in_block(long colsInBlock) {
		_cols_in_block = colsInBlock;
	}

	public void setNnz(long nnz){
		_nnz = nnz;
	}
	
	public long getNnz(){
		return _nnz;
	}
	
	public double getSparsity() {
		
		if ( _dataType == DataType.SCALAR )
			return 1.0;
		
		if (dimsKnown() && _nnz > 0)
			return (double)_nnz/(double)(_dim1*_dim2);
		else 
			return OptimizerUtils.DEF_SPARSITY;
	}
	
	public Kind getKind() {
		return _kind;
	}

	abstract public Lops constructLops() throws HopsException, LopsException;
	
	abstract public SQLLops constructSQLLOPs() throws HopsException; 

	abstract protected ExecType optFindExecType() throws HopsException;
	
	abstract public String getOpString();

	protected boolean isVector() {
		return (dimsKnown() && (_dim1 == 1 || _dim2 == 1) );
	}
	
	protected boolean areDimsBelowThreshold() {
		return (_dim1 <= Hops.CPThreshold && _dim2 <= Hops.CPThreshold );
	}
	
	protected boolean dimsKnown() {
		return ( _dataType == DataType.SCALAR || (_dataType==DataType.MATRIX && _dim1 > 0 && _dim2 > 0) );
	}

	
	public static void resetVisitStatus( ArrayList<Hops> hops )
	{
		for( Hops hopRoot : hops )
			hopRoot.resetVisitStatus();
	}
	
	public void resetVisitStatus() {
		if (this.get_visited() == Hops.VISIT_STATUS.NOTVISITED)
			return;
		for (Hops h : this.getInput())
			h.resetVisitStatus();
		
		if(this.get_sqllops() != null)
			this.get_sqllops().set_visited(VISIT_STATUS.NOTVISITED);
		this.set_visited(Hops.VISIT_STATUS.NOTVISITED);
	}

	public void rule_BlockSizeAndReblock(int GLOBAL_BLOCKSIZE) throws HopsException {

		// TODO BJR: traverse HOP DAG and insert Reblock HOPs

		// Go to the source(s) of the DAG
		for (Hops hi : this.getInput()) {
			if (get_visited() != Hops.VISIT_STATUS.DONE)
				hi.rule_BlockSizeAndReblock(GLOBAL_BLOCKSIZE);
		}

		boolean canReblock = true;
		
		if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE )
			canReblock = false;
		
		if (this instanceof DataOp) {

			// if block size does not match
			if (canReblock && get_dataType() != DataType.SCALAR
					&& (get_rows_in_block() != GLOBAL_BLOCKSIZE || get_cols_in_block() != GLOBAL_BLOCKSIZE)) {

				if (((DataOp) this).get_dataop() == DataOp.DataOpTypes.PERSISTENTREAD) {
				
					// insert reblock after the hop
					ReblockOp r = new ReblockOp(this, GLOBAL_BLOCKSIZE, GLOBAL_BLOCKSIZE);
					r.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					r.computeMemEstimate();
					r.set_visited(Hops.VISIT_STATUS.DONE);
				
				} else if (((DataOp) this).get_dataop() == DataOp.DataOpTypes.PERSISTENTWRITE) {

					if (get_rows_in_block() == -1 && get_cols_in_block() == -1) {

						// if this dataop is for cell ouput, then no reblock is
						// needed as (A) all jobtypes can produce block2cell and
						// cell2cell and (B) we don't generate an explicit
						// instruction for it (the info is conveyed through
						// OutputInfo.

					} else if (getInput().get(0) instanceof ReblockOp && getInput().get(0).getParent().size() == 1) {

						// if a reblock is feeding into this, then use it if
						// this is
						// the only parent, otherwise new Reblock

						getInput().get(0).set_rows_in_block(this.get_rows_in_block());
						getInput().get(0).set_cols_in_block(this.get_cols_in_block());

					} else {

						ReblockOp r = new ReblockOp(this);
						r.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						r.computeMemEstimate();
						r.set_visited(Hops.VISIT_STATUS.DONE);
					}

				} else if (((DataOp) this).get_dataop() == DataOp.DataOpTypes.TRANSIENTWRITE
						|| ((DataOp) this).get_dataop() == DataOp.DataOpTypes.TRANSIENTREAD) {
					if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE ) {
						// simply copy the values from its input
						set_rows_in_block(getInput().get(0).get_rows_in_block());
						set_cols_in_block(getInput().get(0).get_cols_in_block());
					}
					else {
						// by default, all transient reads and writes are in blocked format
						set_rows_in_block(GLOBAL_BLOCKSIZE);
						set_cols_in_block(GLOBAL_BLOCKSIZE);
					}

				} else {
					throw new HopsException(this.printErrorLocation() + "unexpected non-scalar Data HOP in reblock.\n");
				}
			}
		} else {
			// TODO: following two lines are commented, and the subsequent hack is used instead!
			//set_rows_per_block(GLOBAL_BLOCKSIZE);
			//set_cols_per_block(GLOBAL_BLOCKSIZE);
			
			// TODO: this is hack!
			/*
			 * Handle hops whose output dimensions are unknown!
			 * 
			 * Constraint C1:
			 * Currently, only ctable() and groupedAggregate() fall into this category.
			 * The MR jobs for both these functions run in "cell" mode and hence make their
			 * blocking dimensions to (-1,-1).
			 * 
			 * Constraint C2:
			 * Blocking dimensions are not applicable for hops that produce scalars. 
			 * CMCOV and GroupedAgg jobs always run in "cell" mode, and hence they 
			 * produce output in cell format.
			 * 
			 * Constraint C3:
			 * Remaining hops will get their blocking dimensions from their input hops.
			 */
			
			if ( this instanceof ReblockOp ) {
				set_rows_in_block(GLOBAL_BLOCKSIZE);
				set_cols_in_block(GLOBAL_BLOCKSIZE);
			}
			
			// Constraint C1:
			//else if ( (this instanceof ParameterizedBuiltinOp && ((ParameterizedBuiltinOp)this)._op == ParamBuiltinOp.GROUPEDAGG) ) {
			//	set_rows_in_block(-1);
			//	set_cols_in_block(-1);
			//}
			
			// Constraint C2:
			else if ( this.get_dataType() == DataType.SCALAR ) {
				set_rows_in_block(-1);
				set_cols_in_block(-1);
			}

			// Constraint C3:
			else {
				if ( !canReblock ) {
					set_rows_in_block(-1);
					set_cols_in_block(-1);
				}
				else {
					set_rows_in_block(GLOBAL_BLOCKSIZE);
					set_cols_in_block(GLOBAL_BLOCKSIZE);
				}
				
				// if any input is not blocked then the output of current Hop should not be blocked
				for ( Hops h : getInput() ) {
					if ( h.get_dataType() == DataType.MATRIX && h.get_rows_in_block() == -1 && h.get_cols_in_block() == -1 ) {
						set_rows_in_block(-1);
						set_cols_in_block(-1);
						break;
					}
				}
			}
		}

		this.set_visited(Hops.VISIT_STATUS.DONE);

	}

	public void rule_RehangTransientWriteParents(StatementBlock sb) throws HopsException {
		if (this instanceof DataOp && ((DataOp) this).get_dataop() == DataOpTypes.TRANSIENTWRITE
				&& !this.getParent().isEmpty()) {

			// update parents inputs with data op input
			for (Hops p : this.getParent()) {
				p.getInput().set(p.getInput().indexOf(this), this.getInput().get(0));
			}

			// update dataop input parent to add new parents except for
			// dataop itself
			this.getInput().get(0).getParent().addAll(this.getParent());

			// remove dataop parents
			this.getParent().clear();

			// add dataop as root for this Hops DAG
			sb.get_hops().add(this);

			// do the same thing for my inputs (children)
			for (Hops hi : this.getInput()) {
				hi.rule_RehangTransientWriteParents(sb);
			}
		}
	}

	/**
	 * rule_OptimizeMMChains(): This method recurses through all Hops in the DAG
	 * to find chains that need to be optimized.
	 */
	public void rule_OptimizeMMChains() throws HopsException {
		if (this.getKind() == Hops.Kind.AggBinaryOp && ((AggBinaryOp) this).isMatrixMultiply()
				&& this.get_visited() != Hops.VISIT_STATUS.DONE) {
			// Try to find and optimize the chain in which current Hop is the
			// last operator
			this.optimizeMMChain();
		}

		for (Hops hi : this.getInput())
			hi.rule_OptimizeMMChains();

		this.set_visited(Hops.VISIT_STATUS.DONE);
	}

	/**
	 * mmChainDP(): Core method to perform dynamic programming on a given array
	 * of matrix dimensions.
	 */
	public int[][] mmChainDP(long dimArray[], int size) {

		long dpMatrix[][] = new long[size][size];

		int split[][] = new int[size][size];
		int i, j, k, l;

		for (i = 0; i < size; i++) {
			for (j = 0; j < size; j++) {
				dpMatrix[i][j] = 0;
				split[i][j] = -1;
			}
		}

		long cost;
		long MAX = Long.MAX_VALUE;
		for (l = 2; l <= size; l++) { // chain length
			for (i = 0; i < size - l + 1; i++) {
				j = i + l - 1;
				// find cost of (i,j)
				dpMatrix[i][j] = MAX;
				for (k = i; k <= j - 1; k++) {
					cost = dpMatrix[i][k] + dpMatrix[k + 1][j] + (dimArray[i] * dimArray[k + 1] * dimArray[j + 1]);
					if (cost < dpMatrix[i][j]) {
						dpMatrix[i][j] = cost;
						split[i][j] = k;
					}
				}
			}
		}

		return split;
	}

	/**
	 * mmChainRelinkHops(): This method gets invoked after finding the optimal
	 * order (split[][]) from dynamic programming. It relinks the Hops that are
	 * part of the mmChain. mmChain : basic operands in the entire matrix
	 * multiplication chain. mmOperators : Hops that store the intermediate
	 * results in the chain. For example: A = B %*% (C %*% D) there will be
	 * three Hops in mmChain (B,C,D), and two Hops in mmOperators (one for each
	 * %*%) .
	 */
	private void mmChainRelinkHops(Hops h, int i, int j, ArrayList<Hops> mmChain, ArrayList<Hops> mmOperators,
			int opIndex, int[][] split) {
		if (i == j)
			return;

		Hops input1, input2;
		// Set Input1 for current Hop h
		if (i == split[i][j]) {
			input1 = mmChain.get(i);
			h.getInput().add(mmChain.get(i));
			mmChain.get(i).getParent().add(h);
		} else {
			input1 = mmOperators.get(opIndex);
			h.getInput().add(mmOperators.get(opIndex));
			mmOperators.get(opIndex).getParent().add(h);
			opIndex = opIndex + 1;
		}

		// Set Input2 for current Hop h
		if (split[i][j] + 1 == j) {
			input2 = mmChain.get(j);
			h.getInput().add(mmChain.get(j));
			mmChain.get(j).getParent().add(h);
		} else {
			input2 = mmOperators.get(opIndex);
			h.getInput().add(mmOperators.get(opIndex));
			mmOperators.get(opIndex).getParent().add(h);
			opIndex = opIndex + 1;
		}

		// Find children for both the inputs
		mmChainRelinkHops(h.getInput().get(0), i, split[i][j], mmChain, mmOperators, opIndex, split);
		mmChainRelinkHops(h.getInput().get(1), split[i][j] + 1, j, mmChain, mmOperators, opIndex, split);

		// Propagate properties of input hops to current hop h
		h.set_dim1(input1.get_dim1());
		h.set_rows_in_block(input1.get_rows_in_block());
		h.set_dim2(input2.get_dim2());
		h.set_cols_in_block(input2.get_cols_in_block());

	}

	private void clearLinksWithinChain ( ArrayList<Hops> operators ) throws HopsException {
		Hops op, input1, input2;
		
		for ( int i=0; i < operators.size(); i++ ) {
			op = operators.get(i);
			if ( op.getInput().size() != 2 || (i != 0 && op.getParent().size() > 1 ) ) {
				throw new HopsException(this.printErrorLocation() + "Unexpected error while applying optimization on matrix-mult chain. \n");
			}
			input1 = op.getInput().get(0);
			input2 = op.getInput().get(1);
			
			op.getInput().clear();
			input1.getParent().remove(op);
			input2.getParent().remove(op);
		}
	}

	private long [] getDimArray ( ArrayList<Hops> chain ) throws HopsException {
		// Build the array containing dimensions from all matrices in the
		// chain
		
		long dimArray[] = new long[chain.size() + 1];
		
		// check the dimensions in the matrix chain to insure all dimensions are known
		boolean shortCircuit = false;
		for (int i=0; i< chain.size(); i++){
			if (chain.get(i)._dim1 <= 0 || chain.get(i)._dim2 <= 0)
				shortCircuit = true;
		}
		if (shortCircuit){
			for (int i=0; i< dimArray.length; i++){
				dimArray[i] = 1;
			}	
			LOG.trace("short-circuit optimizeMMChain() for matrices with unknown size");
			return dimArray;
		}
		
		
		
		for (int i = 0; i < chain.size(); i++) {
			if (i == 0) {
				dimArray[i] = chain.get(i).get_dim1();
				if (dimArray[i] <= 0) {
					throw new HopsException(this.printErrorLocation() + 
							"Hops::optimizeMMChain() : Invalid Matrix Dimension: "
									+ dimArray[i]);
				}
			} else {
				if (chain.get(i - 1).get_dim2() != chain.get(i)
						.get_dim1()) {
					throw new HopsException(this.printErrorLocation() +
							"Hops::optimizeMMChain() : Matrix Dimension Mismatch");
				}
			}
			dimArray[i + 1] = chain.get(i).get_dim2();
			if (dimArray[i + 1] <= 0) {
				throw new HopsException(this.printErrorLocation() + 
						"Hops::optimizeMMChain() : Invalid Matrix Dimension: "
								+ dimArray[i + 1]);
			}
		}

		return dimArray;
	}

	private int inputCount ( Hops p, Hops h ) {
		int count = 0;
		for ( int i=0; i < p.getInput().size(); i++ )
			if ( p.getInput().get(i).equals(h) )
				count++;
		return count;
	}
	
	/**
	 * optimizeMMChain(): It optimizes the matrix multiplication chain in which
	 * the last Hop is "this". Step-1) Identify the chain (mmChain). (Step-2) clear all
	 * links among the Hops that are involved in mmChain. (Step-3) Find the
	 * optimal ordering (dynamic programming) (Step-4) Relink the hops in
	 * mmChain.
	 */

	private void optimizeMMChain() throws HopsException {
		LOG.trace("MM Chain Optimization for HOP: (" + " " + getKind() + ", " + getHopID() + ", "
					+ get_name() + ")");
		
		ArrayList<Hops> mmChain = new ArrayList<Hops>();
		ArrayList<Hops> mmOperators = new ArrayList<Hops>();
		ArrayList<Hops> tempList;

		/*
		 * Step-1: Identify the chain (mmChain) & clear all links among the Hops
		 * that are involved in mmChain.
		 */

		mmOperators.add(this);
		// Initialize mmChain with my inputs
		for (Hops hi : this.getInput()) {
			mmChain.add(hi);
		}

		// expand each Hop in mmChain to find the entire matrix multiplication
		// chain
		int i = 0;
		while (i < mmChain.size()) {

			boolean expandable = false;

			Hops h = mmChain.get(i);
			/*
			 * Check if mmChain[i] is expandable: 
			 * 1) It must be MATMULT 
			 * 2) It must not have been visited already 
			 *    (one MATMULT should get expanded only in one chain)
			 * 3) Its output should not be used in multiple places
			 *    (either within chain or outside the chain)
			 */

			if (h.getKind() == Hops.Kind.AggBinaryOp && ((AggBinaryOp) h).isMatrixMultiply()
					&& h.get_visited() != Hops.VISIT_STATUS.DONE) {
				// check if the output of "h" is used at multiple places. If yes, it can
				// not be expanded.
				if (h.getParent().size() > 1 || inputCount( (Hops) ((h.getParent().toArray())[0]), h) > 1 ) {
					expandable = false;
					break;
				}
				else 
					expandable = true;
			}

			h.set_visited(Hops.VISIT_STATUS.DONE);

			if ( !expandable ) {
				i = i + 1;
			} else {
				tempList = mmChain.get(i).getInput();
				if (tempList.size() != 2) {
					throw new HopsException(this.printErrorLocation() + "Hops::rule_OptimizeMMChain(): AggBinary must have exactly two inputs.");
				}

				// add current operator to mmOperators, and its input nodes to mmChain
				mmOperators.add(mmChain.get(i));
				mmChain.set(i, tempList.get(0));
				mmChain.add(i + 1, tempList.get(1));
			}
		}

		// print the MMChain
		if (LOG.isTraceEnabled()) {
			LOG.trace("Identified MM Chain: ");
			for (Hops h : mmChain) {
				LOG.trace("Hop " + h.get_name() + "(" + h.getKind() + ", " + h.getHopID() + ")" + " "
						+ h.get_dim1() + "x" + h.get_dim2());
			}
			LOG.trace("--End of MM Chain--");
		}

		if (mmChain.size() == 2) {
			// If the chain size is 2, then there is nothing to optimize.
			return;
		} 
		else {
			 // Step-2: clear the links among Hops within the identified chain
			clearLinksWithinChain ( mmOperators );
			
			 // Step-3: Find the optimal ordering via dynamic programming.
			
			long dimArray[] = getDimArray ( mmChain );
			
			// Invoke Dynamic Programming
			int size = mmChain.size();
			int[][] split = new int[size][size];
			split = mmChainDP(dimArray, mmChain.size());
			
			 // Step-4: Relink the hops using the optimal ordering (split[][]) found from DP.
			mmChainRelinkHops(mmOperators.get(0), 0, size - 1, mmChain, mmOperators, 1, split);
		}
	
	}

	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()) {
			StringBuilder s = new StringBuilder(""); 
			s.append(_kind + " " + getHopID() + "\n");
			s.append("  Label: " + get_name() + "; DataType: " + _dataType + "; ValueType: " + _valueType + "\n");
			s.append("  Parent: ");
			for (Hops h : getParent()) {
				s.append(h.hashCode() + "; ");
			}
			;
			s.append("\n  Input: ");
			for (Hops h : getInput()) {
				s.append(h.getHopID() + "; ");
			}
			
			s.append("\n  dims [" + _dim1 + "," + _dim2 + "] blk [" + _rows_in_block + "," + _cols_in_block + "] nnz " + _nnz);
			s.append("  MemEstimate = Out " + (_outputMemEstimate/1024/1024) + " MB, In&Out " + (_memEstimate/1024/1024) + " MB" );
			LOG.debug(s.toString());
		}
	}

	public long get_dim1() {
		return _dim1;
	}

	public void set_dim1(long dim1) {
		_dim1 = dim1;
	}

	public long get_dim2() {
		return _dim2;
	}

	public Lops get_lops() {
		return _lops;
	}

	public void set_lops(Lops lops) {
		_lops = lops;
	}
	
	public SQLLops get_sqllops() {
		return _sqllops;
	}

	public void set_sqllops(SQLLops sqllops) {
		_sqllops = sqllops;
	}

	public void set_dim2(long dim2) {
		_dim2 = dim2;
	}

	public VISIT_STATUS get_visited() {
		return _visited;
	}

	public DataType get_dataType() {
		return _dataType;
	}
	
	public void set_dataType( DataType dt ) {
		_dataType = dt;
	}

	public void set_visited(VISIT_STATUS visited) {
		_visited = visited;
	}

	public void set_name(String _name) {
		this._name = _name;
	}

	public String get_name() {
		return _name;
	}

	public ValueType get_valueType() {
		return _valueType;
	}

	// TODO BJR: I intend to remove OpOp1.MINUS, once we made the change
	public enum OpOp1 {
		MINUS, NOT, ABS, SIN, COS, TAN, SQRT, LOG, EXP, CAST_AS_SCALAR, PRINT, EIGEN, NROW, NCOL, LENGTH, ROUND, IQM, PRINT2
	}

	// Operations that require two operands
	public enum OpOp2 {
		PLUS, MINUS, MULT, DIV, MODULUS, LESS, LESSEQUAL, GREATER, GREATEREQUAL, EQUAL, NOTEQUAL, 
		MIN, MAX, AND, OR, LOG, POW, PRINT, CONCAT, QUANTILE, INTERQUANTILE, IQM, 
		CENTRALMOMENT, COVARIANCE, APPEND, INVALID
	};

	// Operations that require 3 operands
	public enum OpOp3 {
		QUANTILE, INTERQUANTILE, CTABLE, CENTRALMOMENT, COVARIANCE, INVALID 
	};
	public enum AggOp {
		SUM, MIN, MAX, TRACE, PROD, MEAN, MAXINDEX
	};

	public enum ReorgOp {
		TRANSPOSE, DIAG_V2M, DIAG_M2V
	};

	public enum ParamBuiltinOp {
		INVALID, CDF, GROUPEDAGG, RMEMPTY
	};

	/**
	 * Functions that are built in, but whose execution takes place in an
	 * external library
	 */
	public enum ExtBuiltInOp {
		EIGEN, CHOLESKY
	};

	public enum FileFormatTypes {
		TEXT, BINARY, DELIM, MM
	};

	public enum DataOpTypes {
		PERSISTENTREAD, PERSISTENTWRITE, TRANSIENTREAD, TRANSIENTWRITE, FUNCTIONOUTPUT
	};

	public enum Direction {
		RowCol, Row, Col
	};

	static public HashMap<DataOpTypes, com.ibm.bi.dml.lops.Data.OperationTypes> HopsData2Lops;
	static {
		HopsData2Lops = new HashMap<Hops.DataOpTypes, com.ibm.bi.dml.lops.Data.OperationTypes>();
		HopsData2Lops.put(DataOpTypes.PERSISTENTREAD, com.ibm.bi.dml.lops.Data.OperationTypes.READ);
		HopsData2Lops.put(DataOpTypes.PERSISTENTWRITE, com.ibm.bi.dml.lops.Data.OperationTypes.WRITE);
		HopsData2Lops.put(DataOpTypes.TRANSIENTWRITE, com.ibm.bi.dml.lops.Data.OperationTypes.WRITE);
		HopsData2Lops.put(DataOpTypes.TRANSIENTREAD, com.ibm.bi.dml.lops.Data.OperationTypes.READ);
	}

	static public HashMap<Hops.AggOp, com.ibm.bi.dml.lops.Aggregate.OperationTypes> HopsAgg2Lops;
	static {
		HopsAgg2Lops = new HashMap<Hops.AggOp, com.ibm.bi.dml.lops.Aggregate.OperationTypes>();
		HopsAgg2Lops.put(AggOp.SUM, com.ibm.bi.dml.lops.Aggregate.OperationTypes.KahanSum);
	//	HopsAgg2Lops.put(AggOp.SUM, dml.lops.Aggregate.OperationTypes.Sum);
		HopsAgg2Lops.put(AggOp.TRACE, com.ibm.bi.dml.lops.Aggregate.OperationTypes.KahanTrace);
		HopsAgg2Lops.put(AggOp.MIN, com.ibm.bi.dml.lops.Aggregate.OperationTypes.Min);
		HopsAgg2Lops.put(AggOp.MAX, com.ibm.bi.dml.lops.Aggregate.OperationTypes.Max);
		HopsAgg2Lops.put(AggOp.MAXINDEX, com.ibm.bi.dml.lops.Aggregate.OperationTypes.MaxIndex);
		HopsAgg2Lops.put(AggOp.PROD, com.ibm.bi.dml.lops.Aggregate.OperationTypes.Product);
		HopsAgg2Lops.put(AggOp.MEAN, com.ibm.bi.dml.lops.Aggregate.OperationTypes.Mean);
	}

	static public HashMap<Hops.ReorgOp, com.ibm.bi.dml.lops.Transform.OperationTypes> HopsTransf2Lops;
	static {
		HopsTransf2Lops = new HashMap<Hops.ReorgOp, com.ibm.bi.dml.lops.Transform.OperationTypes>();
		HopsTransf2Lops.put(ReorgOp.TRANSPOSE, com.ibm.bi.dml.lops.Transform.OperationTypes.Transpose);
		HopsTransf2Lops.put(ReorgOp.DIAG_V2M, com.ibm.bi.dml.lops.Transform.OperationTypes.VectortoDiagMatrix);
		//HopsTransf2Lops.put(ReorgOp.DIAG_M2V, dml.lops.Transform.OperationTypes.MatrixtoDiagVector);

	}

	static public HashMap<Hops.Direction, com.ibm.bi.dml.lops.PartialAggregate.DirectionTypes> HopsDirection2Lops;
	static {
		HopsDirection2Lops = new HashMap<Hops.Direction, com.ibm.bi.dml.lops.PartialAggregate.DirectionTypes>();
		HopsDirection2Lops.put(Direction.RowCol, com.ibm.bi.dml.lops.PartialAggregate.DirectionTypes.RowCol);
		HopsDirection2Lops.put(Direction.Col, com.ibm.bi.dml.lops.PartialAggregate.DirectionTypes.Col);
		HopsDirection2Lops.put(Direction.Row, com.ibm.bi.dml.lops.PartialAggregate.DirectionTypes.Row);

	}

	static public HashMap<Hops.OpOp2, com.ibm.bi.dml.lops.Binary.OperationTypes> HopsOpOp2LopsB;
	static {
		HopsOpOp2LopsB = new HashMap<Hops.OpOp2, com.ibm.bi.dml.lops.Binary.OperationTypes>();
		HopsOpOp2LopsB.put(OpOp2.PLUS, com.ibm.bi.dml.lops.Binary.OperationTypes.ADD);
		HopsOpOp2LopsB.put(OpOp2.MINUS, com.ibm.bi.dml.lops.Binary.OperationTypes.SUBTRACT);
		HopsOpOp2LopsB.put(OpOp2.MULT, com.ibm.bi.dml.lops.Binary.OperationTypes.MULTIPLY);
		HopsOpOp2LopsB.put(OpOp2.DIV, com.ibm.bi.dml.lops.Binary.OperationTypes.DIVIDE);
		HopsOpOp2LopsB.put(OpOp2.MODULUS, com.ibm.bi.dml.lops.Binary.OperationTypes.MODULUS);
		HopsOpOp2LopsB.put(OpOp2.LESS, com.ibm.bi.dml.lops.Binary.OperationTypes.LESS_THAN);
		HopsOpOp2LopsB.put(OpOp2.LESSEQUAL, com.ibm.bi.dml.lops.Binary.OperationTypes.LESS_THAN_OR_EQUALS);
		HopsOpOp2LopsB.put(OpOp2.GREATER, com.ibm.bi.dml.lops.Binary.OperationTypes.GREATER_THAN);
		HopsOpOp2LopsB.put(OpOp2.GREATEREQUAL, com.ibm.bi.dml.lops.Binary.OperationTypes.GREATER_THAN_OR_EQUALS);
		HopsOpOp2LopsB.put(OpOp2.EQUAL, com.ibm.bi.dml.lops.Binary.OperationTypes.EQUALS);
		HopsOpOp2LopsB.put(OpOp2.NOTEQUAL, com.ibm.bi.dml.lops.Binary.OperationTypes.NOT_EQUALS);
		HopsOpOp2LopsB.put(OpOp2.MIN, com.ibm.bi.dml.lops.Binary.OperationTypes.MIN);
		HopsOpOp2LopsB.put(OpOp2.MAX, com.ibm.bi.dml.lops.Binary.OperationTypes.MAX);
		HopsOpOp2LopsB.put(OpOp2.AND, com.ibm.bi.dml.lops.Binary.OperationTypes.OR);
		HopsOpOp2LopsB.put(OpOp2.OR, com.ibm.bi.dml.lops.Binary.OperationTypes.AND);
		HopsOpOp2LopsB.put(OpOp2.POW, com.ibm.bi.dml.lops.Binary.OperationTypes.NOTSUPPORTED);
		HopsOpOp2LopsB.put(OpOp2.LOG, com.ibm.bi.dml.lops.Binary.OperationTypes.NOTSUPPORTED);
	}

	static public HashMap<Hops.OpOp2, com.ibm.bi.dml.lops.BinaryCP.OperationTypes> HopsOpOp2LopsBS;
	static {
		HopsOpOp2LopsBS = new HashMap<Hops.OpOp2, com.ibm.bi.dml.lops.BinaryCP.OperationTypes>();
		HopsOpOp2LopsBS.put(OpOp2.PLUS, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.ADD);
		HopsOpOp2LopsBS.put(OpOp2.MINUS, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.SUBTRACT);
		HopsOpOp2LopsBS.put(OpOp2.MULT, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.MULTIPLY);
		HopsOpOp2LopsBS.put(OpOp2.DIV, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.DIVIDE);
		HopsOpOp2LopsBS.put(OpOp2.MODULUS, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.MODULUS);
		HopsOpOp2LopsBS.put(OpOp2.LESS, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.LESS_THAN);
		HopsOpOp2LopsBS.put(OpOp2.LESSEQUAL, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.LESS_THAN_OR_EQUALS);
		HopsOpOp2LopsBS.put(OpOp2.GREATER, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.GREATER_THAN);
		HopsOpOp2LopsBS.put(OpOp2.GREATEREQUAL, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.GREATER_THAN_OR_EQUALS);
		HopsOpOp2LopsBS.put(OpOp2.EQUAL, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.EQUALS);
		HopsOpOp2LopsBS.put(OpOp2.NOTEQUAL, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.NOT_EQUALS);
		HopsOpOp2LopsBS.put(OpOp2.MIN, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.MIN);
		HopsOpOp2LopsBS.put(OpOp2.MAX, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.MAX);
		HopsOpOp2LopsBS.put(OpOp2.AND, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.AND);
		HopsOpOp2LopsBS.put(OpOp2.OR, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.OR);
		HopsOpOp2LopsBS.put(OpOp2.LOG, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.LOG);
		HopsOpOp2LopsBS.put(OpOp2.POW, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.POW);
		HopsOpOp2LopsBS.put(OpOp2.PRINT, com.ibm.bi.dml.lops.BinaryCP.OperationTypes.PRINT);
	}

	static public HashMap<Hops.OpOp2, com.ibm.bi.dml.lops.Unary.OperationTypes> HopsOpOp2LopsU;
	static {
		HopsOpOp2LopsU = new HashMap<Hops.OpOp2, com.ibm.bi.dml.lops.Unary.OperationTypes>();
		HopsOpOp2LopsU.put(OpOp2.PLUS, com.ibm.bi.dml.lops.Unary.OperationTypes.ADD);
		HopsOpOp2LopsU.put(OpOp2.MINUS, com.ibm.bi.dml.lops.Unary.OperationTypes.SUBTRACT);
		HopsOpOp2LopsU.put(OpOp2.MULT, com.ibm.bi.dml.lops.Unary.OperationTypes.MULTIPLY);
		HopsOpOp2LopsU.put(OpOp2.DIV, com.ibm.bi.dml.lops.Unary.OperationTypes.DIVIDE);
		HopsOpOp2LopsU.put(OpOp2.MODULUS, com.ibm.bi.dml.lops.Unary.OperationTypes.MODULUS);
		HopsOpOp2LopsU.put(OpOp2.LESSEQUAL, com.ibm.bi.dml.lops.Unary.OperationTypes.LESS_THAN_OR_EQUALS);
		HopsOpOp2LopsU.put(OpOp2.LESS, com.ibm.bi.dml.lops.Unary.OperationTypes.LESS_THAN);
		HopsOpOp2LopsU.put(OpOp2.GREATEREQUAL, com.ibm.bi.dml.lops.Unary.OperationTypes.GREATER_THAN_OR_EQUALS);
		HopsOpOp2LopsU.put(OpOp2.GREATER, com.ibm.bi.dml.lops.Unary.OperationTypes.GREATER_THAN);
		HopsOpOp2LopsU.put(OpOp2.EQUAL, com.ibm.bi.dml.lops.Unary.OperationTypes.EQUALS);
		HopsOpOp2LopsU.put(OpOp2.NOTEQUAL, com.ibm.bi.dml.lops.Unary.OperationTypes.NOT_EQUALS);
		HopsOpOp2LopsU.put(OpOp2.AND, com.ibm.bi.dml.lops.Unary.OperationTypes.NOTSUPPORTED);
		HopsOpOp2LopsU.put(OpOp2.OR, com.ibm.bi.dml.lops.Unary.OperationTypes.NOTSUPPORTED);
		HopsOpOp2LopsU.put(OpOp2.MAX, com.ibm.bi.dml.lops.Unary.OperationTypes.MAX);
		HopsOpOp2LopsU.put(OpOp2.MIN, com.ibm.bi.dml.lops.Unary.OperationTypes.MIN);
		HopsOpOp2LopsU.put(OpOp2.LOG, com.ibm.bi.dml.lops.Unary.OperationTypes.LOG);
		HopsOpOp2LopsU.put(OpOp2.POW, com.ibm.bi.dml.lops.Unary.OperationTypes.POW);
	}

	static public HashMap<Hops.OpOp1, com.ibm.bi.dml.lops.Unary.OperationTypes> HopsOpOp1LopsU;
	static {
		HopsOpOp1LopsU = new HashMap<Hops.OpOp1, com.ibm.bi.dml.lops.Unary.OperationTypes>();
		HopsOpOp1LopsU.put(OpOp1.MINUS, com.ibm.bi.dml.lops.Unary.OperationTypes.NOTSUPPORTED);
		HopsOpOp1LopsU.put(OpOp1.NOT, com.ibm.bi.dml.lops.Unary.OperationTypes.NOT);
		HopsOpOp1LopsU.put(OpOp1.ABS, com.ibm.bi.dml.lops.Unary.OperationTypes.ABS);
		HopsOpOp1LopsU.put(OpOp1.SIN, com.ibm.bi.dml.lops.Unary.OperationTypes.SIN);
		HopsOpOp1LopsU.put(OpOp1.COS, com.ibm.bi.dml.lops.Unary.OperationTypes.COS);
		HopsOpOp1LopsU.put(OpOp1.TAN, com.ibm.bi.dml.lops.Unary.OperationTypes.TAN);
		HopsOpOp1LopsU.put(OpOp1.SQRT, com.ibm.bi.dml.lops.Unary.OperationTypes.SQRT);
		HopsOpOp1LopsU.put(OpOp1.EXP, com.ibm.bi.dml.lops.Unary.OperationTypes.EXP);
		HopsOpOp1LopsU.put(OpOp1.LOG, com.ibm.bi.dml.lops.Unary.OperationTypes.LOG);
		HopsOpOp1LopsU.put(OpOp1.ROUND, com.ibm.bi.dml.lops.Unary.OperationTypes.ROUND);
		HopsOpOp1LopsU.put(OpOp1.CAST_AS_SCALAR, com.ibm.bi.dml.lops.Unary.OperationTypes.NOTSUPPORTED);
	}

	static public HashMap<Hops.OpOp1, com.ibm.bi.dml.lops.UnaryCP.OperationTypes> HopsOpOp1LopsUS;
	static {
		HopsOpOp1LopsUS = new HashMap<Hops.OpOp1, com.ibm.bi.dml.lops.UnaryCP.OperationTypes>();
		HopsOpOp1LopsUS.put(OpOp1.MINUS, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.NOTSUPPORTED);
		HopsOpOp1LopsUS.put(OpOp1.NOT, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.NOT);
		HopsOpOp1LopsUS.put(OpOp1.ABS, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.ABS);
		HopsOpOp1LopsUS.put(OpOp1.SIN, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.SIN);
		HopsOpOp1LopsUS.put(OpOp1.COS, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.COS);
		HopsOpOp1LopsUS.put(OpOp1.TAN, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.TAN);
		HopsOpOp1LopsUS.put(OpOp1.SQRT, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.SQRT);
		HopsOpOp1LopsUS.put(OpOp1.EXP, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.EXP);
		HopsOpOp1LopsUS.put(OpOp1.LOG, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.LOG);
		HopsOpOp1LopsUS.put(OpOp1.CAST_AS_SCALAR, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.CAST_AS_SCALAR);
		HopsOpOp1LopsUS.put(OpOp1.NROW, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.NROW);
		HopsOpOp1LopsUS.put(OpOp1.NCOL, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.NCOL);
		HopsOpOp1LopsUS.put(OpOp1.LENGTH, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.LENGTH);
		HopsOpOp1LopsUS.put(OpOp1.PRINT, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.PRINT);
		HopsOpOp1LopsUS.put(OpOp1.PRINT2, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.PRINT2);
		HopsOpOp1LopsUS.put(OpOp1.ROUND, com.ibm.bi.dml.lops.UnaryCP.OperationTypes.ROUND);
	}

	static public HashMap<Hops.OpOp1, String> HopsOpOp12String;
	static {
		HopsOpOp12String = new HashMap<OpOp1, String>();	
		HopsOpOp12String.put(OpOp1.ABS, "abs");
		HopsOpOp12String.put(OpOp1.CAST_AS_SCALAR, "castAsScalar");
		HopsOpOp12String.put(OpOp1.COS, "cos");
		HopsOpOp12String.put(OpOp1.EIGEN, "eigen");
		HopsOpOp12String.put(OpOp1.EXP, "exp");
		HopsOpOp12String.put(OpOp1.IQM, "iqm");
		HopsOpOp12String.put(OpOp1.LENGTH, "length");
		HopsOpOp12String.put(OpOp1.LOG, "log");
		HopsOpOp12String.put(OpOp1.MINUS, "-");
		HopsOpOp12String.put(OpOp1.NCOL, "ncol");
		HopsOpOp12String.put(OpOp1.NOT, "!");
		HopsOpOp12String.put(OpOp1.NROW, "nrow");
		HopsOpOp12String.put(OpOp1.PRINT, "print");
		HopsOpOp12String.put(OpOp1.PRINT2, "print2");
		HopsOpOp12String.put(OpOp1.ROUND, "round");
		HopsOpOp12String.put(OpOp1.SIN, "sin");
		HopsOpOp12String.put(OpOp1.SQRT, "sqrt");
		HopsOpOp12String.put(OpOp1.TAN, "tan");
	}
	static public HashMap<Hops.ParamBuiltinOp, com.ibm.bi.dml.lops.ParameterizedBuiltin.OperationTypes> HopsParameterizedBuiltinLops;
	static {
		HopsParameterizedBuiltinLops = new HashMap<Hops.ParamBuiltinOp, com.ibm.bi.dml.lops.ParameterizedBuiltin.OperationTypes>();
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.CDF, com.ibm.bi.dml.lops.ParameterizedBuiltin.OperationTypes.CDF);
		HopsParameterizedBuiltinLops.put(ParamBuiltinOp.RMEMPTY, com.ibm.bi.dml.lops.ParameterizedBuiltin.OperationTypes.RMEMPTY);
	}

	static public HashMap<Hops.FileFormatTypes, com.ibm.bi.dml.lops.OutputParameters.Format> HopsFileFormatTypes2Lops;
	static {
		HopsFileFormatTypes2Lops = new HashMap<Hops.FileFormatTypes, com.ibm.bi.dml.lops.OutputParameters.Format>();
		HopsFileFormatTypes2Lops.put(FileFormatTypes.BINARY, com.ibm.bi.dml.lops.OutputParameters.Format.BINARY);
		HopsFileFormatTypes2Lops.put(FileFormatTypes.TEXT, com.ibm.bi.dml.lops.OutputParameters.Format.TEXT);
		
	}

	static public HashMap<Hops.OpOp2, String> HopsOpOp2String;
	static {
		HopsOpOp2String = new HashMap<Hops.OpOp2, String>();
		HopsOpOp2String.put(OpOp2.PLUS, "+");
		HopsOpOp2String.put(OpOp2.MINUS, "-");
		HopsOpOp2String.put(OpOp2.MULT, "*");
		HopsOpOp2String.put(OpOp2.DIV, "/");
		HopsOpOp2String.put(OpOp2.MIN, "min");
		HopsOpOp2String.put(OpOp2.MAX, "max");
		HopsOpOp2String.put(OpOp2.LESSEQUAL, "<=");
		HopsOpOp2String.put(OpOp2.LESS, "<");
		HopsOpOp2String.put(OpOp2.GREATEREQUAL, ">=");
		HopsOpOp2String.put(OpOp2.GREATER, ">");
		HopsOpOp2String.put(OpOp2.EQUAL, "=");
		HopsOpOp2String.put(OpOp2.NOTEQUAL, "!=");
		HopsOpOp2String.put(OpOp2.OR, "|");
		HopsOpOp2String.put(OpOp2.AND, "&");
		HopsOpOp2String.put(OpOp2.LOG, "log");
		HopsOpOp2String.put(OpOp2.POW, "^");
		HopsOpOp2String.put(OpOp2.CONCAT, "concat");
		HopsOpOp2String.put(OpOp2.INVALID, "?");
		HopsOpOp2String.put(OpOp2.QUANTILE, "quantile");
		HopsOpOp2String.put(OpOp2.INTERQUANTILE, "interquantile");
		HopsOpOp2String.put(OpOp2.IQM, "IQM");
		HopsOpOp2String.put(OpOp2.CENTRALMOMENT, "centraMoment");
		HopsOpOp2String.put(OpOp2.COVARIANCE, "cov");
		HopsOpOp2String.put(OpOp2.APPEND, "APP");
	}
	
	static public HashMap<Hops.OpOp3, String> HopsOpOp3String;
	static {
		HopsOpOp3String = new HashMap<Hops.OpOp3, String>();
		HopsOpOp3String.put(OpOp3.QUANTILE, "quantile");
		HopsOpOp3String.put(OpOp3.INTERQUANTILE, "interquantile");
		HopsOpOp3String.put(OpOp3.CTABLE, "ctable");
		HopsOpOp3String.put(OpOp3.CENTRALMOMENT, "centraMoment");
		HopsOpOp3String.put(OpOp3.COVARIANCE, "cov");
	}

	static public HashMap<Hops.Direction, String> HopsDirection2String;
	static {
		HopsDirection2String = new HashMap<Hops.Direction, String>();
		HopsDirection2String.put(Direction.RowCol, "RC");
		HopsDirection2String.put(Direction.Col, "C");
		HopsDirection2String.put(Direction.Row, "R");
	}

	static public HashMap<Hops.AggOp, String> HopsAgg2String;
	static {
		HopsAgg2String = new HashMap<Hops.AggOp, String>();
		HopsAgg2String.put(AggOp.SUM, "+");
		HopsAgg2String.put(AggOp.PROD, "*");
		HopsAgg2String.put(AggOp.MIN, "min");
		HopsAgg2String.put(AggOp.MAX, "max");
		HopsAgg2String.put(AggOp.MAXINDEX, "maxindex");
		HopsAgg2String.put(AggOp.TRACE, "trace");
		HopsAgg2String.put(AggOp.MEAN, "mean");
	}

	static public HashMap<Hops.ReorgOp, String> HopsTransf2String;
	static {
		HopsTransf2String = new HashMap<Hops.ReorgOp, String>();
		HopsTransf2String.put(ReorgOp.TRANSPOSE, "T");
	//	HopsTransf2String.put(ReorgOp.APPEND, "APP");
		HopsTransf2String.put(ReorgOp.DIAG_M2V, "diagM2V");
		HopsTransf2String.put(ReorgOp.DIAG_V2M, "diagV2M");
	}

	static public HashMap<DataOpTypes, String> HopsData2String;
	static {
		HopsData2String = new HashMap<Hops.DataOpTypes, String>();
		HopsData2String.put(DataOpTypes.PERSISTENTREAD, "PRead");
		HopsData2String.put(DataOpTypes.PERSISTENTWRITE, "PWrite");
		HopsData2String.put(DataOpTypes.TRANSIENTWRITE, "TWrite");
		HopsData2String.put(DataOpTypes.TRANSIENTREAD, "TRead");
	}
	
	public static boolean isFunction(OpOp2 op)
	{
		return op == OpOp2.MIN || op == OpOp2.MAX ||
		op == OpOp2.LOG;// || op == OpOp2.CONCAT; //concat is || in Netezza
	}
	
	public static boolean isSupported(OpOp2 op)
	{
		return op != OpOp2.INVALID && op != OpOp2.QUANTILE &&
		op != OpOp2.INTERQUANTILE && op != OpOp2.IQM;
	}
	
	public static boolean isFunction(OpOp1 op)
	{
		return op == OpOp1.SIN || op == OpOp1.TAN || op == OpOp1.COS ||
		op == OpOp1.ABS || op == OpOp1.EXP || op == OpOp1.LOG ||
		op == OpOp1.ROUND || op == OpOp1.SQRT;
	}
	
	public static boolean isBooleanOperation(OpOp2 op)
	{
		return op == OpOp2.AND || op == OpOp2.EQUAL ||
		op == OpOp2.GREATER || op == OpOp2.GREATEREQUAL ||
		op == OpOp2.LESS || op == OpOp2.LESSEQUAL ||
		op == OpOp2.OR;
	}
	
	public static ValueType getResultValueType(ValueType vt1, ValueType vt2)
	{
		if(vt1 == ValueType.STRING || vt2  == ValueType.STRING)
			return ValueType.STRING;
		else if(vt1 == ValueType.DOUBLE || vt2 == ValueType.DOUBLE)
			return ValueType.DOUBLE;
		else
			return ValueType.INT;
	}
	
	protected GENERATES determineGeneratesFlag()
	{
		//Check whether this is going to be an Insert or With
		GENERATES gen = GENERATES.SQL;
		if(this.getParent().size() > 1)
			gen = GENERATES.DML;
		else
		{
			boolean hasWriteOutput = false;
			for(Hops h : this.getParent())
				if(h instanceof DataOp)
				{
					DataOp o = ((DataOp)h);
					if(o._dataop == DataOpTypes.PERSISTENTWRITE || o._dataop == DataOpTypes.TRANSIENTWRITE)
					{
						hasWriteOutput = true;
						break;
					}
				}
				else if(h instanceof UnaryOp && ((UnaryOp)h).get_op() == OpOp1.PRINT2)
				{
					hasWriteOutput = true;
					break;
				}
			if(hasWriteOutput)
				gen = GENERATES.DML;
		}
		if(BREAKONSCALARS && this.get_dataType() == DataType.SCALAR)
			gen = GENERATES.DML;
		return gen;
	}
	
	/////////////////////////////////////
	// methods for dynamic re-compilation
	/////////////////////////////////////

	/**
	 * Indicates if dynamic recompilation is required for this hop. 
	 */
	public boolean requiresRecompile() 
	{
		return _requiresRecompile;
	}
	
	public void setRequiresRecompile()
	{
		_requiresRecompile = true;
	}
	
	public void unsetRequiresRecompile()
	{
		_requiresRecompile = false;
	}
	
	/**
	 * Update the output size information for this hop.
	 */
	public abstract void refreshSizeInformation();
	
	/**
	 * 
	 * @return
	 */
	public String constructBaseDir()
	{
		StringBuilder sb = new StringBuilder();
		sb.append( ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE) );
		sb.append( Lops.FILE_SEPARATOR );
		sb.append( Lops.PROCESS_PREFIX );
		sb.append( DMLScript.getUUID() );
		sb.append( Lops.FILE_SEPARATOR );
		sb.append( Lops.FILE_SEPARATOR );
		sb.append( ProgramConverter.CP_ROOT_THREAD_ID );
		sb.append( Lops.FILE_SEPARATOR );
	
		return sb.toString();
	}
	
	/**
	 * Clones the attributes of that and copies it over to this.
	 * 
	 * @param that
	 * @throws HopsException 
	 */
	protected void clone( Hops that, boolean withRefs ) 
		throws CloneNotSupportedException 
	{
		if( withRefs )
			throw new CloneNotSupportedException( "Hops deep copy w/ lops/inputs/parents not supported." );
		
		ID = that.ID;
		_kind = that._kind;
		_name = that._name;
		_dataType = that._dataType;
		_valueType = that._valueType;
		_visited = that._visited;
		_dim1 = that._dim1;
		_dim2 = that._dim2;
		_rows_in_block = that._rows_in_block;
		_cols_in_block = that._cols_in_block;
		_nnz = that._nnz;

		//no copy of lops (regenerated)
		_parent = new ArrayList<Hops>();
		_input = new ArrayList<Hops>();
		_lops = null;
		_sqllops = null;
		
		_etype = that._etype;
		_etypeForced = that._etypeForced;
		_outputMemEstimate = that._outputMemEstimate;
		_memEstimate = that._memEstimate;
		_processingMemEstimate = that._processingMemEstimate;
		_requiresRecompile = that._requiresRecompile;
		
		_beginLine = that._beginLine;
		_beginColumn = that._beginColumn;
		_endLine = that._endLine;
		_endColumn = that._endColumn;
	}

	public abstract Object clone() throws CloneNotSupportedException;
	
	///////////////////////////////////////////////////////////////////////////
	// store position information for Hops
	///////////////////////////////////////////////////////////////////////////
	public int _beginLine, _beginColumn;
	public int _endLine, _endColumn;
	
	public void setBeginLine(int passed)    { _beginLine = passed;   }
	public void setBeginColumn(int passed) 	{ _beginColumn = passed; }
	public void setEndLine(int passed) 		{ _endLine = passed;   }
	public void setEndColumn(int passed)	{ _endColumn = passed; }
	
	public void setAllPositions(int blp, int bcp, int elp, int ecp){
		_beginLine	 = blp; 
		_beginColumn = bcp; 
		_endLine 	 = elp;
		_endColumn 	 = ecp;
	}

	public int getBeginLine()	{ return _beginLine;   }
	public int getBeginColumn() { return _beginColumn; }
	public int getEndLine() 	{ return _endLine;   }
	public int getEndColumn()	{ return _endColumn; }
	
	public String printErrorLocation(){
		return "ERROR: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	public String printWarningLocation(){
		return "WARNING: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
} // end class
