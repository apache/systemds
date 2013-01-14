package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.hops.OptimizerUtils.OptimizationType;
import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.BinaryCP;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.MMCJ;
import com.ibm.bi.dml.lops.MMRJ;
import com.ibm.bi.dml.lops.MMTSJ;
import com.ibm.bi.dml.lops.PartialMVMult;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.sql.sqllops.SQLCondition;
import com.ibm.bi.dml.sql.sqllops.SQLJoin;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLSelectStatement;
import com.ibm.bi.dml.sql.sqllops.SQLTableReference;
import com.ibm.bi.dml.sql.sqllops.SQLCondition.BOOLOP;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LopsException;


/* Aggregate binary (cell operations): Sum (aij + bij)
 * 		Properties: 
 * 			Inner Symbol: *, -, +, ...
 * 			Outer Symbol: +, min, max, ...
 * 			2 Operands
 * 	
 * 		Semantic: generate indices, align, cross-operate, generate indices, align, aggregate
 */

public class AggBinaryOp extends Hops {

	OpOp2 innerOp;
	AggOp outerOp;

	private enum MMultMethod { CPMM, RMM, DIST_MVMULT, TSMM, CP };
	
	public AggBinaryOp(String l, DataType dt, ValueType vt, OpOp2 innOp,
			AggOp outOp, Hops in1, Hops in2) {
		super(Kind.AggBinaryOp, l, dt, vt);
		innerOp = innOp;
		outerOp = outOp;
		getInput().add(0, in1);
		getInput().add(1, in2);
		in1.getParent().add(this);
		in2.getParent().add(this);
	}
	
	public boolean isMatrixMultiply () {
		return ( this.innerOp == OpOp2.MULT && this.outerOp == AggOp.SUM );			
	}
	
	/**
	 * Returns true if the operation is a matrix-vector or vector-matrix multiplication.
	 * 
	 * @return
	 */
	private boolean isMatrixVectorMultiply() {
		if ( //(getInput().get(0).isVector() && !getInput().get(1).isVector()) || 
				(!getInput().get(0).isVector() && getInput().get(1).isVector()) )
			return true;
		return false;
	}
	
	/**
	 * NOTE: overestimated mem in case of transpose-identity matmult, but 3/2 at worst
	 *       and existing mem estimate advantageous in terms of consistency hops/lops 
	 */
	public Lops constructLops() throws HopsException, LopsException {

		if (get_lops() == null) {
			if ( isMatrixMultiply() ) {
				ExecType et = optFindExecType();
				MMTSJType mmtsj = checkTransposeSelf();
				
				if ( et == ExecType.CP ) {
					System.out.println("Method = CP");
					Lops matmultCP = null;
					if( mmtsj == MMTSJType.NONE ) {
						matmultCP = new BinaryCP(getInput().get(0).constructLops(),getInput().get(1).constructLops(), 
												 BinaryCP.OperationTypes.MATMULT, get_dataType(), get_valueType());
					}
					else {
						matmultCP = new MMTSJ(getInput().get((mmtsj==MMTSJType.LEFT)?1:0).constructLops(),
								              get_dataType(), get_valueType(),et, mmtsj);
					}
					
					matmultCP.getOutputParameters().setDimensions(get_dim1(), get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
					matmultCP.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					set_lops(matmultCP);
				}
				else if ( et == ExecType.MR ) {
				
					MMultMethod method = optFindMMultMethod ( 
								getInput().get(0).get_dim1(), getInput().get(0).get_dim2(), 
								getInput().get(0).get_rows_in_block(), getInput().get(0).get_cols_in_block(),    
								getInput().get(1).get_dim1(), getInput().get(1).get_dim2(), 
								getInput().get(1).get_rows_in_block(), getInput().get(1).get_cols_in_block(),
								mmtsj);
					// System.out.println("Method = " + method);
					
					if ( method == MMultMethod.DIST_MVMULT) {
						PartialMVMult mvmult = new PartialMVMult(getInput().get(0).constructLops(), getInput().get(1).constructLops(), get_dataType(), get_valueType());
						Group grp = new Group(mvmult, Group.OperationTypes.Sort, get_dataType(), get_valueType());
						Aggregate agg1 = new Aggregate(grp, HopsAgg2Lops.get(outerOp), get_dataType(), get_valueType(), ExecType.MR);
						
						mvmult.getOutputParameters().setDimensions(get_dim1(), get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
						grp.getOutputParameters().setDimensions(get_dim1(), get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
						agg1.getOutputParameters().setDimensions(get_dim1(), get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
						
						agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						// aggregation uses kahanSum but the inputs do not have correction values
						agg1.setupCorrectionLocation(CorrectionLocationType.NONE);  
						
						set_lops(agg1);
					}
					else if ( method == MMultMethod.CPMM ) {
						MMCJ mmcj = new MMCJ(
								getInput().get(0).constructLops(), getInput().get(1)
										.constructLops(), get_dataType(), get_valueType());
						mmcj.getOutputParameters().setDimensions(get_dim1(), get_dim2(), 
								get_rows_in_block(), get_cols_in_block(), getNnz());
						
						mmcj.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						Group grp = new Group(
								mmcj, Group.OperationTypes.Sort, get_dataType(), get_valueType());
						grp.getOutputParameters().setDimensions(get_dim1(), get_dim2(), 
								get_rows_in_block(), get_cols_in_block(), getNnz());
						
						grp.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						Aggregate agg1 = new Aggregate(
								grp, HopsAgg2Lops.get(outerOp), get_dataType(), get_valueType(), ExecType.MR);
						agg1.getOutputParameters().setDimensions(get_dim1(), get_dim2(), 
								get_rows_in_block(), get_cols_in_block(), getNnz());
						
						agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						// aggregation uses kahanSum but the inputs do not have correction values
						agg1.setupCorrectionLocation(CorrectionLocationType.NONE);  
						
						set_lops(agg1);
					}
					else if (method == MMultMethod.RMM ) {
						MMRJ rmm = new MMRJ(
								getInput().get(0).constructLops(), getInput().get(1)
								.constructLops(), get_dataType(), get_valueType());
						
						rmm.getOutputParameters().setDimensions(get_dim1(), get_dim2(),
								get_rows_in_block(), get_cols_in_block(), getNnz());
						
						rmm.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						set_lops(rmm);
					}
					else if( method == MMultMethod.TSMM )
					{
						MMTSJ tsmm = new MMTSJ(getInput().get((mmtsj==MMTSJType.LEFT)?1:0).constructLops(),
								              get_dataType(), get_valueType(),et, mmtsj);
						tsmm.getOutputParameters().setDimensions(get_dim1(), get_dim2(),
								get_rows_in_block(), get_cols_in_block(), getNnz());
						tsmm.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						Aggregate agg1 = new Aggregate(
								tsmm, HopsAgg2Lops.get(outerOp), get_dataType(), get_valueType(), ExecType.MR);
						agg1.getOutputParameters().setDimensions(get_dim1(), get_dim2(), 
								get_rows_in_block(), get_cols_in_block(), getNnz());
						agg1.setupCorrectionLocation(CorrectionLocationType.NONE); // aggregation uses kahanSum but the inputs do not have correction values
						agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						set_lops(agg1);
					}
				}
			} 
			else  {
				throw new HopsException(this.printErrorLocation() + "Invalid operation in AggBinary Hop, aggBin(" + innerOp + "," + outerOp + ") while constructing lops.");
			}
		}
		return get_lops();
	}
	
	@Override
	public String getOpString() {
		String s = new String("");
		s += "a(" + HopsAgg2String.get(outerOp) + HopsOpOp2String.get(innerOp)+")";
		return s;
	}

	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (get_visited() != VISIT_STATUS.DONE) {
				super.printMe();
				LOG.debug("  InnerOperation: " + innerOp);
				LOG.debug("  OuterOperation: " + outerOp);
				for (Hops h : getInput()) {
					h.printMe();
				}
				;
			}
			set_visited(VISIT_STATUS.DONE);
		}
	}

	private boolean isOuterProduct() {
		if ( getInput().get(0).isVector() && getInput().get(1).isVector() ) {
			if ( getInput().get(0).get_dim1() == 1 && getInput().get(0).get_dim1() > 1
					&& getInput().get(1).get_dim1() > 1 && getInput().get(1).get_dim2() == 1 )
				return true;
			else
				return false;
		}
		return false;
	}
	
	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}
	
	@Override
	public double computeMemEstimate() {
		
		if (dimsKnown() && isMatrixMultiply()) {
			Hops input1 = getInput().get(0);
			Hops input2 = getInput().get(1);
			double outputSparsity = OptimizerUtils.matMultSparsity( input1.getSparsity(), input2.getSparsity(), 
																	input1.get_dim1(), input1.get_dim2(), input2.get_dim2());

			_outputMemEstimate = OptimizerUtils.estimateSize(get_dim1(), get_dim2(), outputSparsity);
		} else {
			_outputMemEstimate = OptimizerUtils.DEFAULT_SIZE;
		}
		
		_memEstimate = getInputOutputSize();
		
		return _memEstimate;
	}

	@Override
	protected ExecType optFindExecType() {
		
		checkAndSetForcedPlatform();

		if( _etypeForced != null ) 			
		{
			_etype = _etypeForced;
		}
		else 
		{
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown() )
				setRequiresRecompile();
			
			if ( OptimizerUtils.getOptType() == OptimizationType.MEMORY_BASED ) {
				_etype = findExecTypeByMemEstimate();
			}
			// choose CP if the dimensions of both inputs are below Hops.CPThreshold 
			// OR if it is vector-vector inner product
			else if ( (getInput().get(0).areDimsBelowThreshold() && getInput().get(1).areDimsBelowThreshold())
						|| (getInput().get(0).isVector() && getInput().get(1).isVector() && !isOuterProduct()) )
			{
				_etype = ExecType.CP;
			}
			else
			{
				_etype = ExecType.MR;
			}
		}
		return _etype;
	}
	
	private MMTSJType checkTransposeSelf()
	{
		MMTSJType ret = MMTSJType.NONE;
		
		Hops in1 = getInput().get(0);
		Hops in2 = getInput().get(1);
		
		if(    in1 instanceof com.ibm.bi.dml.hops.ReorgOp 
			&& ((com.ibm.bi.dml.hops.ReorgOp)in1).op == Hops.ReorgOp.TRANSPOSE 
			&& in1.getInput().get(0) == in2 )
		{
			ret = MMTSJType.LEFT;
		}
		
		if(    in2 instanceof com.ibm.bi.dml.hops.ReorgOp 
			&& ((com.ibm.bi.dml.hops.ReorgOp)in2).op == Hops.ReorgOp.TRANSPOSE 
			&& in2.getInput().get(0) == in1 )
		{
			ret = MMTSJType.RIGHT;
		}
		
		
		return ret;
	}
	
	/*
	 * Optimization that chooses between two methods to perform matrix multiplication on map-reduce -- CPMM or RMM.
	 * 
	 * More details on the cost-model used: refer ICDE 2011 paper. 
	 */
	private static MMultMethod optFindMMultMethod ( long m1_rows, long m1_cols, long m1_rpb, long m1_cpb, long m2_rows, long m2_cols, long m2_rpb, long m2_cpb, MMTSJType mmtsj ) {
		
		// If transpose self pattern and result is single block:
		// use specialized TSMM method (always better than generic jobs)
		if(    ( mmtsj == MMTSJType.LEFT && m2_cols <= m2_cpb )
			|| ( mmtsj == MMTSJType.RIGHT && m1_rows <= m1_rpb ) )
		{
			//return MMultMethod.RMM;
			return MMultMethod.TSMM;
		}

		if ( m2_cols == 1 ) {
			// matrix-vector multiplication. 
			// Choose DIST_MVMULT if the "dense" vector fits in memory.
			//double vec_size = OptimizerUtils.estimateSize(m2_rows, m2_cols, 1.0);
			//if ( vec_size < 0.9 * getMemBudget(false) )
				return MMultMethod.DIST_MVMULT;
		}
		
		// If the dimensions are unknown at compilation time, 
		// simply assume the worst-case scenario and produce the 
		// most robust plan -- which is CPMM
		if ( m1_rows == -1 || m1_cols == -1 || m2_rows == -1 || m2_cols == -1 )
			return MMultMethod.CPMM;

		int m1_nrb = (int) Math.ceil((double)m1_rows/m1_rpb); // number of row blocks in m1
		int m1_ncb = (int) Math.ceil((double)m1_cols/m1_cpb); // number of column blocks in m1
		int m2_ncb = (int) Math.ceil((double)m2_cols/m2_cpb); // number of column blocks in m2

		// TODO: we must factor in the "sparsity"
		double m1_size = m1_rows * m1_cols;
		double m2_size = m2_rows * m2_cols;
		double result_size = m1_rows * m2_cols;

		int numReducers = OptimizerUtils.getNumReducers(false);
		
		/* Estimate the cost of RMM */
		// RMM phase 1
		double rmm_shuffle = (m2_ncb*m1_size) + (m1_nrb*m2_size);
		double rmm_io = m1_size + m2_size + result_size;
		double rmm_nred = Math.min( m1_nrb * m2_ncb, //max used reducers 
				                    numReducers); //available reducers
		// RMM total costs
		double rmm_costs = (rmm_shuffle + rmm_io) / rmm_nred;
		
		/* Estimate the cost of CPMM */
		// CPMM phase 1
		double cpmm_shuffle1 = m1_size + m2_size;
		double cpmm_nred1 = Math.min( m1_ncb, //max used reducers 
                                      numReducers); //available reducers		
		double cpmm_io1 = m1_size + m2_size + cpmm_nred1 * result_size;
		// CPMM phase 2
		double cpmm_shuffle2 = cpmm_nred1 * result_size;
		double cpmm_io2 = cpmm_nred1 * result_size + result_size;			
		double cpmm_nred2 = Math.min( m1_nrb * m2_ncb, //max used reducers 
                                      numReducers); //available reducers		
		// CPMM total costs
		double cpmm_costs =  (cpmm_shuffle1+cpmm_io1)/cpmm_nred1  //cpmm phase1
		                    +(cpmm_shuffle2+cpmm_io2)/cpmm_nred2; //cpmm phase2
		
		//final mmult method decision 
		if ( cpmm_costs < rmm_costs ) 
			return MMultMethod.CPMM;
		else 
			return MMultMethod.RMM;
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
	
		if(this.get_sqllops() == null)
		{
			if(this.getInput().size() != 2)
				throw new HopsException(this.printErrorLocation() + "In AggBinary Hop, The binary aggregation hop must have two inputs");
			
			//Check whether this is going to be an Insert or With
			GENERATES gen = determineGeneratesFlag();
			
			Hops hop1 = this.getInput().get(0);
			Hops hop2 = this.getInput().get(1);
			
			
			if(this.isMatrixMultiply())
			{
				SQLLops sqllop = getMatrixMultSQLLOP(gen);
				sqllop.set_properties(getProperties(hop1, hop2));
				this.set_sqllops(sqllop);
			}
			else
			{
				SQLLops sqllop = new SQLLops(this.get_name(),
										gen,
										hop1.constructSQLLOPs(),
										hop2.constructSQLLOPs(),
										this.get_valueType(), this.get_dataType());
	
				String sql = getSQLSelectCode(hop1, hop2);
			
				sqllop.set_sql(sql);
				sqllop.set_properties(getProperties(hop1, hop2));
			
				this.set_sqllops(sqllop);
			}
			this.set_visited(VISIT_STATUS.DONE);
		}
		return this.get_sqllops();
	}
	
	private SQLLopProperties getProperties(Hops hop1, Hops hop2)
	{
		SQLLopProperties prop = new SQLLopProperties();
		JOINTYPE join = JOINTYPE.FULLOUTERJOIN;
		AGGREGATIONTYPE agg = AGGREGATIONTYPE.NONE;
		
		if(innerOp == OpOp2.MULT || innerOp == OpOp2.AND || outerOp == AggOp.PROD)
			join = JOINTYPE.INNERJOIN;
		else if(innerOp == OpOp2.DIV)
			join = JOINTYPE.LEFTJOIN;
		
		//TODO: PROD
		if(outerOp == AggOp.SUM || outerOp == AggOp.TRACE)
			agg = AGGREGATIONTYPE.SUM;
		else if(outerOp == AggOp.MAX)
			agg = AGGREGATIONTYPE.MAX;
		else if(outerOp == AggOp.MIN)
			agg = AGGREGATIONTYPE.MIN;
		
		prop.setAggType(agg);
		prop.setJoinType(join);
		prop.setOpString(Hops.HopsAgg2String.get(outerOp) + "(" 
				+ hop1.get_sqllops().get_tableName() + " "
				+ Hops.HopsOpOp2String.get(innerOp) + " "
				+ hop1.get_sqllops().get_tableName() + ")");
		return prop;
	}
	
	private SQLSelectStatement getSQLSelect(Hops hop1, Hops hop2) throws HopsException
	{
		if(!(hop1.get_sqllops().get_dataType() == DataType.MATRIX && hop2.get_sqllops().get_dataType() == DataType.MATRIX))
			throw new HopsException(this.printErrorLocation() + "In AggBinary Hop, Aggregates only work for two matrices");
		
		boolean isvalid = Hops.isSupported(this.innerOp);
		if(!isvalid)
			throw new HopsException(this.printErrorLocation() + "In AggBinary Hop, This operation is not supported for SQL Select");
		
		boolean isfunc = Hops.isFunction(this.innerOp);
		
		String inner_opr = SQLLops.OpOp2ToString(innerOp);

		SQLSelectStatement stmt = new SQLSelectStatement();

		JOINTYPE t = JOINTYPE.FULLOUTERJOIN;
		if(innerOp == OpOp2.MULT || innerOp == OpOp2.AND)
			t = JOINTYPE.INNERJOIN;
		else if(innerOp == OpOp2.DIV)
			t = JOINTYPE.LEFTJOIN;
		
		SQLJoin join = new SQLJoin();
		join.setJoinType(t);
		join.setTable1(new SQLTableReference(SQLLops.addQuotes(hop1.get_sqllops().get_tableName()), SQLLops.ALIAS_A));
		join.setTable2(new SQLTableReference(SQLLops.addQuotes(hop2.get_sqllops().get_tableName()), SQLLops.ALIAS_B));

		stmt.setTable(join);
		
		String inner = isfunc ? String.format(SQLLops.FUNCTIONOP_PART, inner_opr) :
				String.format(SQLLops.BINARYOP_PART, inner_opr);
		
		if(this.outerOp == AggOp.TRACE)
		{
			join.getConditions().add(new SQLCondition("alias_a.row = alias_a.col"));
			join.getConditions().add(new SQLCondition(BOOLOP.AND, "alias_a.row = alias_b.col"));
			join.getConditions().add(new SQLCondition(BOOLOP.AND, "alias_a.col = alias_b.row"));
			
			stmt.getColumns().add("coalesce(alias_a.row, alias_b.row) AS row");
			stmt.getColumns().add("coalesce(alias_a.col, alias_b.col) AS col");
			stmt.getColumns().add("SUM(" + inner + ")");
		}
		else if(this.outerOp == AggOp.SUM)
		{
			join.getConditions().add(new SQLCondition("alias_a.col = alias_b.row"));
			stmt.getColumns().add("alias_a.row AS row");
			stmt.getColumns().add("alias_b.col AS col");
			stmt.getColumns().add("SUM(" + inner + ")");
			
			stmt.getGroupBys().add("alias_a.row");
			stmt.getGroupBys().add("alias_b.col");
		}
		else
		{
			String outer = Hops.HopsAgg2String.get(this.outerOp);

			join.getConditions().add(new SQLCondition("alias_a.col = alias_b.row"));
			stmt.getColumns().add("alias_a.row AS row");
			stmt.getColumns().add("alias_b.col AS col");
			stmt.getColumns().add(outer);
			
			stmt.getGroupBys().add("alias_a.row");
			stmt.getGroupBys().add("alias_b.col");
		}

		return stmt;
	}
	
	private String getSQLSelectCode(Hops hop1, Hops hop2) throws HopsException
	{
		if(!(hop1.get_sqllops().get_dataType() == DataType.MATRIX && hop2.get_sqllops().get_dataType() == DataType.MATRIX))
			throw new HopsException(this.printErrorLocation() + "in AggBinary Hop, error in getSQLSelectCode() -- Aggregates only work for two matrices");
		
		//min, max, log, quantile, interquantile and iqm cannot be done that way
		boolean isvalid = Hops.isSupported(this.innerOp);

		//But min, max and log can still be done
		boolean isfunc = Hops.isFunction(this.innerOp);
		
		String inner_opr = SQLLops.OpOp2ToString(innerOp);
		
		if(isvalid)
		{
			//String for the inner operation
			String inner = isfunc ? String.format(SQLLops.FUNCTIONOP_PART, inner_opr)
					:
					String.format(SQLLops.BINARYOP_PART, inner_opr);
			
			String sql = null;
			String join = SQLLops.FULLOUTERJOIN;
			if(innerOp == OpOp2.MULT || innerOp == OpOp2.AND)
				join = SQLLops.JOIN;
			else if(innerOp == OpOp2.DIV)
				join = SQLLops.LEFTJOIN;
			
			if(this.outerOp == AggOp.PROD)
			{
				// This is only a temporary solution.
				// Idea is that ln(x1 * x2 * ... * xn) = ln(x1) + ln(x2) + ... + ln(xn)
				// So that x1 * x2 * ... * xn = exp( ln(x1) + ln(x2) + ... + ln(xn) )
				// Which is EXP(SUM(LN(v)))
				
				sql = String.format(SQLLops.BINARYPROD,
						inner, hop1.get_sqllops().get_tableName(), hop2.get_sqllops().get_tableName());
			}
			//Special case for trace because it needs a special SELECT
			else if(this.outerOp == AggOp.TRACE)
			{
				sql = String.format(SQLLops.AGGTRACEOP, inner, hop1.get_sqllops().get_tableName(), join, hop2.get_sqllops().get_tableName());
			}
			//Should be handled before
			else if(this.outerOp == AggOp.SUM)
			{
				//sql = String.format(SQLLops.AGGSUMOP, inner, hop1.get_sqllops().get_tableName(), join, hop2.get_sqllops().get_tableName());
				//sql = getMatrixMultSQLString(inner, hop1.get_sqllops().get_tableName(), hop2.get_sqllops().get_tableName());
			}
			//Here the outerOp is just appended, it can only be min or max
			else
			{
				String outer = Hops.HopsAgg2String.get(this.outerOp);
				sql = String.format(SQLLops.AGGBINOP, outer, inner, 
					hop1.get_sqllops().get_tableName(), join, hop2.get_sqllops().get_tableName());
			}
			return sql;
		}
		throw new HopsException(this.printErrorLocation() + "In AggBinary Hop, error in getSQLSelectCode() -- This operation is not supported");
	}
	
	private SQLLops getPart1SQLLop(String operation, String op1, String op2, long size, SQLLops input1, SQLLops input2)
	{		
		String where = SQLLops.ALIAS_A + ".col <= " + size
		+ " AND " + SQLLops.ALIAS_B + ".row <= " + size;
		String sql = String.format(SQLLops.SPLITAGGSUMOP, operation, op1, SQLLops.JOIN, op2, where);
		SQLLops lop = new SQLLops("part1_" + this.getHopID(), GENERATES.SQL, input1, input2, ValueType.DOUBLE, DataType.MATRIX);
		SQLLopProperties prop = new SQLLopProperties();
		prop.setAggType(AGGREGATIONTYPE.SUM);
		prop.setJoinType(JOINTYPE.INNERJOIN);
		prop.setOpString("Part 1 of matrix mult");
		lop.set_properties(prop);
		lop.set_sql(sql);
		return lop;
	}
	
	private SQLLops getPart2SQLLop(String operation, String op1, String op2, long size, SQLLops input1, SQLLops input2)
	{
		String where = SQLLops.ALIAS_A + ".col > " + size
		+ " AND " + SQLLops.ALIAS_B + ".row > " + size;
		String sql = String.format(SQLLops.SPLITAGGSUMOP, operation, op1, SQLLops.JOIN, op2, where);
		SQLLops lop = new SQLLops("part2_" + this.getHopID(), GENERATES.SQL, input1, input2, ValueType.DOUBLE, DataType.MATRIX);
		SQLLopProperties prop = new SQLLopProperties();
		prop.setAggType(AGGREGATIONTYPE.SUM);
		prop.setJoinType(JOINTYPE.INNERJOIN);
		prop.setOpString("Part 2 of matrix mult");
		lop.set_properties(prop);
		lop.set_sql(sql);
		return lop;
	}
	
	private SQLLops getMatrixMultSQLLOP(GENERATES flag) throws HopsException
	{
		Hops hop1 = this.getInput().get(0);
		Hops hop2 = this.getInput().get(1);
		
		boolean m_large = hop1.get_dim1() > SQLLops.HMATRIXSPLIT;
		boolean k_large = hop1.get_dim2() > SQLLops.VMATRIXSPLIT;
		boolean n_large = hop2.get_dim2() > SQLLops.HMATRIXSPLIT;
		
		String name = this.get_name() + "_" + this.getHopID();
		
		String inner_opr = SQLLops.OpOp2ToString(innerOp);
		String operation = String.format(SQLLops.BINARYOP_PART, inner_opr);
		
		SQLLops input1 = hop1.constructSQLLOPs();
		SQLLops input2 = hop2.constructSQLLOPs();
		
		String i1 = input1.get_tableName();
		String i2 = input2.get_tableName();
		
		if(!SPLITLARGEMATRIXMULT || (!m_large && !k_large && !n_large))
		{
			String sql = String.format(SQLLops.AGGSUMOP, operation, i1, SQLLops.JOIN, i2);
			SQLLops lop = new SQLLops(name, flag, hop1.constructSQLLOPs(), hop2.constructSQLLOPs(), ValueType.DOUBLE, DataType.MATRIX);
			lop.set_sql(sql);
			return lop;
		}
		else if(m_large)
		{
			StringBuilder sb = new StringBuilder();
			//Split first matrix horizontally

			long total = 0;
			for(long s = SQLLops.HMATRIXSPLIT; s <= hop1.get_dim1(); s += SQLLops.HMATRIXSPLIT)
			{
				String where = SQLLops.ALIAS_A + ".row BETWEEN " + (s - SQLLops.HMATRIXSPLIT + 1) + " AND " + s;
				sb.append(String.format(SQLLops.SPLITAGGSUMOP, operation, i1, SQLLops.JOIN, i2, where));
				
				total = s;
				if(total < hop1.get_dim1())
					sb.append(" \r\nUNION ALL \r\n");
			}
			if(total < hop1.get_dim1())
			{
				String where = SQLLops.ALIAS_A + ".row BETWEEN " + (total + 1) + " AND " + hop1.get_dim1();
				sb.append(String.format(SQLLops.SPLITAGGSUMOP, operation, i1, SQLLops.JOIN, i2, where));
			}
			String sql = sb.toString();
			SQLLops lop = new SQLLops(name, flag, hop1.get_sqllops(), hop2.get_sqllops(), ValueType.DOUBLE, DataType.MATRIX);
			lop.set_sql(sql);
			return lop;
		}
		else if(k_large)
		{
			//The parts are both DML and have the same input, so it cannot be SQL even though in the HOPs DAg it might have only
			// one output
			/*if(input1.get_flag() == GENERATES.SQL)
				input1.set_flag(GENERATES.DML);
			if(input2.get_flag() == GENERATES.SQL)
				input2.set_flag(GENERATES.DML);
			*/
			SQLLops h1 = getPart1SQLLop(operation, i1, i2, hop1.get_dim2() / 2, input1, input2);
			SQLLops h2 = getPart2SQLLop(operation, i1, i2, hop1.get_dim2() / 2, input1, input2);
			
			String p1 = SQLLops.addQuotes(h1.get_tableName());
			String p2 = SQLLops.addQuotes(h2.get_tableName());
			
			String sql = "SELECT coalesce(" + p1 + ".row, " + p2 + ".row) AS row, coalesce(" + p1 + ".col, " + p2 + ".col) AS col, "
				+ "coalesce(" + p1 + ".value,0) + coalesce(" + p2 + ".value,0) AS value FROM " + p1 + " FULL OUTER JOIN " + p2
				+ " ON " + p1 + ".row = " + p2 + ".row AND " + p1 + ".col = " + p2 + ".col";

			SQLLops lop = new SQLLops(name, flag, h1, h2, ValueType.DOUBLE, DataType.MATRIX);
			lop.set_sql(sql);
			return lop;
		}
		else
		{
			String sql = String.format(SQLLops.AGGSUMOP, operation, i1, SQLLops.JOIN, i2);
			SQLLops lop = new SQLLops(name, flag, hop1.get_sqllops(), hop2.get_sqllops(), ValueType.DOUBLE, DataType.MATRIX);
			lop.set_sql(sql);
			return lop;
		}
	}
	
	private String getMatrixMultSQLString(String operation, String op1, String op2)
	{
		Hops hop1 = this.getInput().get(0);
		Hops hop2 = this.getInput().get(1);
		
		boolean m_large = hop1.get_dim1() > SQLLops.HMATRIXSPLIT;
		boolean k_large = hop1.get_dim2() > SQLLops.VMATRIXSPLIT;
		boolean n_large = hop2.get_dim2() > SQLLops.HMATRIXSPLIT;
		
		if(!SPLITLARGEMATRIXMULT || (!m_large && !k_large && !n_large))
			return String.format(SQLLops.AGGSUMOP, operation, op1, SQLLops.JOIN, op2);
		else
		{
			StringBuilder sb = new StringBuilder();
			//Split first matrix horizontally
			if(m_large)
			{
				long total = 0;
				for(long s = SQLLops.HMATRIXSPLIT; s <= hop1.get_dim1(); s += SQLLops.HMATRIXSPLIT)
				{
					String where = SQLLops.ALIAS_A + ".row BETWEEN " + (s - SQLLops.HMATRIXSPLIT + 1) + " AND " + s;
					sb.append(String.format(SQLLops.SPLITAGGSUMOP, operation, op1, SQLLops.JOIN, op2, where));
					
					total = s;
					if(total < hop1.get_dim1())
						sb.append(" \r\nUNION ALL \r\n");
				}
				if(total < hop1.get_dim1())
				{
					String where = SQLLops.ALIAS_A + ".row BETWEEN " + (total + 1) + " AND " + hop1.get_dim1();
					sb.append(String.format(SQLLops.SPLITAGGSUMOP, operation, op1, SQLLops.JOIN, op2, where));
				}
				return sb.toString();
			}
			//Split first matrix vertically and second matrix horizontally
			else if(k_large)
			{
				long middle = hop1.get_dim2() / 2;
				
				String where1 = SQLLops.ALIAS_A + ".col <= " + middle
				+ " AND " + SQLLops.ALIAS_B + ".row <= " + middle;
				
				String where2 = SQLLops.ALIAS_A + ".col > " + middle
				+ " AND " + SQLLops.ALIAS_B + ".row > " + middle;
				
				sb.append("\r\nWITH part1 AS ( " + String.format(SQLLops.SPLITAGGSUMOP, operation, op1, SQLLops.JOIN, op2, where1) + "),\r\n");						
				sb.append("part2 AS ( " + String.format(SQLLops.SPLITAGGSUMOP, operation, op1, SQLLops.JOIN, op2, where2) + ")\r\n");
				sb.append("SELECT coalesce(part1.row, part2.row) AS row, coalesce(part1.col, part2.col) AS col, "
						+ "coalesce(part1.value,0) + coalesce(part2.value,0) AS value FROM part1 FULL OUTER JOIN part2 "
						+ "ON part1.row = part2.row AND part1.col = part2.col");
				
				return sb.toString();
				//TODO split
				//return String.format(SQLLops.AGGSUMOP, operation, op1, join, op2);
			}
			//Split second matrix vertically
			else
			{
				//TODO split
				return String.format(SQLLops.AGGSUMOP, operation, op1, SQLLops.JOIN, op2);
			}
		}
	}
	
/*	public void refreshDims()
	{
		//TODO
		Hops input1 = getInput().get(0);
		Hops input2 = getInput().get(1);
		
		if( isMatrixMultiply() )
		{
				set_dim1(input1.get_dim1());
				set_dim2(input2.get_dim2());
		}
		
		input1.set_visited(VISIT_STATUS.NOTVISITED);
		input2.set_visited(VISIT_STATUS.NOTVISITED);
		
		refreshMemEstimates();
		
	}*/
	
	@Override
	public void refreshSizeInformation()
	{
		Hops input1 = getInput().get(0);
		Hops input2 = getInput().get(1);
		
		if( isMatrixMultiply() )
		{
			set_dim1(input1.get_dim1());
			set_dim2(input2.get_dim2());
		}
	}
}
