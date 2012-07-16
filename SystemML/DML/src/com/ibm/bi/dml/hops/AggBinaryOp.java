package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.BinaryCP;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.MMCJ;
import com.ibm.bi.dml.lops.MMRJ;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
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

	private enum MMultMethod { CPMM, RMM, CP };
	
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
	
	public Lops constructLops() throws HopsException, LopsException {

		if (get_lops() == null) {
			if ( isMatrixMultiply() ) {
				ExecType et = optFindExecType();
				if ( et == ExecType.CP ) {
					BinaryCP bcp = new BinaryCP(getInput().get(0).constructLops(), 
							getInput().get(1).constructLops(), BinaryCP.OperationTypes.MATMULT, get_dataType(), get_valueType());
					bcp.getOutputParameters().setDimensions(get_dim1(), get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
					set_lops(bcp);
				}
				else if ( et == ExecType.MR ) {
				
					MMultMethod method = optFindMMultMethod ( 
							getInput().get(0).get_dim1(), getInput().get(0).get_dim2(), 
							getInput().get(0).get_rows_in_block(), getInput().get(0).get_cols_in_block(),    
							getInput().get(1).get_dim1(), getInput().get(1).get_dim2(), 
							getInput().get(1).get_rows_in_block(), getInput().get(1).get_cols_in_block());
					if ( method == MMultMethod.CPMM ) {
						MMCJ mmcj = new MMCJ(
								getInput().get(0).constructLops(), getInput().get(1)
										.constructLops(), get_dataType(), get_valueType());
						mmcj.getOutputParameters().setDimensions(get_dim1(), get_dim2(), 
								get_rows_in_block(), get_cols_in_block(), getNnz());
						
						Group grp = new Group(
								mmcj, Group.OperationTypes.Sort, get_dataType(), get_valueType());
						grp.getOutputParameters().setDimensions(get_dim1(), get_dim2(), 
								get_rows_in_block(), get_cols_in_block(), getNnz());
						
						Aggregate agg1 = new Aggregate(
								grp, HopsAgg2Lops.get(outerOp), get_dataType(), get_valueType(), ExecType.MR);
						agg1.getOutputParameters().setDimensions(get_dim1(), get_dim2(), 
								get_rows_in_block(), get_cols_in_block(), getNnz());
						
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
						set_lops(rmm);
					}
				}
			} 
			else  {
				throw new HopsException("Invalid operation aggBin(" + innerOp + "," + outerOp + ") while constructing lops.");
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
		if (get_visited() != VISIT_STATUS.DONE) {
			super.printMe();
			System.out.println("  InnerOperation: " + innerOp);
			System.out.println("  OuterOperation: " + outerOp + "\n");
			for (Hops h : getInput()) {
				h.printMe();
			}
			;
		}
		set_visited(VISIT_STATUS.DONE);
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
	protected ExecType optFindExecType() {
		
		if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE )
			return ExecType.CP;
		else if ( DMLScript.rtplatform == RUNTIME_PLATFORM.HADOOP )
			return ExecType.MR;

		if( _etype != null ) 			
			return _etype;
		
		// choose CP if the dimensions of both inputs are below Hops.CPThreshold 
		// OR if it is vector-vector inner product
		if ( (getInput().get(0).areDimsBelowThreshold() && getInput().get(1).areDimsBelowThreshold())
				|| (getInput().get(0).isVector() && getInput().get(1).isVector() && !isOuterProduct()) )
			return ExecType.CP;
		else
			return ExecType.MR;
	}
	
	/*
	 * Optimization that chooses between two methods to perform matrix multiplication on map-reduce -- CPMM or RMM.
	 * 
	 * More details on the cost-model used: refer ICDE 2011 paper. 
	 */
	private static MMultMethod optFindMMultMethod ( long m1_rows, long m1_cols, long m1_rpb, long m1_cpb, long m2_rows, long m2_cols, long m2_rpb, long m2_cpb ) {
		
		int m1_nrb = (int) Math.ceil((double)m1_rows/m1_rpb); // number of row blocks in m1
		int m2_ncb = (int) Math.ceil((double)m2_cols/m2_cpb); // number of column blocks in m2
		
		double rmm_shuffle, rmm_io, cpmm_shuffle, cpmm_io;
		rmm_shuffle = rmm_io = cpmm_shuffle = cpmm_io = 0;
		
		// TODO: we must factor in the "sparsity"
		double m1_size = m1_rows * m1_cols;
		double m2_size = m2_rows * m2_cols;
		double result_size = m1_rows * m2_cols;
		
		/* Estimate the cost of RMM */
		rmm_shuffle = (m2_ncb*m1_size) + (m1_nrb*m2_size);
		rmm_io = m1_size + m2_size + result_size;
		
		
		/* Estimate the cost of CPMM */
		int r = 5; // TODO: remove hard-coding to number of reducers
		cpmm_shuffle = m1_size + m2_size + (r*result_size);
		cpmm_io = m1_size + m2_size + result_size + (2*r*result_size);
		
		if ( cpmm_shuffle + cpmm_io < rmm_shuffle + rmm_io ) {
			//System.out.println("CPMM --> c(rmm)=" + (rmm_shuffle+rmm_io) + ", c(cpmm)=" + (cpmm_shuffle+cpmm_io) ); 
			return MMultMethod.CPMM;
		}
		else {
			//System.out.println("RMM --> c(rmm)=" + (rmm_shuffle+rmm_io) + ", c(cpmm)=" + (cpmm_shuffle+cpmm_io) ); 
			return MMultMethod.RMM;
		}
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
	
		if(this.get_sqllops() == null)
		{
			if(this.getInput().size() != 2)
				throw new HopsException("The binary aggregation hop must have two inputs");
			
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
			throw new HopsException("Aggregates only work for two matrices");
		
		boolean isvalid = Hops.isSupported(this.innerOp);
		if(!isvalid)
			throw new HopsException("This operation is not supported");
		
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
			throw new HopsException("Aggregates only work for two matrices");
		
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
				// Based on http://www.infosoft.biz/PDF/Product_Function_extends_SQL.pdf
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
		throw new HopsException("This operation is not supported");
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
}
