package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.Binary;
import com.ibm.bi.dml.lops.BinaryCP;
import com.ibm.bi.dml.lops.CentralMoment;
import com.ibm.bi.dml.lops.CoVariance;
import com.ibm.bi.dml.lops.CombineBinary;
import com.ibm.bi.dml.lops.CombineUnary;
import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.PartialAggregate;
import com.ibm.bi.dml.lops.PickByCount;
import com.ibm.bi.dml.lops.SortKeys;
import com.ibm.bi.dml.lops.Unary;
import com.ibm.bi.dml.lops.UnaryCP;
import com.ibm.bi.dml.lops.CombineBinary.OperationTypes;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
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


/* Binary (cell operations): aij + bij
 * 		Properties: 
 * 			Symbol: *, -, +, ...
 * 			2 Operands
 * 		Semantic: align indices (sort), then perform operation
 */

public class BinaryOp extends Hops {

	Hops.OpOp2 op;

	public BinaryOp(String l, DataType dt, ValueType vt, Hops.OpOp2 o,
			Hops inp1, Hops inp2) {
		super(Kind.BinaryOp, l, dt, vt);
		op = o;
		getInput().add(0, inp1);
		getInput().add(1, inp2);

		inp1.getParent().add(this);
		inp2.getParent().add(this);
	}

	public Lops constructLops() throws HopsException {

		if (get_lops() == null) {

			try {
			if (op == Hops.OpOp2.IQM) {
				ExecType et = optFindExecType();
				if ( et == ExecType.MR ) {
					CombineBinary combine = CombineBinary.constructCombineLop(
							OperationTypes.PreSort, (Lops) getInput().get(0)
									.constructLops(), (Lops) getInput().get(1)
									.constructLops(), DataType.MATRIX,
							get_valueType());
					combine.getOutputParameters().setDimensions(
							getInput().get(0).get_dim1(),
							getInput().get(0).get_dim2(),
							getInput().get(0).get_rows_in_block(),
							getInput().get(0).get_cols_in_block(), 
							getInput().get(0).getNnz());

					SortKeys sort = SortKeys.constructSortByValueLop(
							combine,
							SortKeys.OperationTypes.WithNoWeights,
							DataType.MATRIX, ValueType.DOUBLE, ExecType.MR);

					// Sort dimensions are same as the first input
					sort.getOutputParameters().setDimensions(
							getInput().get(0).get_dim1(),
							getInput().get(0).get_dim2(),
							getInput().get(0).get_rows_in_block(),
							getInput().get(0).get_cols_in_block(), 
							getInput().get(0).getNnz());

					Data lit = new Data(null, Data.OperationTypes.READ, null,
							Double.toString(0.25), DataType.SCALAR,
							get_valueType(), false);
					
					lit.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

					PickByCount pick = new PickByCount(
							sort, lit, DataType.MATRIX, get_valueType(),
							PickByCount.OperationTypes.RANGEPICK);

					pick.getOutputParameters().setDimensions(-1, -1, 
							get_rows_in_block(), get_cols_in_block(), -1);

					pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					PartialAggregate pagg = new PartialAggregate(pick,
							HopsAgg2Lops.get(Hops.AggOp.SUM),
							HopsDirection2Lops.get(Hops.Direction.RowCol),
							DataType.MATRIX, get_valueType());

					pagg.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					// Set the dimensions of PartialAggregate LOP based on the
					// direction in which aggregation is performed
					pagg.setDimensionsBasedOnDirection(get_dim1(), get_dim2(),
							get_rows_in_block(), get_cols_in_block());

					Group group1 = new Group(pagg, Group.OperationTypes.Sort,
							DataType.MATRIX, get_valueType());
					group1.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_in_block(),
							get_cols_in_block(), getNnz());
					
					group1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

					Aggregate agg1 = new Aggregate(group1, HopsAgg2Lops
							.get(Hops.AggOp.SUM), DataType.MATRIX,
							get_valueType(), ExecType.MR);
					agg1.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_in_block(),
							get_cols_in_block(), getNnz());
					agg1.setupCorrectionLocation(pagg.getCorrectionLocaion());
					
					agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

					UnaryCP unary1 = new UnaryCP(agg1, HopsOpOp1LopsUS
							.get(OpOp1.CAST_AS_SCALAR), DataType.SCALAR,
							get_valueType());
					unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
					unary1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					

					BinaryCP binScalar1 = new BinaryCP(sort, lit,
							BinaryCP.OperationTypes.IQSIZE, DataType.SCALAR,
							get_valueType());
					binScalar1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
					binScalar1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					BinaryCP binScalar2 = new BinaryCP(unary1, binScalar1,
							HopsOpOp2LopsBS.get(Hops.OpOp2.DIV),
							DataType.SCALAR, get_valueType());
					binScalar2.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
					binScalar2.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					set_lops(binScalar2);
				}
				else {
					SortKeys sort = SortKeys.constructSortByValueLop(
							getInput().get(0).constructLops(), 
							getInput().get(1).constructLops(), 
							SortKeys.OperationTypes.WithWeights, 
							getInput().get(0).get_dataType(), getInput().get(0).get_valueType(), et);
					sort.getOutputParameters().setDimensions(
							getInput().get(0).get_dim1(),
							getInput().get(0).get_dim2(), 
							getInput().get(0).get_rows_in_block(), 
							getInput().get(0).get_cols_in_block(), 
							getInput().get(0).getNnz());
					PickByCount pick = new PickByCount(
							sort,
							null,
							get_dataType(),
							get_valueType(),
							PickByCount.OperationTypes.IQM, et, true);
					pick.getOutputParameters().setDimensions(get_dim1(),
							get_dim1(), get_rows_in_block(), get_cols_in_block(), getNnz());
	
					pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					set_lops(pick);
				}

			} else if (op == Hops.OpOp2.CENTRALMOMENT) {
				ExecType et = optFindExecType();
				// The output data type is a SCALAR if central moment 
				// gets computed in CP, and it will be MATRIX otherwise.
				DataType dt = (et == ExecType.MR ? DataType.MATRIX : DataType.SCALAR );
				CentralMoment cm = new CentralMoment(getInput().get(0)
						.constructLops(), getInput().get(1).constructLops(),
						dt, get_valueType(), et);

				cm.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
				
				if ( et == ExecType.MR ) {
					cm.getOutputParameters().setDimensions(1, 1, 0, 0, -1);
					UnaryCP unary1 = new UnaryCP(cm, HopsOpOp1LopsUS
							.get(OpOp1.CAST_AS_SCALAR), get_dataType(),
							get_valueType());
					unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
					unary1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					set_lops(unary1);
				}
				else {
					//System.out.println("CM executing in CP...");
					cm.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
					set_lops(cm);
				}

			} else if (op == Hops.OpOp2.COVARIANCE) {
				ExecType et = optFindExecType();
				if ( et == ExecType.MR ) {
					// combineBinary -> CoVariance -> CastAsScalar
					CombineBinary combine = CombineBinary.constructCombineLop(
							OperationTypes.PreCovUnweighted, getInput().get(
									0).constructLops(), getInput().get(1)
									.constructLops(), DataType.MATRIX,
							get_valueType());
	
					combine.getOutputParameters().setDimensions(
							getInput().get(0).get_dim1(),
							getInput().get(0).get_dim2(),
							getInput().get(0).get_rows_in_block(),
							getInput().get(0).get_cols_in_block(), 
							getInput().get(0).getNnz());
	
					CoVariance cov = new CoVariance(combine, DataType.MATRIX,
							get_valueType(), et);
					cov.getOutputParameters().setDimensions(1, 1, 0, 0, -1);
					
					cov.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
					UnaryCP unary1 = new UnaryCP(cov, HopsOpOp1LopsUS
							.get(OpOp1.CAST_AS_SCALAR), get_dataType(),
							get_valueType());
					unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
					unary1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
					set_lops(unary1);
				}
				else {
					//System.out.println("COV executing in CP...");
					CoVariance cov = new CoVariance(
							getInput().get(0).constructLops(), 
							getInput().get(1).constructLops(), 
							get_dataType(), get_valueType(), et);
					cov.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
					cov.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					set_lops(cov);
				}

			} else if (op == Hops.OpOp2.QUANTILE
					|| op == Hops.OpOp2.INTERQUANTILE) {
				// 1st arguments needs to be a 1-dimensional matrix
				// For QUANTILE: 2nd argument is scalar or 1-dimensional matrix
				// For INTERQUANTILE: 2nd argument is always a scalar
				
				ExecType et = optFindExecType();
				if ( et == ExecType.MR ) {
					CombineUnary combine = CombineUnary.constructCombineLop(
							getInput().get(0).constructLops(),
							get_dataType(), get_valueType());
	
					SortKeys sort = SortKeys.constructSortByValueLop(
							combine, SortKeys.OperationTypes.WithNoWeights,
							DataType.MATRIX, ValueType.DOUBLE, et);
	
					combine.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
	
					// Sort dimensions are same as the first input
					sort.getOutputParameters().setDimensions(
							getInput().get(0).get_dim1(),
							getInput().get(0).get_dim2(),
							getInput().get(0).get_rows_in_block(),
							getInput().get(0).get_cols_in_block(), 
							getInput().get(0).getNnz());
	
					// If only a single quantile is computed, then "pick" operation executes in CP.
					ExecType et_pick = (getInput().get(1).get_dataType() == DataType.SCALAR ? ExecType.CP : ExecType.MR);
					PickByCount pick = new PickByCount(
							sort,
							getInput().get(1).constructLops(),
							get_dataType(),
							get_valueType(),
							(op == Hops.OpOp2.QUANTILE) ? PickByCount.OperationTypes.VALUEPICK
									: PickByCount.OperationTypes.RANGEPICK, et_pick, false);
	
					pick.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
					
					pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
					set_lops(pick);
				}
				else {
					SortKeys sort = SortKeys.constructSortByValueLop(
										getInput().get(0).constructLops(), 
										SortKeys.OperationTypes.WithNoWeights, 
										DataType.MATRIX, ValueType.DOUBLE, et );
					sort.getOutputParameters().setDimensions(
							getInput().get(0).get_dim1(),
							getInput().get(0).get_dim2(),
							getInput().get(0).get_rows_in_block(),
							getInput().get(0).get_cols_in_block(), 
							getInput().get(0).getNnz());
					PickByCount pick = new PickByCount(
							sort,
							getInput().get(1).constructLops(),
							get_dataType(),
							get_valueType(),
							(op == Hops.OpOp2.QUANTILE) ? PickByCount.OperationTypes.VALUEPICK
									: PickByCount.OperationTypes.RANGEPICK, et, true);
	
					pick.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
					
					pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
					set_lops(pick);
				}
			} else {
				/* Default behavior for BinaryOp */
				// it depends on input data types
				DataType dt1 = getInput().get(0).get_dataType();
				DataType dt2 = getInput().get(1).get_dataType();
				
				if (dt1 == dt2 && dt1 == DataType.SCALAR) {

					// Both operands scalar
					BinaryCP binScalar1 = new BinaryCP(getInput().get(0)
							.constructLops(),
							getInput().get(1).constructLops(), HopsOpOp2LopsBS
									.get(op), get_dataType(), get_valueType());
					binScalar1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
					
					binScalar1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					set_lops(binScalar1);

				} else if ((dt1 == DataType.MATRIX && dt2 == DataType.SCALAR)
						   || (dt1 == DataType.SCALAR && dt2 == DataType.MATRIX)) {

					// One operand is Matrix and the other is scalar
					ExecType et = optFindExecType();
					Unary unary1 = new Unary(getInput().get(0).constructLops(),
							getInput().get(1).constructLops(), HopsOpOp2LopsU
									.get(op), get_dataType(), get_valueType(), et);
					unary1.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_in_block(),
							get_cols_in_block(), getNnz());
					unary1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					set_lops(unary1);
					
				} else {

					// Both operands are Matrixes
					ExecType et = optFindExecType();
					if ( et == ExecType.CP ) {
						Binary binary = new Binary(getInput().get(0).constructLops(), getInput().get(1).constructLops(), HopsOpOp2LopsB.get(op),
								get_dataType(), get_valueType(), et);
						
						binary.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						binary.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(),
								get_cols_in_block(), getNnz());
						set_lops(binary);
					}
					else {
						Group group1, group2;
						Binary binary = null;
	
						group1 = group2 = null;
	
						// Both operands are Matrixes
						group1 = new Group(getInput().get(0).constructLops(),
								Group.OperationTypes.Sort, get_dataType(),
								get_valueType());
						
						group1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						group2 = new Group(getInput().get(1).constructLops(),
								Group.OperationTypes.Sort, get_dataType(),
								get_valueType());
	
						group2.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						binary = new Binary(group1, group2, HopsOpOp2LopsB.get(op),
								get_dataType(), get_valueType(), et);
						group1.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(),
								get_cols_in_block(), getNnz());
						group2.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(),
								get_cols_in_block(), getNnz());
						binary.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(),
								get_cols_in_block(), getNnz());
						
						binary.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						set_lops(binary);
					}
				}
			}
			} catch (Exception e) {
				throw new HopsException(this.printErrorLocation() + "error constructing Lops for BinaryOp Hop -- \n" + e);
			}
		}
		return get_lops();
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += "b(" + HopsOpOp2String.get(op) + ")";
		return s;
	}

	public void printMe() throws HopsException {
		if (get_visited() != VISIT_STATUS.DONE) {
			super.printMe();
			System.out.println("  Operation: " + op + "\n");
			for (Hops h : getInput()) {
				h.printMe();
			}
			;
		}
		set_visited(VISIT_STATUS.DONE);
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		if (get_sqllops() == null) {
			if (this.getInput().size() != 2)
				throw new HopsException(this.printErrorLocation() + "Binary Hop must have two inputs \n");

			GENERATES gen = determineGeneratesFlag();

			Hops hop1 = this.getInput().get(0);
			Hops hop2 = this.getInput().get(1);

			SQLLops lop1 = hop1.constructSQLLOPs();
			SQLLops lop2 = hop2.constructSQLLOPs();

			// Densification
			/*
			 * String name = ""; if(makeDense && mm) { lop2 =
			 * SQLLops.getDensifySQLLop(denseMatrix, lop2); name =
			 * lop2.get_tableName(); } else if(makeDense && ms) {
			 * if(hop1.get_dataType() == DataType.MATRIX) { lop1 =
			 * SQLLops.getDensifySQLLop(denseMatrix, lop1); name =
			 * lop1.get_tableName(); } else { lop2 =
			 * SQLLops.getDensifySQLLop(denseMatrix, lop2); name =
			 * lop2.get_tableName(); } }
			 */

			SQLLops sqllop = new SQLLops(this.get_name(), gen, lop1, lop2, this
					.get_valueType(), this.get_dataType());

			if (this.get_dataType() == DataType.SCALAR && gen == GENERATES.DML)
				sqllop.set_tableName(Lops.VARIABLE_NAME_PLACEHOLDER + sqllop.get_tableName() + Lops.VARIABLE_NAME_PLACEHOLDER);

			// String sql = this.getSQLSelectCode(hop1, hop2, makeDense, name);
			String sql = this.getSQLSelectCode(false, "");
			// String sql = this.getSQLSelect(hop1, hop2).toString(); //Does not
			// work with LinLogReg, but with GNMF

			sqllop.set_sql(sql);
			sqllop.set_properties(getProperties());
			this.set_sqllops(sqllop);
		}
		return this.get_sqllops();
	}

	private SQLLopProperties getProperties() {
		Hops hop1 = this.getInput().get(0);
	
		SQLLopProperties prop = new SQLLopProperties();
		JOINTYPE join = JOINTYPE.FULLOUTERJOIN;
		AGGREGATIONTYPE agg = AGGREGATIONTYPE.NONE;

		agg = AGGREGATIONTYPE.NONE;

		if (op == OpOp2.MULT || op == OpOp2.AND)
			join = JOINTYPE.INNERJOIN;
		else if (op == OpOp2.DIV)
			join = JOINTYPE.LEFTJOIN;

		prop.setAggType(agg);
		prop.setJoinType(join);
		prop.setOpString(hop1.get_sqllops().get_tableName() + " "
				+ Hops.HopsOpOp2String.get(op) + " "
				+ hop1.get_sqllops().get_tableName());

		return prop;
	}

	private SQLSelectStatement getSQLSelect() throws HopsException {
		Hops hop1 = this.getInput().get(0);
		Hops hop2 = this.getInput().get(1);

		String name1 = hop1.get_sqllops().get_tableName();
		String name2 = hop2.get_sqllops().get_tableName();

		if (op == OpOp2.COVARIANCE)
			return getBinaryCovarianceSQLSelect(SQLLops.addQuotes(name1),
					SQLLops.addQuotes(name2));
		if (op == OpOp2.CENTRALMOMENT) {
			String m = SQLLops.addQuotes(name1);
			String s = name2;

			if (hop2.get_sqllops().get_flag() == GENERATES.SQL)
				s = SQLLops.addQuotes(s);

			return getBinaryCentralMomentSQLSelect(m, s);
		}

		SQLSelectStatement stmt = new SQLSelectStatement();
		// Is input mixed (scalar and matrix)?
		boolean ms = hop1.get_sqllops().get_dataType() == DataType.MATRIX
				&& hop2.get_sqllops().get_dataType() == DataType.SCALAR;
		boolean sm = hop1.get_sqllops().get_dataType() == DataType.SCALAR
				&& hop2.get_sqllops().get_dataType() == DataType.MATRIX;

		// Is input two matrices?
		boolean mm = (hop1.get_sqllops().get_dataType() == DataType.MATRIX && hop2
				.get_sqllops().get_dataType() == DataType.MATRIX);
		boolean isvalid = Hops.isSupported(this.op);
		boolean isfunc = Hops.isFunction(this.op);
		boolean isMaxMin = op == OpOp2.MAX || op == OpOp2.MIN;

		if (!isvalid)
			throw new HopsException(this.printErrorLocation() + "In BinaryOp Hop, Cannot create SQL for operation "
					+ Hops.HopsOpOp2String.get(this.op));

		String operation = "";
		String opr = SQLLops.OpOp2ToString(op);

		// String concatenation happens with || in Netezza
		if (op == OpOp2.PLUS
				&& (hop1.get_sqllops().get_valueType() == ValueType.STRING || hop2
						.get_sqllops().get_valueType() == ValueType.STRING))
			opr = "||";
		else if (op == OpOp2.AND)
			opr = "AND";
		else if (op == OpOp2.OR)
			opr = "OR";

		// Matrix and scalar
		if (ms || sm) {
			String table = SQLLops.addQuotes(ms ? name1 : name2);
			String scalar = ms ? name2 : name1;

			String matval = SQLLops.ALIAS_A + "." + SQLLops.VALUECOLUMN;
			String scalval = scalar;

			if (ms && hop2.get_sqllops().get_flag() == GENERATES.SQL || sm
					&& hop1.get_sqllops().get_flag() == GENERATES.SQL) {
				scalar = SQLLops.addQuotes(scalar);
				scalval = scalar + "." + SQLLops.SCALARVALUECOLUMN;
			}

			// Check divisor for 0
			if (op == OpOp2.DIV) {
				if (ms)
					scalval = String.format("decode(%s=0,true,null,false,%s)",
							scalval, scalval);
				else
					matval = String.format("decode(%s=0,true,null,false,%s)",
							matval, matval);
			} else if (isMaxMin) {
				operation = String.format("decode(%s,true,%s,false,%s)",
						scalval
								+ (op == OpOp2.MAX && ms || op == OpOp2.MIN
										&& sm ? "<" : ">") + matval, scalval,
						matval);
			}
			// Normal case
			else {
				if (ms) {
					if (!isfunc)
						operation = scalval + " " + opr + " " + matval;
					else
						operation = Hops.HopsOpOp2String.get(this.op) + "("
								+ scalval + ", " + matval + ")";
				} else {
					if (!isfunc)
						operation = matval + " " + opr + " " + scalval;
					else
						operation = Hops.HopsOpOp2String.get(this.op) + "("
								+ matval + ", " + scalval + ")";
				}
			}
			// "SELECT alias_a.row AS row, alias_a.col AS col, %s as value FROM \"%s\" alias_a %s";
			stmt.getColumns().add("alias_a.row AS row");
			stmt.getColumns().add("alias_a.col AS col");
			stmt.getColumns().add(operation + " AS value");

			if (ms && hop2.get_sqllops().get_flag() == GENERATES.SQL || sm
					&& hop1.get_sqllops().get_flag() == GENERATES.SQL) {
				SQLJoin crossJoin = new SQLJoin();
				crossJoin.setJoinType(JOINTYPE.CROSSJOIN);

				crossJoin.setTable1(new SQLTableReference(table,
						SQLLops.ALIAS_A));
				crossJoin.setTable2(new SQLTableReference(scalar, ""));
				stmt.setTable(crossJoin);
			} else
				stmt.setTable(new SQLTableReference(table, SQLLops.ALIAS_A));
		} else if (mm) {
			if (isMaxMin)
				operation = String.format(SQLLops.BINARYOP_MAXMIN_PART,
						(op == OpOp2.MAX ? ">" : "<"));
			else if (!isfunc) {
				if (op == OpOp2.DIV)
					operation = SQLLops.BINARYMATRIXDIV_PART;
				else
					operation = String.format(SQLLops.BINARYOP_PART, opr);
			} else
				operation = String.format(SQLLops.FUNCTIONOP_PART,
						Hops.HopsOpOp2String.get(this.op));

			String op1 = SQLLops.addQuotes(hop1.get_sqllops().get_tableName());
			String op2 = SQLLops.addQuotes(hop2.get_sqllops().get_tableName());

			// Determine the right join to use
			JOINTYPE jt = JOINTYPE.FULLOUTERJOIN;
			if (op == OpOp2.MULT || op == OpOp2.AND)
				jt = JOINTYPE.INNERJOIN;
			else if (op == OpOp2.DIV)
				jt = JOINTYPE.LEFTJOIN;

			if (Hops.isBooleanOperation(op))
				operation = "decode(" + operation + ",true,1,false,0)";

			stmt.getColumns().add("coalesce(alias_a.row, alias_b.row) AS row");
			stmt.getColumns().add("coalesce(alias_a.col, alias_b.col) AS col");
			stmt.getColumns().add(operation + " AS " + SQLLops.VALUECOLUMN);

			SQLJoin join = new SQLJoin();
			join.setJoinType(jt);
			join.setTable1(new SQLTableReference(op1, SQLLops.ALIAS_A));
			join.setTable2(new SQLTableReference(op2, SQLLops.ALIAS_B));

			join.getConditions().add(
					new SQLCondition("alias_a.col = alias_b.col"));
			join.getConditions().add(
					new SQLCondition(BOOLOP.AND, "alias_a.row = alias_b.row"));
			stmt.setTable(join);
		}
		// Two scalars
		else {
			String o1 = name1;
			String o2 = name2;

			if (hop1.get_sqllops().get_flag() == GENERATES.SQL)
				o1 = SQLLops.addQuotes(o1) + "." + SQLLops.SCALARVALUECOLUMN;

			if (hop2.get_sqllops().get_flag() == GENERATES.SQL)
				o2 = SQLLops.addQuotes(o2) + "." + SQLLops.SCALARVALUECOLUMN;

			if (op == OpOp2.DIV)
				o2 = String.format("decode(%s=0,true,null,false,%s)", o2, o2);

			if (isMaxMin)
				operation = String.format(SQLLops.BINARYMAXMINOP, o1
						+ (op == OpOp2.MAX ? ">" : "<") + o2, o1, o2);
			else if (!isfunc)
				operation = o1 + " " + opr + " " + o2;
			else
				operation = String.format("%s(%s, %s)", Hops.HopsOpOp2String
						.get(this.op), o1, o2);

			stmt.getColumns().add(
					operation + " AS " + SQLLops.SCALARVALUECOLUMN);

			boolean twoSQL = hop1.get_sqllops().get_flag() == GENERATES.SQL
					&& hop2.get_sqllops().get_flag() == GENERATES.SQL;

			if (hop1.get_sqllops().get_flag() == GENERATES.SQL) {
				name1 = SQLLops.addQuotes(name1);
				if (!twoSQL)
					stmt.setTable(new SQLTableReference(name1));
			}
			if (hop2.get_sqllops().get_flag() == GENERATES.SQL) {
				name2 = SQLLops.addQuotes(name2);
				if (!twoSQL)
					stmt.setTable(new SQLTableReference(name2));
				else {
					SQLJoin join = new SQLJoin();
					join.setTable1(new SQLTableReference(name1));
					join.setTable2(new SQLTableReference(name2));

					stmt.setTable(join);
				}
			}
		}

		return stmt;
	}

	private SQLSelectStatement getBinaryCentralMomentSQLSelect(String m,
			String s) {
		// "SELECT (1.0 / COUNT(*)) * SUM((val - (SELECT AVG(val) FROM %s))^%s) FROM %s"
		SQLSelectStatement stmt = new SQLSelectStatement();
		stmt
				.getColumns()
				.add(
						String
								.format(
										"(1.0 / COUNT(*)) * SUM((value - (SELECT AVG(value) FROM %s))^%s)",
										m, s));
		stmt.setTable(new SQLTableReference(m, ""));
		return stmt;
	}

	private SQLSelectStatement getBinaryCovarianceSQLSelect(String m1, String m2) {
		SQLSelectStatement stmt = new SQLSelectStatement();
		// "SELECT (1.0 / COUNT(*)) * SUM((alias_a.value - (SELECT AVG(val) FROM
		// %s)) * (alias_b.value - (SELECT AVG(val) FROM %s)))
		// FROM %s alias_a JOIN %s alias_b ON alias_a.col = alias_b.col AND
		// alias_a.row = alias_b.row";

		stmt
				.getColumns()
				.add(
						String
								.format(
										"(1.0 / COUNT(*)) * SUM((alias_a.value - (SELECT AVG(val) FROM %s)) * (alias_b.value - (SELECT AVG(val) FROM %s)))",
										m1, m2));

		SQLJoin join = new SQLJoin();
		join.setJoinType(JOINTYPE.INNERJOIN);
		join.setTable1(new SQLTableReference(m1, SQLLops.ALIAS_A));
		join.setTable2(new SQLTableReference(m2, SQLLops.ALIAS_B));
		join.getConditions().add(new SQLCondition("alias_a.col = alias_b.col"));
		join.getConditions().add(new SQLCondition("alias_a.row = alias_b.row"));

		return stmt;
	}

	private String getSQLSelectCode(boolean makeDense, String denseName)
			throws HopsException {
		Hops hop1 = this.getInput().get(0);
		Hops hop2 = this.getInput().get(1);

		String sql = null;

		// Is input mixed (scalar and matrix)?
		boolean ms = ((hop1.get_sqllops().get_dataType() == DataType.MATRIX && hop2
				.get_sqllops().get_dataType() == DataType.SCALAR) || (hop1
				.get_sqllops().get_dataType() == DataType.SCALAR && hop2
				.get_sqllops().get_dataType() == DataType.MATRIX));

		// Is input two matrices?
		boolean mm = (hop1.get_sqllops().get_dataType() == DataType.MATRIX && hop2
				.get_sqllops().get_dataType() == DataType.MATRIX);

		// Can only process operations where output value is a matrix
		// if(mm || ms)
		// {
		// min, max, log, concat, invalid, quantile, interquantile, iqm,
		// centralMoment and covariance cannot be done that way
		boolean isvalid = Hops.isSupported(this.op);

		// But min, max and log can still be done
		boolean isfunc = Hops.isFunction(this.op);
		boolean isMaxMin = op == OpOp2.MAX || op == OpOp2.MIN;

		/*
		 * In the following section we have to differentiate between: - Two
		 * scalars - First input is a scalar - Second input is a scalar - Two
		 * matrices ----------------------------------------- Makes 3 possible
		 * paths
		 */

		// No unsupported operations?
		if (isvalid) {
			String operation = "";
			String opr = SQLLops.OpOp2ToString(op);

			boolean coalesce = op != OpOp2.DIV;

			// String concatenation happens with || in Netezza
			if (op == OpOp2.PLUS
					&& (hop1.get_sqllops().get_valueType() == ValueType.STRING || hop2
							.get_sqllops().get_valueType() == ValueType.STRING)) {
				opr = "||";
			} else if (op == OpOp2.AND)
				opr = "AND";
			else if (op == OpOp2.OR)
				opr = "OR";

			// Here the first parameter must be a matrix and the other a scalar
			// There is a special SELECT for the binary central moment
			if (op == OpOp2.COVARIANCE) {
				String m1 = SQLLops.addQuotes(hop1.get_sqllops()
						.get_tableName());
				String m2 = SQLLops.addQuotes(hop2.get_sqllops()
						.get_tableName());

				sql = String.format(SQLLops.BINCOVARIANCE, m1, m2, m1, m2);
			}
			// Here the first parameter must be a matrix and the other a scalar
			// There is a special SELECT for the binary central moment
			if (op == OpOp2.CENTRALMOMENT) {
				String m = SQLLops
						.addQuotes(hop1.get_sqllops().get_tableName());
				String s = hop2.get_sqllops().get_tableName();

				if (hop2.get_sqllops().get_flag() == GENERATES.SQL)
					s = SQLLops.addQuotes(s);

				sql = String.format(SQLLops.BINCENTRALMOMENT, m, s, m);
			}

			// If one input is a matrix and the other a scalar...
			else if (ms) {
				String a = null;
				String tbl = "";// The string for the FROM part

				// First value is a scalar (path 1)
				if (hop1.get_sqllops().get_dataType() == DataType.SCALAR) {
					// Switch s and a
					String s = hop1.get_sqllops().get_tableName();
					a = hop2.get_sqllops().get_tableName();

					if (hop1.get_sqllops().get_flag() == GENERATES.SQL)
						s = SQLLops.addQuotes(s) + "."
								+ SQLLops.SCALARVALUECOLUMN;

					String aval = SQLLops.ALIAS_A + "." + SQLLops.VALUECOLUMN;

					// Check divisor for 0
					if (op == OpOp2.DIV) {
						aval = String.format("decode(%s=0,true,null,false,%s)",
								aval, aval);
					}
					if (isMaxMin) {
						operation = String
								.format(SQLLops.BINARYMAXMINOP, s
										+ (op == OpOp2.MAX ? ">" : "<") + aval,
										s, aval);
					} else if (coalesce) {
						String avalcoal = SQLLops.addCoalesce(aval);
						String scoal = SQLLops.addCoalesce(s);

						if (!isfunc)
							operation = scoal + " " + opr + " " + avalcoal;
						else
							operation = Hops.HopsOpOp2String.get(this.op) + "("
									+ scoal + ", " + avalcoal + ")";
					} else {
						if (!isfunc)
							operation = s + " " + opr + " " + aval;
						else
							operation = Hops.HopsOpOp2String.get(this.op) + "("
									+ s + ", " + aval + ")";
					}
					if (hop1.get_sqllops().get_flag() == GENERATES.SQL)
						tbl = ", "
								+ SQLLops.addQuotes(hop1.get_sqllops()
										.get_tableName());
				}
				// Second value is a scalar (path 2)
				else {
					String s = hop2.get_sqllops().get_tableName(); // The scalar
					a = hop1.get_sqllops().get_tableName(); // The input table

					if (hop2.get_sqllops().get_flag() == GENERATES.SQL)
						s = SQLLops.addQuotes(s) + "."
								+ SQLLops.SCALARVALUECOLUMN;
					String aval = SQLLops.ALIAS_A + "." + SQLLops.VALUECOLUMN;

					// Check divisor for 0
					if (op == OpOp2.DIV) {
						s = String.format("decode(%s=0,true,null,false,%s)", s,
								s);
					}
					if (isMaxMin) {
						operation = String
								.format(SQLLops.BINARYMAXMINOP, s
										+ (op == OpOp2.MAX ? ">" : "<") + aval,
										s, aval);
					} else if (coalesce) {
						String avalcoal = SQLLops.addCoalesce(aval);
						String scoal = SQLLops.addCoalesce(s);
						if (!isfunc)
							operation = avalcoal + " " + opr + " " + scoal;
						else
							operation = Hops.HopsOpOp2String.get(this.op)
									+ avalcoal + ", " + scoal + " )";
					} else {
						if (!isfunc)
							operation = aval + " " + opr + " " + s;
						else
							operation = Hops.HopsOpOp2String.get(this.op) + "("
									+ aval + ", " + s + ")";
					}

					if (hop2.get_sqllops().get_flag() == GENERATES.SQL)
						tbl = ", "
								+ SQLLops.addQuotes(hop2.get_sqllops()
										.get_tableName());
				}

				if (Hops.isBooleanOperation(op))
					operation = "decode(" + operation + ",true,1,false,0)";

				sql = String.format(SQLLops.MATRIXSCALAROP, operation,
						makeDense ? denseName : a, tbl);
			}
			// Else if input is two matrices (path 3)
			else if (mm) {
				if (isMaxMin)
					operation = String.format(SQLLops.BINARYOP_MAXMIN_PART,
							(op == OpOp2.MAX ? ">" : "<"));
				else if (!isfunc) {
					if (op == OpOp2.DIV) {
						operation = SQLLops.BINARYMATRIXDIV_PART;
					} else if (coalesce)
						operation = String.format(SQLLops.BINARYOP_PART_COAL,
								opr);
					else
						operation = String.format(SQLLops.BINARYOP_PART, opr);
				} else
					operation = String.format(SQLLops.FUNCTIONOP_PART,
							Hops.HopsOpOp2String.get(this.op));

				String op1 = hop1.get_sqllops().get_tableName();
				String op2 = makeDense ? denseName : hop2.get_sqllops()
						.get_tableName();

				// Determine the right join to use
				String join = SQLLops.FULLOUTERJOIN;
				if (op == OpOp2.MULT || op == OpOp2.AND)
					join = SQLLops.JOIN;
				else if (op == OpOp2.DIV) {
					join = SQLLops.LEFTJOIN;
				}

				if (Hops.isBooleanOperation(op))
					operation = "decode(" + operation + ",true,1,false,0)";

				sql = String.format(SQLLops.MATRIXMATRIXOP, operation, op1,
						join, op2);
			}
			// If input is two scalars
			else {
				String o1 = hop1.get_sqllops().get_tableName();
				String o2 = hop2.get_sqllops().get_tableName();

				if (hop1.get_sqllops().get_flag() == GENERATES.SQL)
					o1 = SQLLops.addQuotes(o1) + "."
							+ SQLLops.SCALARVALUECOLUMN;

				if (hop2.get_sqllops().get_flag() == GENERATES.SQL)
					o2 = SQLLops.addQuotes(o2) + "."
							+ SQLLops.SCALARVALUECOLUMN;

				if (op == OpOp2.DIV)
					o2 = String.format("decode(%s=0,true,null,false,%s)", o2,
							o2);

				// This is done to avoid an error for something like 10^-6,
				// which does not work.
				// Instead it must be 10^(-6)
				if (op == OpOp2.POW)
					o2 = "(" + o2 + ")";
				if (isMaxMin)
					operation = String.format(SQLLops.BINARYMAXMINOP, o1
							+ (op == OpOp2.MAX ? ">" : "<") + o2, o1, o2);
				else if (!isfunc)
					operation = o1 + " " + opr + " " + o2;
				else
					operation = String.format("%s(%s, %s)",
							Hops.HopsOpOp2String.get(this.op), o1, o2);

				String table = "";
				if (hop1.get_sqllops().get_flag() == GENERATES.SQL)
					table = SQLLops.addQuotes(hop1.get_sqllops()
							.get_tableName());

				if (hop2.get_sqllops().get_flag() == GENERATES.SQL) {
					if (table.length() > 0)
						table += ", ";
					table += SQLLops.addQuotes(hop2.get_sqllops()
							.get_tableName());
				}

				// if(Hops.isBooleanOperation(op))
				// operation = "decode(" + operation + ",true,1,false,0)";

				if (table.length() > 0)
					sql = String.format(SQLLops.SIMPLESCALARSELECTFROM,
							operation, table);
				else
					sql = String.format(SQLLops.SIMPLESCALARSELECT, operation);
			}
		} else
			throw new HopsException(this.printErrorLocation() + "In BinaryOp Hop, Cannot create SQL for operation "
					+ Hops.HopsOpOp2String.get(this.op));
		return sql;
	}

	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}
	
	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();
		
		if( _etypeForced != null ) 			
			_etype = _etypeForced;
		else
		{
			_etype = null;
			DataType dt1 = getInput().get(0).get_dataType();
			DataType dt2 = getInput().get(1).get_dataType();
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
				_etype = ExecType.MR;
		}
		
		return _etype;
	}
}
