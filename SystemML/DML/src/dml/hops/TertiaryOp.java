package dml.hops;

import dml.lops.Aggregate;
import dml.lops.CentralMoment;
import dml.lops.CoVariance;
import dml.lops.CombineBinary;
import dml.lops.CombineTertiary;
import dml.lops.DoubleVal;
import dml.lops.Group;
import dml.lops.IndexPair;
import dml.lops.Lops;
import dml.lops.PickByCount;
import dml.lops.ReBlock;
import dml.lops.SortKeys;
import dml.lops.Tertiary;
import dml.lops.UnaryCP;
import dml.lops.CombineBinary.OperationTypes;
import dml.lops.LopProperties.ExecType;
import dml.lops.PartialAggregate.CorrectionLocationType;
import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.sql.sqllops.ISQLSelect;
import dml.sql.sqllops.SQLCondition;
import dml.sql.sqllops.SQLJoin;
import dml.sql.sqllops.SQLLopProperties;
import dml.sql.sqllops.SQLLops;
import dml.sql.sqllops.SQLSelectStatement;
import dml.sql.sqllops.SQLTableReference;
import dml.sql.sqllops.SQLUnion;
import dml.sql.sqllops.SQLCondition.BOOLOP;
import dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import dml.sql.sqllops.SQLLops.GENERATES;
import dml.sql.sqllops.SQLUnion.UNIONTYPE;
import dml.utils.HopsException;

/* Primary use cases for now, are
 * 		quantile (<n-1-matrix>, <n-1-matrix>, <literal>):      quantile (A, w, 0.5)
 * 		quantile (<n-1-matrix>, <n-1-matrix>, <scalar>):       quantile (A, w, s)
 * 		interquantile (<n-1-matrix>, <n-1-matrix>, <scalar>):  interquantile (A, w, s)
 * 
 * Keep in mind, that we also have binaries for it w/o weights.
 * 	quantile (A, 0.5)
 * 	quantile (A, s)
 * 	interquantile (A, s)
 *
 */

public class TertiaryOp extends Hops {

	Hops.OpOp3 op;

	public TertiaryOp(String l, DataType dt, ValueType vt, Hops.OpOp3 o,
			Hops inp1, Hops inp2, Hops inp3) {
		super(Hops.Kind.TertiaryOp, l, dt, vt);
		op = o;
		getInput().add(0, inp1);
		getInput().add(1, inp2);
		getInput().add(2, inp3);
		inp1.getParent().add(this);
		inp2.getParent().add(this);
		inp3.getParent().add(this);
	}

	public Lops constructLops() throws HopsException {

		if (get_lops() == null) {
			if (op == OpOp3.CENTRALMOMENT) {
				CombineBinary combine = CombineBinary.constructCombineLop(
						OperationTypes.PreCentralMoment, 
						(Lops) getInput().get(0).constructLops(), 
						(Lops) getInput().get(1).constructLops(), 
						DataType.MATRIX, get_valueType());
				CentralMoment cm = new CentralMoment(combine, (Lops) getInput()
						.get(2).constructLops(), DataType.MATRIX,
						get_valueType());
				UnaryCP unary1 = new UnaryCP(cm, HopsOpOp1LopsUS
						.get(OpOp1.CAST_AS_SCALAR), get_dataType(),
						get_valueType());
				set_lops(unary1);

			} else if (op == Hops.OpOp3.COVARIANCE) {
				// combineTertiary -> CoVariance -> CastAsScalar
				CombineTertiary combine = CombineTertiary
						.constructCombineLop(
								dml.lops.CombineTertiary.OperationTypes.PreCovWeighted,
								(Lops) getInput().get(0).constructLops(),
								(Lops) getInput().get(1).constructLops(),
								(Lops) getInput().get(2).constructLops(),
								DataType.MATRIX, get_valueType());

				combine.getOutputParameters().setDimensions(
						getInput().get(0).get_dim1(),
						getInput().get(0).get_dim2(),
						getInput().get(0).get_rows_per_block(),
						getInput().get(0).get_cols_per_block());

				CoVariance cov = new CoVariance(
						combine, DataType.MATRIX, get_valueType());

				UnaryCP unary1 = new UnaryCP(
						cov, HopsOpOp1LopsUS.get(OpOp1.CAST_AS_SCALAR),
						get_dataType(), get_valueType());
				set_lops(unary1);

			} else if (op == OpOp3.QUANTILE || op == OpOp3.INTERQUANTILE) {
				CombineBinary combine = CombineBinary
						.constructCombineLop(
								OperationTypes.PreSort,
								(Lops) getInput().get(0).constructLops(),
								(Lops) getInput().get(1).constructLops(),
								DataType.MATRIX, get_valueType());

				SortKeys sort = SortKeys
						.constructSortByValueLop(
								(Lops) combine,
								SortKeys.OperationTypes.WithWeights,
								DataType.MATRIX, get_valueType());

				PickByCount<IndexPair, DoubleVal, IndexPair, DoubleVal> pick = new PickByCount<IndexPair, DoubleVal, IndexPair, DoubleVal>(
						sort,
						getInput().get(2).constructLops(),
						get_dataType(),
						get_valueType(),
						(op == Hops.OpOp3.QUANTILE) ? PickByCount.OperationTypes.VALUEPICK
								: PickByCount.OperationTypes.RANGEPICK);

				combine.getOutputParameters().setDimensions(
						getInput().get(0).get_dim1(),
						getInput().get(0).get_dim2(), 1, 1);
				sort.getOutputParameters().setDimensions(
						getInput().get(0).get_dim1(),
						getInput().get(0).get_dim2(), 1, 1);
				pick.getOutputParameters().setDimensions(get_dim1(),
						get_dim1(), get_rows_per_block(), get_cols_per_block());

				set_lops(pick);

			} else if (op == OpOp3.CTABLE) {
				/*
				 * We must handle three different cases: case1 : all three
				 * inputs are vectors (e.g., F=ctable(A,B,W)) case2 : two
				 * vectors and one scalar (e.g., F=ctable(A,B)) case3 : one
				 * vector and two scalars (e.g., F=ctable(A))
				 */

				Group group1, group2, group3, group4;

				group1 = group2 = group3 = group4 = null;

				// identify the particular case
				int type = 1; // F=ctable(A,B,W)
				DataType dt2 = getInput().get(1).get_dataType(); // data type of
				// 2nd input
				DataType dt3 = getInput().get(2).get_dataType(); // data type of
				// 3rd input

				if (dt2 == DataType.MATRIX && dt3 == DataType.SCALAR) {
					type = 2; // F = ctable(A,B) or F = ctable(A,B,1)
				} else if (dt2 == DataType.SCALAR && dt3 == DataType.SCALAR) {
					type = 3; // F=ctable(A,1) or F = ctable(A,1,1)
				} else if (dt2 == DataType.SCALAR && dt3 == DataType.MATRIX) {
					type = 4; // F=ctable(A,1,W)
				}

				group1 = new Group(
						getInput().get(0).constructLops(),
						Group.OperationTypes.Sort, get_dataType(),
						get_valueType());
				group1.getOutputParameters().setDimensions(get_dim1(),
						get_dim2(), get_rows_per_block(), get_cols_per_block());

				Tertiary tertiary = null;
				// create "group" lops for MATRIX inputs
				switch (type) {
				case 1:
					// F = ctable(A,B,W)
					group2 = new Group(
							getInput().get(1).constructLops(),
							Group.OperationTypes.Sort, get_dataType(),
							get_valueType());
					group2.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_per_block(),
							get_cols_per_block());
					group3 = new Group(
							getInput().get(2).constructLops(),
							Group.OperationTypes.Sort, get_dataType(),
							get_valueType());
					group3.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_per_block(),
							get_cols_per_block());

					tertiary = new Tertiary(
							group1, group2, group3,
							Tertiary.OperationTypes.CTABLE_TRANSFORM,
							get_dataType(), get_valueType());
					break;

				case 2:
					// F = ctable(A,B) or F = ctable(A,B,1)
					group2 = new Group(
							getInput().get(1).constructLops(),
							Group.OperationTypes.Sort, get_dataType(),
							get_valueType());
					group2.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_per_block(),
							get_cols_per_block());
					tertiary = new Tertiary(
							group1,
							group2,
							getInput().get(2).constructLops(),
							Tertiary.OperationTypes.CTABLE_TRANSFORM_SCALAR_WEIGHT,
							get_dataType(), get_valueType());
					break;
				case 3:
					// F=ctable(A,1) or F = ctable(A,1,1)
					tertiary = new Tertiary(
							group1, getInput().get(1).constructLops(),
							getInput().get(2).constructLops(),
							Tertiary.OperationTypes.CTABLE_TRANSFORM_HISTOGRAM,
							get_dataType(), get_valueType());
					break;
				case 4:
					// F=ctable(A,1,W)
					group3 = new Group(
							getInput().get(2).constructLops(),
							Group.OperationTypes.Sort, get_dataType(),
							get_valueType());
					group3.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_per_block(),
							get_cols_per_block());
					tertiary = new Tertiary(
							group1,
							getInput().get(1).constructLops(),
							group3,
							Tertiary.OperationTypes.CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM,
							get_dataType(), get_valueType());
					break;
				}

				// output dimensions are not known at compilation time
				tertiary.getOutputParameters().setDimensions(-1, -1, -1, -1);

				group4 = new Group(
						tertiary, Group.OperationTypes.Sort, get_dataType(),
						get_valueType());
				group4.getOutputParameters().setDimensions(-1, -1, -1, -1);

				Aggregate agg1 = new Aggregate(
						group4, HopsAgg2Lops.get(AggOp.SUM), get_dataType(),
						get_valueType(), ExecType.MR);
				agg1.getOutputParameters().setDimensions(-1, -1, -1, -1);

				// kahamSum is used for aggreagtion but inputs do not have
				// correction values
				agg1.setupCorrectionLocation(CorrectionLocationType.NONE);

				// set_lops(agg1);

				ReBlock reblock = null;
				try {
					reblock = new ReBlock(
							agg1, (long) get_rows_per_block(),
							(long) get_cols_per_block(), get_dataType(),
							get_valueType());
				} catch (Exception e) {
					throw new HopsException(e);
				}
				reblock.getOutputParameters().setDimensions(-1, -1,
						get_rows_per_block(), get_cols_per_block());

				set_lops(reblock);
			} else if (op == OpOp3.SPEARMAN) {
				Group group1, group2, group3, group4;

				group1 = group2 = group3 = group4 = null;

				group1 = new Group(
						getInput().get(0).constructLops(),
						Group.OperationTypes.Sort, DataType.MATRIX,
						get_valueType());
				group1.getOutputParameters().setDimensions(
						getInput().get(0).get_dim1(),
						getInput().get(0).get_dim2(),
						getInput().get(0).get_rows_per_block(),
						getInput().get(0).get_cols_per_block());

				group2 = new Group(
						getInput().get(1).constructLops(),
						Group.OperationTypes.Sort, DataType.MATRIX,
						get_valueType());
				group2.getOutputParameters().setDimensions(
						getInput().get(1).get_dim1(),
						getInput().get(1).get_dim2(),
						getInput().get(1).get_rows_per_block(),
						getInput().get(1).get_cols_per_block());

				Tertiary tertiary = null;
				if (getInput().get(2).get_dataType() == DataType.MATRIX) {
					group3 = new Group(
							getInput().get(2).constructLops(),
							Group.OperationTypes.Sort, DataType.MATRIX,
							get_valueType());
					group3.getOutputParameters().setDimensions(
							getInput().get(2).get_dim1(),
							getInput().get(2).get_dim2(),
							getInput().get(2).get_rows_per_block(),
							getInput().get(2).get_cols_per_block());

					tertiary = new Tertiary(
							group1, group2, group3,
							Tertiary.OperationTypes.CTABLE_TRANSFORM,
							DataType.MATRIX, get_valueType());
				} else if (getInput().get(2).get_dataType() == DataType.SCALAR) {
					tertiary = new Tertiary(
							group1,
							group2,
							getInput().get(2).constructLops(),
							Tertiary.OperationTypes.CTABLE_TRANSFORM_SCALAR_WEIGHT,
							DataType.MATRIX, get_valueType());
				}
				tertiary.getOutputParameters().setDimensions(-1, -1, -1, -1);

				group4 = new Group(
						tertiary, Group.OperationTypes.Sort, get_dataType(),
						get_valueType());
				group4.getOutputParameters().setDimensions(-1, -1, -1, -1);

				Aggregate agg1 = new Aggregate(
						group4, HopsAgg2Lops.get(AggOp.SUM), DataType.MATRIX,
						get_valueType(), ExecType.MR);
				agg1.getOutputParameters().setDimensions(-1, -1, -1, -1);

				// kahamSum is used for aggreagtion but inputs do not have
				// correction values
				agg1.setupCorrectionLocation(CorrectionLocationType.NONE);

				UnaryCP unary1 = new UnaryCP(
						agg1, UnaryCP.OperationTypes.SPEARMANHELPER,
						get_dataType(), get_valueType());
				unary1.getOutputParameters().setDimensions(get_dim1(),
						get_dim2(), get_rows_per_block(), get_cols_per_block());

				set_lops(unary1);

			} else {
				throw new HopsException("Incorrect TertiaryOp (" + op
						+ ") while constructing LOPs!");
			}
		}
		return get_lops();
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += "t(" + HopsOpOp3String.get(op) + ")";
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
		if (this.op == OpOp3.CTABLE) {
			if (this.getInput().size() != 3)
				throw new HopsException("A tertiary hop must have three inputs");

			GENERATES gen = determineGeneratesFlag();

			Hops hop1 = this.getInput().get(0);
			Hops hop2 = this.getInput().get(1);
			Hops hop3 = this.getInput().get(2);

			hop3.constructSQLLOPs();

			SQLLops maxsqllop = new SQLLops("", GENERATES.SQL, hop1
					.constructSQLLOPs(), hop2.constructSQLLOPs(),
					ValueType.DOUBLE, DataType.MATRIX);

			maxsqllop.set_properties(new SQLLopProperties());
			String resultName = "result_" + this.get_name() + "_"
					+ this.getHopID();
			String maxName = "maximum_" + this.get_name() + "_"
					+ this.getHopID();

			String qhop1name = SQLLops.addQuotes(hop1.get_sqllops()
					.get_tableName());
			String qhop2name = SQLLops.addQuotes(hop2.get_sqllops()
					.get_tableName());
			String qhop3name = SQLLops.addQuotes(hop3.get_sqllops()
					.get_tableName());

			String maxsql = String.format(SQLLops.MAXVAL2TABLES, qhop1name,
					qhop2name);
			maxsqllop.set_sql(maxsql);
			maxsqllop.set_tableName(maxName);
			maxsqllop.set_properties(new SQLLopProperties());
			SQLLopProperties maxprop = new SQLLopProperties();
			maxprop.setJoinType(JOINTYPE.NONE);
			maxprop.setAggType(AGGREGATIONTYPE.NONE);
			maxprop.setOpString("maxrow(" + hop1.get_sqllops().get_tableName()
					+ "),\r\n,maxcol(" + hop2.get_sqllops().get_tableName()
					+ ")");

			maxsqllop.set_properties(maxprop);

			SQLLops resultsqllop = new SQLLops("", GENERATES.SQL, hop1
					.constructSQLLOPs(), hop2.constructSQLLOPs(),
					ValueType.DOUBLE, DataType.MATRIX);

			String resultsql = String.format(SQLLops.CTABLE, qhop1name,
					qhop2name, qhop3name);
			resultsqllop.set_tableName(resultName);
			resultsqllop.set_sql(resultsql);

			SQLLopProperties resprop = new SQLLopProperties();
			resprop.setJoinType(JOINTYPE.TWO_INNERJOINS);
			resprop.setAggType(AGGREGATIONTYPE.SUM);
			resprop.setOpString("CTable(" + hop1.get_sqllops().get_tableName()
					+ ", " + hop2.get_sqllops().get_tableName() + ", "
					+ hop3.get_sqllops().get_tableName() + ")");

			resultsqllop.set_properties(resprop);

			SQLLops sqllop = new SQLLops(this.get_name(), gen, resultsqllop,
					maxsqllop, this.get_valueType(), this.get_dataType());

			// TODO Uncomment this to make scalar placeholders
			if (this.get_dataType() == DataType.SCALAR && gen == GENERATES.DML)
				sqllop.set_tableName("##" + sqllop.get_tableName() + "##");

			String qResultName = SQLLops.addQuotes(resultName);
			String qMaxName = SQLLops.addQuotes(maxName);
			sqllop.set_sql(String.format(SQLLops.ATTACHLASTZERO, qResultName,
					qMaxName, qResultName, qMaxName));

			SQLLopProperties zeroprop = new SQLLopProperties();
			zeroprop.setJoinType(JOINTYPE.NONE);
			zeroprop.setAggType(AGGREGATIONTYPE.NONE);
			zeroprop.setOpString("Last cell not empty");

			this.set_sqllops(sqllop);

			return sqllop;
		}
		return null;
	}

	private ISQLSelect getCTableSelect(String name1, String name2, String name3) {
		SQLSelectStatement stmt = new SQLSelectStatement();
		stmt.getColumns().add("alias_a.value as row");
		stmt.getColumns().add("alias_b.value as col");
		stmt.getColumns().add("sum(alias_c.value) as value");

		SQLJoin j1 = new SQLJoin();
		j1.setJoinType(JOINTYPE.INNERJOIN);
		j1.setTable1(new SQLTableReference(name1));

		SQLJoin j2 = new SQLJoin();
		j2.setJoinType(JOINTYPE.INNERJOIN);
		j2.setTable1(new SQLTableReference(name2));
		j2.setTable2(new SQLTableReference(name3));
		j2.getConditions().add(new SQLCondition("alias_a.row = alias_b.row"));
		j2.getConditions().add(
				new SQLCondition(BOOLOP.AND, "alias_a.col = alias_b.col"));
		j1.setTable2(j2);
		j1.getConditions().add(new SQLCondition("alias_c.row = alias_a.row"));
		j1.getConditions().add(
				new SQLCondition(BOOLOP.AND, "alias_c.col = alias_a.col"));

		stmt.setTable(j1);
		stmt.getGroupBys().add("alias_a.value");
		stmt.getGroupBys().add("alias_b.value");
		return stmt;
	}

	private ISQLSelect getMaxrow2TablesSelect(String name1, String name2) {
		SQLSelectStatement stmt = new SQLSelectStatement();
		stmt.getColumns().add("max(alias_a.value) AS mrow");
		stmt.getColumns().add("max(alias_b.value) AS mcol");

		SQLJoin crossJoin = new SQLJoin();
		crossJoin.setJoinType(JOINTYPE.CROSSJOIN);
		crossJoin.setTable1(new SQLTableReference(name1, SQLLops.ALIAS_A));
		crossJoin.setTable2(new SQLTableReference(name2, SQLLops.ALIAS_B));
		stmt.setTable(crossJoin);

		return stmt;
	}

	private ISQLSelect getAttachLastZeroSelect(String res, String max) {
		SQLUnion union = new SQLUnion();
		SQLSelectStatement stmt1 = new SQLSelectStatement();
		SQLSelectStatement stmt2 = new SQLSelectStatement();

		stmt1.getColumns().add("*");
		stmt1.setTable(new SQLTableReference(res));

		stmt2.getColumns().add("mrow AS row");
		stmt2.getColumns().add("mcol AS col");
		stmt2.getColumns().add("0 AS value");

		stmt2.setTable(new SQLTableReference(max));

		SQLSelectStatement tmp = new SQLSelectStatement();
		SQLJoin join = new SQLJoin();
		join.setJoinType(JOINTYPE.INNERJOIN);
		join.setTable1(new SQLTableReference(res));
		join.setTable2(new SQLTableReference(max));
		join.getConditions().add(new SQLCondition("mrow = row"));
		join.getConditions().add(new SQLCondition(BOOLOP.AND, "mcol = col"));
		tmp.setTable(join);
		tmp.getColumns().add("*");

		stmt2.getWheres().add(
				new SQLCondition("NOT EXISTS (" + tmp.toString() + ")"));
		union.setSelect1(stmt1);
		union.setSelect2(stmt2);
		union.setUnionType(UNIONTYPE.UNIONALL);
		return union;
	}

	@Override
	protected ExecType optFindExecType() throws HopsException {
		// TODO Auto-generated method stub
		return null;
	}
}
