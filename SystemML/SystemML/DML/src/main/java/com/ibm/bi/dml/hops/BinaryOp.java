/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.AppendM;
import com.ibm.bi.dml.lops.AppendCP;
import com.ibm.bi.dml.lops.AppendG;
import com.ibm.bi.dml.lops.AppendR;
import com.ibm.bi.dml.lops.Binary;
import com.ibm.bi.dml.lops.BinaryCP;
import com.ibm.bi.dml.lops.CentralMoment;
import com.ibm.bi.dml.lops.CoVariance;
import com.ibm.bi.dml.lops.CombineBinary;
import com.ibm.bi.dml.lops.CombineUnary;
import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.PartialAggregate;
import com.ibm.bi.dml.lops.PickByCount;
import com.ibm.bi.dml.lops.SortKeys;
import com.ibm.bi.dml.lops.Unary;
import com.ibm.bi.dml.lops.UnaryCP;
import com.ibm.bi.dml.lops.CombineBinary.OperationTypes;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
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


/* Binary (cell operations): aij + bij
 * 		Properties: 
 * 			Symbol: *, -, +, ...
 * 			2 Operands
 * 		Semantic: align indices (sort), then perform operation
 */

public class BinaryOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//we use the full remote memory budget (but reduced by sort buffer), 
	public static final double APPEND_MEM_MULTIPLIER = 1.0;
	
	Hop.OpOp2 op;

	private enum AppendMethod { 
		CP_APPEND, //in-memory general case append
		MR_MAPPEND, //map-only append (rhs must be vector and fit in mapper mem)
		MR_RAPPEND, //reduce-only append (output must have at most one column block)
		MR_GAPPEND, //map-reduce general case append (map-extend, aggregate)
	};
	
	private BinaryOp() {
		//default constructor for clone
	}
	
	public BinaryOp(String l, DataType dt, ValueType vt, Hop.OpOp2 o,
			Hop inp1, Hop inp2) {
		super(Kind.BinaryOp, l, dt, vt);
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
	
	@Override
	public Lop constructLops() 
		throws HopsException, LopsException 
	{	
		if (get_lops() == null) {

			try {
			if (op == Hop.OpOp2.IQM) {
				ExecType et = optFindExecType();
				if ( et == ExecType.MR ) {
					CombineBinary combine = CombineBinary.constructCombineLop(
							OperationTypes.PreSort, (Lop) getInput().get(0)
									.constructLops(), (Lop) getInput().get(1)
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

					Data lit = Data.createLiteralLop(ValueType.DOUBLE, Double.toString(0.25));
					
					lit.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

					PickByCount pick = new PickByCount(
							sort, lit, DataType.MATRIX, get_valueType(),
							PickByCount.OperationTypes.RANGEPICK);

					pick.getOutputParameters().setDimensions(-1, -1, 
							get_rows_in_block(), get_cols_in_block(), -1);

					pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					PartialAggregate pagg = new PartialAggregate(pick,
							HopsAgg2Lops.get(Hop.AggOp.SUM),
							HopsDirection2Lops.get(Hop.Direction.RowCol),
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
							.get(Hop.AggOp.SUM), DataType.MATRIX,
							get_valueType(), ExecType.MR);
					agg1.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_in_block(),
							get_cols_in_block(), getNnz());
					agg1.setupCorrectionLocation(pagg.getCorrectionLocation());
					
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
							HopsOpOp2LopsBS.get(Hop.OpOp2.DIV),
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

			} else if (op == Hop.OpOp2.CENTRALMOMENT) {
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
					cm.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
					set_lops(cm);
				}

			} else if (op == Hop.OpOp2.COVARIANCE) {
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
					CoVariance cov = new CoVariance(
							getInput().get(0).constructLops(), 
							getInput().get(1).constructLops(), 
							get_dataType(), get_valueType(), et);
					cov.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
					cov.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					set_lops(cov);
				}

			} else if (op == Hop.OpOp2.QUANTILE
					|| op == Hop.OpOp2.INTERQUANTILE) {
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
							(op == Hop.OpOp2.QUANTILE) ? PickByCount.OperationTypes.VALUEPICK
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
							(op == Hop.OpOp2.QUANTILE) ? PickByCount.OperationTypes.VALUEPICK
									: PickByCount.OperationTypes.RANGEPICK, et, true);
	
					pick.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
					
					pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
	
					set_lops(pick);
				}
			}
			else if(op == Hop.OpOp2.APPEND)
			{
				ExecType et = optFindExecType();
				DataType dt1 = getInput().get(0).get_dataType();
				DataType dt2 = getInput().get(1).get_dataType();
				if(dt1!=DataType.MATRIX || dt2!=DataType.MATRIX)
					throw new HopsException("Append can only apply to two matrices!");
						
				if( et == ExecType.MR )
				{
					Lop lop = constructAppendLop(getInput().get(0), getInput().get(1), get_dataType(), get_valueType(), this);
					set_lops( lop );						
				}
				else //CP
				{
					Lop offset = createAppendOffsetLop( getInput().get(0) ); //offset 1st input
					
					AppendCP app = new AppendCP(getInput().get(0).constructLops(), getInput().get(1).constructLops(),	
							                    offset, get_dataType(), get_valueType());
					app.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					app.getOutputParameters().setDimensions(getInput().get(0).get_dim1(), getInput().get(0).get_dim2()+getInput().get(1).get_dim2(), 
							                                get_rows_in_block(), get_cols_in_block(), getNnz());
					set_lops(app);
				}
			}
			else {
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
					
					//select specific operator implementations
					com.ibm.bi.dml.lops.Unary.OperationTypes ot = null;
					Hop right = getInput().get(1);
					if( op==OpOp2.POW && right instanceof LiteralOp && ((LiteralOp)right).getDoubleValue()==2.0  )
						ot = com.ibm.bi.dml.lops.Unary.OperationTypes.POW2;
					else if( op==OpOp2.MULT && right instanceof LiteralOp && ((LiteralOp)right).getDoubleValue()==2.0  )
						ot = com.ibm.bi.dml.lops.Unary.OperationTypes.MULTIPLY2;
					else //general case
						ot = HopsOpOp2LopsU.get(op);
					
					
					Unary unary1 = new Unary(getInput().get(0).constructLops(),
								   getInput().get(1).constructLops(), ot, get_dataType(), get_valueType(), et);
				
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
	
						// Both operands are matrices
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
				throw new HopsException(this.printErrorLocation() + "error constructing Lops for BinaryOp Hop -- \n" , e);
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
		if (LOG.isDebugEnabled()){
			if (get_visited() != VISIT_STATUS.DONE) {
				super.printMe();
				LOG.debug("  Operation: " + op );
				for (Hop h : getInput()) {
					h.printMe();
				}
				;
			}
			set_visited(VISIT_STATUS.DONE);
		}
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		if (get_sqllops() == null) {
			if (this.getInput().size() != 2)
				throw new HopsException(this.printErrorLocation() + "Binary Hop must have two inputs \n");

			GENERATES gen = determineGeneratesFlag();

			Hop hop1 = this.getInput().get(0);
			Hop hop2 = this.getInput().get(1);

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
				sqllop.set_tableName(Lop.VARIABLE_NAME_PLACEHOLDER + sqllop.get_tableName() + Lop.VARIABLE_NAME_PLACEHOLDER);

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
		Hop hop1 = this.getInput().get(0);
	
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
				+ Hop.HopsOpOp2String.get(op) + " "
				+ hop1.get_sqllops().get_tableName());

		return prop;
	}

	private SQLSelectStatement getSQLSelect() throws HopsException {
		Hop hop1 = this.getInput().get(0);
		Hop hop2 = this.getInput().get(1);

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
		boolean isvalid = Hop.isSupported(this.op);
		boolean isfunc = Hop.isFunction(this.op);
		boolean isMaxMin = op == OpOp2.MAX || op == OpOp2.MIN;

		if (!isvalid)
			throw new HopsException(this.printErrorLocation() + "In BinaryOp Hop, Cannot create SQL for operation "
					+ Hop.HopsOpOp2String.get(this.op));

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
						operation = Hop.HopsOpOp2String.get(this.op) + "("
								+ scalval + ", " + matval + ")";
				} else {
					if (!isfunc)
						operation = matval + " " + opr + " " + scalval;
					else
						operation = Hop.HopsOpOp2String.get(this.op) + "("
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
						Hop.HopsOpOp2String.get(this.op));

			String op1 = SQLLops.addQuotes(hop1.get_sqllops().get_tableName());
			String op2 = SQLLops.addQuotes(hop2.get_sqllops().get_tableName());

			// Determine the right join to use
			JOINTYPE jt = JOINTYPE.FULLOUTERJOIN;
			if (op == OpOp2.MULT || op == OpOp2.AND)
				jt = JOINTYPE.INNERJOIN;
			else if (op == OpOp2.DIV)
				jt = JOINTYPE.LEFTJOIN;

			if (Hop.isBooleanOperation(op))
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
				operation = String.format("%s(%s, %s)", Hop.HopsOpOp2String
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
		Hop hop1 = this.getInput().get(0);
		Hop hop2 = this.getInput().get(1);

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
		boolean isvalid = Hop.isSupported(this.op);

		// But min, max and log can still be done
		boolean isfunc = Hop.isFunction(this.op);
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
							operation = Hop.HopsOpOp2String.get(this.op) + "("
									+ scoal + ", " + avalcoal + ")";
					} else {
						if (!isfunc)
							operation = s + " " + opr + " " + aval;
						else
							operation = Hop.HopsOpOp2String.get(this.op) + "("
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
							operation = Hop.HopsOpOp2String.get(this.op)
									+ avalcoal + ", " + scoal + " )";
					} else {
						if (!isfunc)
							operation = aval + " " + opr + " " + s;
						else
							operation = Hop.HopsOpOp2String.get(this.op) + "("
									+ aval + ", " + s + ")";
					}

					if (hop2.get_sqllops().get_flag() == GENERATES.SQL)
						tbl = ", "
								+ SQLLops.addQuotes(hop2.get_sqllops()
										.get_tableName());
				}

				if (Hop.isBooleanOperation(op))
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
							Hop.HopsOpOp2String.get(this.op));

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

				if (Hop.isBooleanOperation(op))
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
							Hop.HopsOpOp2String.get(this.op), o1, o2);

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
					+ Hop.HopsOpOp2String.get(this.op));
		return sql;
	}

	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		double ret = 0;
		
		//preprocessing step (recognize unknowns)
		if( dimsKnown() && _nnz<0 ) //never after inference
			nnz = -1; 
		
		if(op==OpOp2.APPEND && !OptimizerUtils.ALLOW_DYN_RECOMPILATION ) {	
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
					double sp1 = (input1.getNnz()>0 && input1.get_dataType()==DataType.MATRIX) ? OptimizerUtils.getSparsity(input1.get_dim1(), input1.get_dim2(), input1.getNnz()) : 1.0;
					double sp2 = (input2.getNnz()>0 && input2.get_dataType()==DataType.MATRIX) ? OptimizerUtils.getSparsity(input2.get_dim1(), input2.get_dim2(), input2.getNnz()) : 1.0;
					sparsity = OptimizerUtils.getBinaryOpSparsity(sp1, sp2, op, true);	
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
		if ( op == OpOp2.QUANTILE || op == OpOp2.IQM ) {
			// buffer (=2*input_size) and output (=input_size) for SORT operation 
			// getMemEstimate works for both cases of known dims and worst-case
			ret = getInput().get(0).getMemEstimate() * 3; 
		}
		else if ( op == OpOp2.SOLVE ) {
			// x=solve(A,b) relies on QR decomposition of A, which is done using Apache commons-math
			// matrix of size same as the first input
			double interOutput = OptimizerUtils.estimateSizeExactSparsity(getInput().get(0).get_dim1(), getInput().get(0).get_dim2(), 1.0); 
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
		DataType dt1 = input1.get_dataType();
		DataType dt2 = input2.get_dataType();
		
		if( op== OpOp2.APPEND )
		{
			if( mc[0].dimsKnown() && mc[1].dimsKnown() ) 
				ret = new long[]{mc[0].get_rows(), mc[0].get_cols()+mc[1].get_cols(), mc[0].getNonZeros() + mc[1].getNonZeros()};
		}
		else if ( op == OpOp2.SOLVE ) {
			// Output is a (likely to be dense) vector of size number of columns in the first input
			if ( mc[0].get_cols() > 0 ) {
				ret = new long[]{ mc[0].get_cols(), 1, mc[0].get_cols()};
			}
		}
		else //general case
		{
			long ldim1, ldim2;
			double sp1 = 1.0, sp2 = 1.0;
			
			if( dt1 == DataType.MATRIX && dt2 == DataType.SCALAR && mc[0].dimsKnown() )
			{
				ldim1 = mc[0].get_rows();
				ldim2 = mc[0].get_cols();
			}
			else if( dt1 == DataType.SCALAR && dt2 == DataType.MATRIX  ) 
			{
				ldim1 = mc[1].get_rows();
				ldim2 = mc[1].get_cols();
			}
			else //MATRIX - MATRIX 
			{
				ldim1 = (mc[0].get_rows()>0) ? mc[0].get_rows() : mc[1].get_rows();
				ldim2 = (mc[0].get_cols()>0) ? mc[0].get_cols() : mc[1].get_cols();
				sp1 = (mc[0].getNonZeros()>0)?OptimizerUtils.getSparsity(ldim1, ldim2, mc[0].getNonZeros()):1.0;
				sp2 = (mc[1].getNonZeros()>0)?OptimizerUtils.getSparsity(ldim1, ldim2, mc[1].getNonZeros()):1.0;
			}
			
			if( ldim1>0 && ldim2>0 )
			{
				long lnnz = (long) (ldim1*ldim2*OptimizerUtils.getBinaryOpSparsity(sp1, sp2, op, true));
				ret = new long[]{ldim1, ldim2, lnnz};
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
		
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && ((!dimsKnown(true)&&_etype==ExecType.MR) || op == OpOp2.APPEND) )
				setRequiresRecompile();
		}
		
		if ( op == OpOp2.SOLVE ) {
			_etype = ExecType.CP;
			/*
			if ( getMemEstimate() > OptimizerUtils.getMemBudget(true) )
				throw new HopsException("Insufficient memory to execute function: solve()" );
				*/
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
	public static Lop constructAppendLop( Hop left, Hop right, DataType dt, ValueType vt, Hop current ) 
		throws HopsException, LopsException
	{
		Lop ret = null;
		
		long m1_dim1 = left.get_dim1();
		long m1_dim2 = left.get_dim2();		
		long m2_dim1 = right.get_dim1();
		long m2_dim2 = right.get_dim2();
		long m3_dim2 = (m1_dim2>0 && m2_dim2>0) ? (m1_dim2 + m2_dim2) : -1; //output cols
		long m3_nnz = (left.getNnz()>0 && right.getNnz()>0) ? (left.getNnz() + right.getNnz()) : -1; //output nnz
		long brlen = left.get_rows_in_block();
		long bclen = left.get_cols_in_block();
		
		Lop offset = createAppendOffsetLop( left ); //offset 1st input
		AppendMethod am = optFindAppendMethod(m1_dim1, m1_dim2, m2_dim1, m2_dim2, brlen, bclen);
	
		switch( am )
		{
			case MR_MAPPEND: 
			{
				//special case map-only append
				AppendM appM = new AppendM(left.constructLops(), right.constructLops(),	offset, dt, vt);
				appM.setAllPositions(current.getBeginLine(), current.getBeginColumn(), current.getEndLine(), current.getEndColumn());
				appM.getOutputParameters().setDimensions(m1_dim1, m3_dim2, brlen, bclen, m3_nnz);
				ret = appM;
				break;
			}
			case MR_RAPPEND:
			{
				//special case reduce append w/ one column block
				//group
				Group group1 = new Group(left.constructLops(), Group.OperationTypes.Sort, DataType.MATRIX, vt);
				group1.getOutputParameters().setDimensions(m1_dim1, m1_dim2, brlen, bclen, left.getNnz());
				group1.setAllPositions(left.getBeginLine(), left.getBeginColumn(), left.getEndLine(), left.getEndColumn());
				
				Group group2 = new Group(right.constructLops(), Group.OperationTypes.Sort, DataType.MATRIX, vt);
				group1.getOutputParameters().setDimensions(m2_dim1, m2_dim2, brlen, bclen, right.getNnz());
				group1.setAllPositions(right.getBeginLine(), right.getBeginColumn(), right.getEndLine(), right.getEndColumn());
				
				AppendR appR = new AppendR(group1, group2, dt, vt);
				appR.getOutputParameters().setDimensions(m1_dim1, m3_dim2, brlen, bclen, m3_nnz);
				appR.setAllPositions(current.getBeginLine(), current.getBeginColumn(), current.getEndLine(), current.getEndColumn());
				
				ret = appR;
				break;
			}	
			case MR_GAPPEND:
			{
				//general case: map expand append, reduce aggregate
				Lop offset2 = createAppendOffsetLop( right ); //offset second input
				
				AppendG appG = new AppendG(left.constructLops(), right.constructLops(),	offset, offset2, dt, vt);
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
		}
		
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
		long m1_dim1 = left.get_dim1();
		long m1_dim2 = left.get_dim2();		
		long m2_dim1 = right1.get_dim1();
		long m2_dim2 = right1.get_dim2();
		long m3_dim1 = right2.get_dim1();
		long m3_dim2 = right2.get_dim2();		
		long m41_dim2 = (m1_dim2>0 && m2_dim2>0) ? (m1_dim2 + m2_dim2) : -1; //output cols
		long m41_nnz = (left.getNnz()>0 && right1.getNnz()>0) ? 
				      (left.getNnz() + right1.getNnz()) : -1; //output nnz
		long m42_dim2 = (m1_dim2>0 && m2_dim2>0 && m3_dim2>0) ? (m1_dim2 + m2_dim2 + m3_dim2) : -1; //output cols
		long m42_nnz = (left.getNnz()>0 && right1.getNnz()>0 && right2.getNnz()>0) ? 
				      (left.getNnz() + right1.getNnz()+ right2.getNnz()) : -1; //output nnz
		long brlen = left.get_rows_in_block();
		long bclen = left.get_cols_in_block();
		
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
		
		AppendR appR1 = new AppendR(group1, group2, dt, vt);
		appR1.getOutputParameters().setDimensions(m1_dim1, m41_dim2, brlen, bclen, m41_nnz);
		appR1.setAllPositions(current.getBeginLine(), current.getBeginColumn(), current.getEndLine(), current.getEndColumn());
		
		AppendR appR2 = new AppendR(appR1, group3, dt, vt);
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
		footprint += OptimizerUtils.estimateSize(Math.min(m1_dim1, m1_rpb), Math.min(m1_dim2, m1_cpb), 1.0);
		
		// size of right input (vector)
		footprint += OptimizerUtils.estimateSize(m2_dim1, m2_dim2, 1.0);
		
		// size of the output (only boundary block is merged)
		footprint += OptimizerUtils.estimateSize(Math.min(m1_dim1, m1_rpb), Math.min(m1_dim2+m2_dim2, m1_cpb), 1.0);
		
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
		//check for best case (map-only)		
		if(    m2_dim2 == 1  //rhs is vector
		    && m2_dim1 >= 1 ) // rhs row dim known 
		{
			double footprint = BinaryOp.footprintInMapper(m1_dim1, m1_dim2, m2_dim1, m2_dim2, m1_rpb, m1_cpb);
			if ( footprint < APPEND_MEM_MULTIPLIER * OptimizerUtils.getRemoteMemBudget(true) )
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
	
	/**
	 * 
	 * @param inputPos
	 * @return
	 * @throws HopsException
	 * @throws LopsException
	 */
	private static Lop createAppendOffsetLop( Hop hop ) 
		throws HopsException, LopsException
	{
		Lop offset = null;
		
		if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && hop.dimsKnown() )
		{
			// If dynamic recompilation is enabled and dims are known, we can replace the ncol with 
			// a literal in order to increase the piggybacking potential. This is safe because append 
			// is always marked for recompilation and hence, we have propagated the exact dimensions.
			offset = Data.createLiteralLop(ValueType.INT, String.valueOf(hop.get_dim2()));
		}
		else
		{
			offset = new UnaryCP(hop.constructLops(), 
					UnaryCP.OperationTypes.NCOL, DataType.SCALAR, ValueType.INT);
		}
		
		offset.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
		offset.setAllPositions(hop.getBeginLine(), hop.getBeginColumn(), hop.getEndLine(), hop.getEndColumn());
		
		return offset;
	}
	
	
	@Override
	public void refreshSizeInformation()
	{
		Hop input1 = getInput().get(0);
		Hop input2 = getInput().get(1);		
		DataType dt1 = input1.get_dataType();
		DataType dt2 = input2.get_dataType();
		
		if ( get_dataType() == DataType.SCALAR ) 
		{
			//do nothing always known
		}
		else //MATRIX OUTPUT
		{
			//TODO quantile
			if( op== OpOp2.APPEND )
			{
				set_dim1( (input1.get_dim1()>0) ? input1.get_dim1() : input2.get_dim1() );
				
				//ensure both columns are known, otherwise dangerous underestimation due to +(-1)
				if( input1.get_dim2()>0 && input2.get_dim2()>0 )
					set_dim2( input1.get_dim2() + input2.get_dim2() );
				//ensure both nnz are known, otherwise dangerous underestimation due to +(-1)
				if( input1.getNnz()>0 && input2.getNnz()>0 )
					setNnz( input1.getNnz() + input2.getNnz() );
			}
			else //general case
			{
				long ldim1, ldim2, lnnz1 = -1, lnnz2 = -1;
				
				if( dt1 == DataType.MATRIX && dt2 == DataType.SCALAR )
				{
					ldim1 = input1.get_dim1();
					ldim2 = input1.get_dim2();
					lnnz1 = input1.getNnz();
				}
				else if( dt1 == DataType.SCALAR && dt2 == DataType.MATRIX  ) 
				{
					ldim1 = input2.get_dim1();
					ldim2 = input2.get_dim2();	
					lnnz2 = input2.getNnz();
				}
				else //MATRIX - MATRIX 
				{
					ldim1 = (input1.get_dim1()>0) ? input1.get_dim1() : input2.get_dim1();
					ldim2 = (input1.get_dim2()>0) ? input1.get_dim2() : input2.get_dim2();
					lnnz1 = input1.getNnz();
					lnnz2 = input2.getNnz();
				}
				
				set_dim1( ldim1 );
				set_dim2( ldim2 );
				
				//update nnz only if we can ensure exact results, 
				//otherwise propagated via worst-case estimates
				if(op == OpOp2.POW)
					setNnz( lnnz1 );
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
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( that._kind!=Kind.BinaryOp )
			return false;
		
		BinaryOp that2 = (BinaryOp)that;
		return (   op == that2.op
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
		         ||op==OpOp2.LOG     ||op==OpOp2.POW
		         ||op==OpOp2.POW2CM );
	}
}
