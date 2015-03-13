/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import java.util.ArrayList;

import com.ibm.bi.dml.lops.Aggregate;
import com.ibm.bi.dml.lops.CombineUnary;
import com.ibm.bi.dml.lops.CumsumOffsetBinary;
import com.ibm.bi.dml.lops.CumsumPartialAggregate;
import com.ibm.bi.dml.lops.CumsumSplitAggregate;
import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.Group;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.PartialAggregate;
import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.lops.PickByCount;
import com.ibm.bi.dml.lops.SortKeys;
import com.ibm.bi.dml.lops.Unary;
import com.ibm.bi.dml.lops.UnaryCP;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.sql.sqllops.SQLCondition;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;
import com.ibm.bi.dml.sql.sqllops.SQLSelectStatement;
import com.ibm.bi.dml.sql.sqllops.SQLTableReference;


/* Unary (cell operations): e.g, b_ij = round(a_ij)
 * 		Semantic: given a value, perform the operation (independent of other values)
 */

public class UnaryOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private OpOp1 _op = null;

	
	private UnaryOp() {
		//default constructor for clone
	}
	
	public UnaryOp(String l, DataType dt, ValueType vt, OpOp1 o, Hop inp)
			throws HopsException 
	{
		super(Hop.Kind.UnaryOp, l, dt, vt);

		getInput().add(0, inp);
		inp.getParent().add(this);

		_op = o;
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	// this is for OpOp1, e.g. A = -B (0-B); and a=!b
	public OpOp1 getOp() {
		return _op;
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
	public String getOpString() {
		String s = new String("");
		s += "u(" + _op + ")";
		// s += HopsOpOp1String.get(_op) + ")";
		return s;
	}

	@Override
	public Lop constructLops()
		throws HopsException, LopsException 
	{		
		//reuse existing lop
		if( getLops() != null )
			return getLops();
		
		try 
		{
			Hop input = getInput().get(0);
			
			if (getDataType() == DataType.SCALAR || _op == OpOp1.CAST_AS_MATRIX) 
			{
				if (_op == Hop.OpOp1.IQM)  //special handling IQM
				{
					Lop iqmLop = constructLopsIQM();
					setLops(iqmLop);
				} 
				else if(_op == Hop.OpOp1.MEDIAN) {
					Lop medianLop = constructLopsMedian();
					setLops(medianLop);
				}
				else //general case SCALAR/CAST (always in CP)
				{
					UnaryCP.OperationTypes optype = HopsOpOp1LopsUS.get(_op);
					if( optype == null )
						throw new HopsException("Unknown UnaryCP lop type for UnaryOp operation type '"+_op+"'");
					
					UnaryCP unary1 = new UnaryCP(input.constructLops(), optype, getDataType(), getValueType());
					unary1.getOutputParameters().setDimensions(getDim1(), getDim2(), 
							             getRowsInBlock(), getColsInBlock(), getNnz());
					unary1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					setLops(unary1);
				}
			} 
			else //general case MATRIX
			{
				ExecType et = optFindExecType();
				if ( et == ExecType.SPARK )  {
					// throw new HopsException("constructLops (cumsum) for UnaryOp not implemented for Spark");
					et = ExecType.CP;
				}
				
				if( _op == Hop.OpOp1.CUMSUM && et==ExecType.MR )  //special handling MR-cumsum
				{
					//TODO additional physical operation if offsets fit in memory
					Lop cumsumLop = constructLopsMRCumsum();
					setLops(cumsumLop);
				}
				else //default unary 
				{
					Unary unary1 = new Unary(input.constructLops(), HopsOpOp1LopsU.get(_op), 
							                 getDataType(), getValueType(), et);
					unary1.getOutputParameters().setDimensions(getDim1(),
							getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
					unary1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					setLops(unary1);
				}
			}
		} 
		catch (Exception e) 
		{
			throw new HopsException(this.printErrorLocation() + "error constructing Lops for UnaryOp Hop -- \n " , e);
		}
		
		return getLops();
	}
	

	private Lop constructLopsMedian() throws HopsException, LopsException {
		ExecType et = optFindExecType();
		if ( et == ExecType.SPARK )  {
			// throw new HopsException("constructLopsMedian for UnaryOp not implemented for Spark");
			et = ExecType.CP;
		}
		if ( et == ExecType.MR ) {
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

			return pick;
		}
		else {
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
			return pick;
		}
	}
	
	private Lop constructLopsIQM() throws HopsException, LopsException {
		ExecType et = optFindExecType();
		if ( et == ExecType.SPARK )  {
			// throw new HopsException("constructLopsIQM for UnaryOp not implemented for Spark");
			et = ExecType.CP;
		}
		
		Hop input = getInput().get(0);
		if ( et == ExecType.MR ) {
			CombineUnary combine = CombineUnary.constructCombineLop(input.constructLops(),
							                       DataType.MATRIX, getValueType());
			combine.getOutputParameters().setDimensions(
					input.getDim1(),
					input.getDim2(), 
					input.getRowsInBlock(),
					input.getColsInBlock(),
					input.getNnz());

			SortKeys sort = SortKeys.constructSortByValueLop(combine,
							           SortKeys.OperationTypes.WithoutWeights,
							           DataType.MATRIX, ValueType.DOUBLE, ExecType.MR);

			// Sort dimensions are same as the first input
			sort.getOutputParameters().setDimensions(
					input.getDim1(),
					input.getDim2(),
					input.getRowsInBlock(),
					input.getColsInBlock(),
					input.getNnz());

			Data lit = Data.createLiteralLop(ValueType.DOUBLE, Double.toString(0.25));
			
			lit.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
            			
			PickByCount pick = new PickByCount(
					sort, lit, DataType.MATRIX, getValueType(),
					PickByCount.OperationTypes.RANGEPICK);

			pick.getOutputParameters().setDimensions(-1, -1,  
					getRowsInBlock(), getColsInBlock(), -1);
			
			pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

			PartialAggregate pagg = new PartialAggregate(
					pick, HopsAgg2Lops.get(Hop.AggOp.SUM),
					HopsDirection2Lops.get(Hop.Direction.RowCol),
					DataType.MATRIX, getValueType());
			
			pagg.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

			// Set the dimensions of PartialAggregate LOP based on the
			// direction in which aggregation is performed
			pagg.setDimensionsBasedOnDirection(getDim1(),
						getDim2(), getRowsInBlock(),
						getColsInBlock());

			Group group1 = new Group(
					pagg, Group.OperationTypes.Sort, DataType.MATRIX,
					getValueType());
			group1.getOutputParameters().setDimensions(getDim1(),
					getDim2(), getRowsInBlock(),
					getColsInBlock(), getNnz());
			group1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

			Aggregate agg1 = new Aggregate(
					group1, HopsAgg2Lops.get(Hop.AggOp.SUM),
					DataType.MATRIX, getValueType(), ExecType.MR);
			agg1.getOutputParameters().setDimensions(getDim1(),
					getDim2(), getRowsInBlock(),
					getColsInBlock(), getNnz());
			agg1.setupCorrectionLocation(pagg.getCorrectionLocation());
			
			agg1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			
			UnaryCP unary1 = new UnaryCP(
					agg1, HopsOpOp1LopsUS.get(OpOp1.CAST_AS_SCALAR),
					getDataType(), getValueType());
			unary1.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			unary1.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

			Unary iqm = new Unary(sort, unary1, Unary.OperationTypes.MR_IQM, DataType.SCALAR, ValueType.DOUBLE, ExecType.CP);
			iqm.getOutputParameters().setDimensions(0, 0, 0, 0, -1);
			iqm.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

			return iqm;
		}
		else {
			SortKeys sort = SortKeys.constructSortByValueLop(
					input.constructLops(), 
					SortKeys.OperationTypes.WithoutWeights, 
					DataType.MATRIX, ValueType.DOUBLE, et );
			sort.getOutputParameters().setDimensions(
					input.getDim1(),
					input.getDim2(),
					input.getRowsInBlock(),
					input.getColsInBlock(),
					input.getNnz());
			PickByCount pick = new PickByCount(sort, null,
					getDataType(),getValueType(),
					PickByCount.OperationTypes.IQM, et, true);

			pick.getOutputParameters().setDimensions(getDim1(),
					getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());

			pick.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			
			return pick;
		}
	}
	
	/**
	 * MR Cumsum is currently based on a multipass algorithm of (1) preaggregation and (2) subsequent offsetting. 
	 * Note that we currently support one robust physical operator but many alternative
	 * realizations are possible for specific scenarios (e.g., when the preaggregated intermediate
	 * fit into the map task memory budget) or by creating custom job types.
	 * 
	 * 
	 * 
	 * @return
	 * @throws HopsException
	 * @throws LopsException
	 */
	private Lop constructLopsMRCumsum() 
		throws HopsException, LopsException 
	{
		Hop input = getInput().get(0);
		long rlen = input.getDim1();
		long clen = input.getDim2();
		long brlen = input.getRowsInBlock();
		long bclen = input.getColsInBlock();
		boolean unknownSize = !dimsKnown();
		
		Lop X = input.constructLops();
		Lop TEMP = X;
		ArrayList<Lop> DATA = new ArrayList<Lop>();
		int level = 0;
		
		//recursive preaggregation until aggregates fit into CP memory budget
		while( ((2*OptimizerUtils.estimateSize(TEMP.getOutputParameters().getNumRows(), clen) + OptimizerUtils.estimateSize(1, clen)) 
				 > OptimizerUtils.getLocalMemBudget()
			   && TEMP.getOutputParameters().getNumRows()>1) || unknownSize )
		{
			DATA.add(TEMP);
	
			//preaggregation per block
			long rlenAgg = (long)Math.ceil((double)TEMP.getOutputParameters().getNumRows()/brlen);
			Lop preagg = new CumsumPartialAggregate(TEMP, DataType.MATRIX, ValueType.DOUBLE);
			preagg.getOutputParameters().setDimensions(rlenAgg, clen, brlen, bclen, -1);
			setLineNumbers(preagg);
			
			Group group = new Group( preagg, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE );
			group.getOutputParameters().setDimensions(rlenAgg, clen, brlen, bclen, -1);
			setLineNumbers(group);
			
			Aggregate agg = new Aggregate(group, HopsAgg2Lops.get(AggOp.SUM), getDataType(), getValueType(), ExecType.MR);
			agg.getOutputParameters().setDimensions(rlenAgg, clen, brlen, bclen, -1);
			agg.setupCorrectionLocation(CorrectionLocationType.NONE); // aggregation uses kahanSum but the inputs do not have correction values
			setLineNumbers(agg);
			TEMP = agg;	
			level++;
			unknownSize = false; //in case of unknowns, generate one level
		}
		
		//in-memory cum sum (of partial aggregates)
		if( TEMP.getOutputParameters().getNumRows()!=1 ){
			Unary unary1 = new Unary( TEMP, HopsOpOp1LopsU.get(_op), DataType.MATRIX, ValueType.DOUBLE, ExecType.CP);
			unary1.getOutputParameters().setDimensions(TEMP.getOutputParameters().getNumRows(), clen, brlen, bclen, -1);
			setLineNumbers(unary1);
			TEMP = unary1;
		}
		
		//split, group and mr cumsum
		while( level-- > 0  ) {
			CumsumSplitAggregate split = new CumsumSplitAggregate(TEMP, DataType.MATRIX, ValueType.DOUBLE);
			split.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, -1);
			setLineNumbers(split);
			
			Group group1 = new Group( DATA.get(level), Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE );
			group1.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, -1);
			setLineNumbers(group1);
			
			Group group2 = new Group( split, Group.OperationTypes.Sort, DataType.MATRIX, ValueType.DOUBLE );
			group2.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, -1);
			setLineNumbers(group2);
			
			CumsumOffsetBinary binary = new CumsumOffsetBinary(group1, group2, DataType.MATRIX, ValueType.DOUBLE);
			binary.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, -1);
			setLineNumbers(binary);
			TEMP = binary;
		}
		
		return TEMP;
	}
	

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		
		if(this.getSqlLops() == null)
		{
			if( getInput().isEmpty() )
				throw new HopsException(this.printErrorLocation() + "Unary hop needs one input \n");
			
			//Check whether this is going to be an Insert or With
			GENERATES gen = determineGeneratesFlag();
			
			if( this._op == OpOp1.PRINT )
				gen = GENERATES.PRINT;
			else if(this.getDataType() == DataType.SCALAR && gen != GENERATES.DML_PERSISTENT 
					&& gen != GENERATES.DML_TRANSIENT)
				gen = GENERATES.DML;
			
			Hop input = this.getInput().get(0);
			
			SQLLops sqllop = new SQLLops(this.getName(),
										gen,
										input.constructSQLLOPs(),
										this.getValueType(),
										this.getDataType());

			//TODO Uncomment this to make scalar placeholders
			if(this.getDataType() == DataType.SCALAR && gen == GENERATES.DML)
				sqllop.set_tableName("##" + sqllop.get_tableName() + "##");
			
			String sql = this.getSQLSelectCode(input);
			sqllop.set_sql(sql);
			sqllop.set_properties(getProperties(input));
			this.setSqlLops(sqllop);
		}
		return this.getSqlLops();
	}
	
	private SQLLopProperties getProperties(Hop input)
	{
		SQLLopProperties prop = new SQLLopProperties();
		prop.setJoinType(JOINTYPE.NONE);
		prop.setAggType(AGGREGATIONTYPE.NONE);
		
		prop.setOpString(Hop.HopsOpOp12String.get(this._op) + "(" + input.getSqlLops().get_tableName() + ")");
		
		return prop;
	}
	
	
	
	private String getSQLSelectCode(Hop input) throws HopsException
	{
		String sql = null;

		if(input.getSqlLops().getDataType() == DataType.MATRIX)
		{
			if(this._op == OpOp1.CAST_AS_SCALAR)
			{
				//sqllop.setDataType(DataType.SCALAR);
				sql = String.format(SQLLops.CASTASSCALAROP, input.getSqlLops().get_tableName());
			}
			if(Hop.isFunction(this._op))
			{
				String op = Hop.HopsOpOp12String.get(this._op);
				if(this._op == OpOp1.LOG)
					op = "ln";
				sql = String.format(SQLLops.UNARYFUNCOP, op, input.getSqlLops().get_tableName());
			}
			//Not is in SLQ not "!" but "NOT"
			else if(this._op == OpOp1.NOT)
			{
				sql = String.format(SQLLops.UNARYNOT, input.getSqlLops().get_tableName());
			}
		}
		
		else if(input.getSqlLops().getDataType() == DataType.SCALAR)
		{
			String table = "";
			String opr = null;
			String s = input.getSqlLops().get_tableName();
			
			if(input.getSqlLops().get_flag() == GENERATES.SQL)
			{
				String squoted = SQLLops.addQuotes(s);
				table = " FROM " + squoted;
				s = squoted + "." + SQLLops.SCALARVALUECOLUMN;
			}
			if( this._op == OpOp1.PRINT )
			{
				//sqllop.set_flag(GENERATES.PRINT);
				String tname = input.getSqlLops().get_tableName();
				
				if(input.getSqlLops().getDataType() == DataType.MATRIX
						|| input.getSqlLops().get_flag() == GENERATES.SQL)
				{
					tname = SQLLops.addQuotes(tname);
				
					sql = String.format(SQLLops.SIMPLESCALARSELECTFROM,
						tname + "." + SQLLops.SCALARVALUECOLUMN, tname);
				}
				else
					sql = tname;
			}
			else
			{
				//Create correct string for operation
				if(Hop.isFunction(this._op))
					opr = Hop.HopsOpOp12String.get(this._op) + "( " + s + " )";
				else if(this._op == OpOp1.NOT)
					opr = "NOT " + s;

				sql = String.format(SQLLops.UNARYSCALAROP + table, opr);
			}
		}
		else
			throw new HopsException(this.printErrorLocation() + "Other unary operations than matrix and scalar operations are currently not supported \n");
		return sql;
	}
	
	@SuppressWarnings("unused")
	private SQLSelectStatement getSQLSelect(Hop input) throws HopsException
	{
		SQLSelectStatement stmt = new SQLSelectStatement();
		
		if(input.getSqlLops().getDataType() == DataType.MATRIX)
		{
			if(this._op == OpOp1.CAST_AS_SCALAR)
			{
				stmt.getColumns().add("max(value) AS sval");
				stmt.setTable(new SQLTableReference(input.getSqlLops().get_tableName()));
			}
			if(Hop.isFunction(this._op))
			{
				String op = Hop.HopsOpOp12String.get(this._op);
				//SELECT %s(alias_a.value) as value FROM \"%s\" alias_a
				stmt.getColumns().add("row");
				stmt.getColumns().add("col");
				stmt.getColumns().add(op + "(value) AS value");
				stmt.setTable(new SQLTableReference(input.getSqlLops().get_tableName()));
			}
			//Not is in SLQ not "!" but "NOT"
			else if(this._op == OpOp1.NOT)
			{
				//SELECT alias_a.row AS row, alias_a.col AS col, 1 as value FROM \"%s\" alias_a WHERE alias_a.value == 0";
				
				stmt.getColumns().add("row");
				stmt.getColumns().add("col");
				stmt.getColumns().add("1 AS value");
				stmt.setTable(new SQLTableReference(input.getSqlLops().get_tableName()));
				stmt.getWheres().add(new SQLCondition("value == 0"));
			}
		}
		
		else if(input.getSqlLops().getDataType() == DataType.SCALAR)
		{
			String opr = null;

			if( this._op == OpOp1.PRINT )
			{
				//sqllop.set_flag(GENERATES.PRINT);
				String tname = input.getSqlLops().get_tableName();
				
				if(input.getSqlLops().getDataType() == DataType.MATRIX
						|| input.getSqlLops().get_flag() == GENERATES.SQL)
				{
					tname = SQLLops.addQuotes(tname);
				
					//SELECT %s AS sval FROM %s
					
					stmt.getColumns().add(SQLLops.SCALARVALUECOLUMN);
					stmt.setTable(new SQLTableReference(tname));
				}
				//else
				//	sql = tname;
			}
			else
			{
				//Create correct string for operation
				if(Hop.isFunction(this._op))
					opr = Hop.HopsOpOp12String.get(this._op) + "( sval )";
				else if(this._op == OpOp1.NOT)
					opr = "NOT sval";
				
				stmt.getColumns().add(opr + " AS sval");
				if(input.getSqlLops().get_flag() == GENERATES.SQL)
					stmt.setTable(new SQLTableReference(SQLLops.addQuotes(input.getSqlLops().get_tableName())));
			}
		}
		else
			throw new HopsException(this.printErrorLocation() + "Other unary operations than matrix and scalar operations are currently not supported \n");
		return stmt;
	}
	
	@Override
	public void computeMemEstimate(MemoTable memo)
	{
		//overwrites default hops behavior
		super.computeMemEstimate(memo);
		
		if( _op == Hop.OpOp1.NROW || _op == Hop.OpOp1.NCOL ) //specific case for meta data ops
		{
			_memEstimate = OptimizerUtils.INT_SIZE;
			//_outputMemEstimate = OptimizerUtils.INT_SIZE;
			//_processingMemEstimate = 0;
		}
	}

	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		double ret = 0;
		
		if ( _op == OpOp1.IQM  || _op == OpOp1.MEDIAN) {
			// buffer (=2*input_size) and output (=input_size) for SORT operation
			// getMemEstimate works for both cases of known dims and worst-case stats
			ret = getInput().get(0).getMemEstimate() * 3; 
		}
		
		return ret;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		long[] ret = null;
	
		Hop input = getInput().get(0);
		MatrixCharacteristics mc = memo.getAllInputStats(input);
		if( mc.dimsKnown() ) {
			if( _op==OpOp1.ABS || _op==OpOp1.COS || _op==OpOp1.SIN || _op==OpOp1.TAN 
				|| _op==OpOp1.ACOS || _op==OpOp1.ASIN || _op==OpOp1.ATAN  
				|| _op==OpOp1.SQRT || _op==OpOp1.ROUND ) //sparsity preserving
			{
				ret = new long[]{mc.getRows(), mc.getCols(), mc.getNonZeros()};
			}
			else 
				ret = new long[]{mc.getRows(), mc.getCols(), -1};	
		}
		
		return ret;
	}
	

	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}
	
	private boolean isInMemoryOperation() {
		switch(_op) {
		case INVERSE:
			return true;
		default:
			return false;
		}
	}
	
	@Override
	protected ExecType optFindExecType() 
		throws HopsException 
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
			// Choose CP, if the input dimensions are below threshold or if the input is a vector
			// Also, matrix inverse is currently implemented only in CP (through commons math)
			else if ( getInput().get(0).areDimsBelowThreshold() || getInput().get(0).isVector() 
						|| isInMemoryOperation() )
			{
				_etype = ExecType.CP;
			}
			else 
			{
				_etype = ExecType.MR;
			}
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
			
			//mark for recompile (forever)
			if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==ExecType.MR )
				setRequiresRecompile();
		}
		
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		if ( getDataType() == DataType.SCALAR ) 
		{
			//do nothing always known
		}
		else if( _op == OpOp1.CAST_AS_MATRIX && getInput().get(0).getDataType()==DataType.SCALAR )
		{
			//prevent propagating 0 from scalar (which would be interpreted as unknown)
			setDim1( 1 );
			setDim2( 1 );
		}
		else //general case
		{
			// If output is a Matrix then this operation is of type (B = op(A))
			// Dimensions of B are same as that of A, and sparsity may/maynot change
			Hop input = getInput().get(0);
			setDim1( input.getDim1() );
			setDim2( input.getDim2() );
			if( _op==OpOp1.ABS || _op==OpOp1.COS || _op==OpOp1.SIN || _op==OpOp1.TAN  
				|| _op==OpOp1.ACOS || _op==OpOp1.ASIN || _op==OpOp1.ATAN
				|| _op==OpOp1.SQRT || _op==OpOp1.ROUND ) //sparsity preserving
			{
				setNnz( input.getNnz() );
			}
		}
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		UnaryOp ret = new UnaryOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._op = _op;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( that._kind!=Kind.UnaryOp )
			return false;
		
		UnaryOp that2 = (UnaryOp)that;		
		return (   _op == that2._op
				&& getInput().get(0) == that2.getInput().get(0));
	}
}
