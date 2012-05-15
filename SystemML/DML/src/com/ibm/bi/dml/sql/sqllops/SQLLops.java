package com.ibm.bi.dml.sql.sqllops;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.Hops.OpOp2;
import com.ibm.bi.dml.hops.Hops.VISIT_STATUS;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.AGGREGATIONTYPE;
import com.ibm.bi.dml.sql.sqllops.SQLLopProperties.JOINTYPE;

//import com.ibm.jaql.lang.expr.core.AggregateExpr.AggType;


public class SQLLops {

	// static variable to assign an unique ID to every SQLLop that is created
	private static int UniqueSQLLopID = 0;
	
	public SQLLops(String name, GENERATES flag, ValueType vt, DataType dt)
	{
		init(flag, name, vt, dt);
	}
	
	public SQLLops(String name, GENERATES flag, SQLLops input1, SQLLops input2, ValueType vt, DataType dt) {
		init(flag, name, vt, dt);
		this.getInputs().add(input1);
		this.getInputs().add(input2);
		input1.getOutputs().add(this);
		input2.getOutputs().add(this);
	}
	
	public SQLLops(String name, GENERATES flag, SQLLops input1, SQLLops input2, SQLLops input3, ValueType vt, DataType dt) {
		init(flag, name, vt, dt);
		this.getInputs().add(input1);
		this.getInputs().add(input2);
		this.getInputs().add(input3);
		input1.getOutputs().add(this);
		input2.getOutputs().add(this);
		input3.getOutputs().add(this);
	}
	
	public SQLLops(String name, GENERATES flag, SQLLops input, ValueType vt, DataType dt) {
		init(flag, name, vt, dt);
		this.getInputs().add(input);
		input.getOutputs().add(this);
	}

	private void init(GENERATES flag, String name, ValueType vt, DataType dt)
	{
		_inputs = new ArrayList<SQLLops>();
		_outputs = new ArrayList<SQLLops>();
		this._flag = flag;
		
		this.set_dataType(dt);
		this.set_valueType(vt);
		this.id = ++UniqueSQLLopID;
		this.set_tableName(name + "_" + id);
	}
	
	public static final long HMATRIXSPLIT = 1000000;
	public static final long VMATRIXSPLIT = 2000000;
	public static final String ROWCOLUMN = "row";
	public static final String COLUMNCOLUMN = "col";
	public static final String VALUECOLUMN = "value";
	public static final String SCALARVALUECOLUMN = "sval";
	
	public static final String TBLA = "alias_a";
	public static final String TBLB = "alias_b";
	
	public static final String JOIN = "INNER JOIN";
	public static final String FULLOUTERJOIN = "FULL OUTER JOIN";
	public static final String LEFTJOIN = "LEFT JOIN";
	
	public static final String ALIAS_A = "alias_a";
	public static final String ALIAS_B = "alias_b";
	public static final String ALIAS_C = "alias_c";
	
	/**
	 * Format params: Inner operation, Table 1, Table 2
	 */
	public static final String BINARYPROD = "WITH agg AS (SELECT coalesce(a.row, b.col), coalesce(a.col, b.row), %s FROM %s a"
	+" JOIN %s b ON a.col = b.row GROUP BY a.row, b.col),"
	+"num_pos AS ( SELECT sum( "
	+"DECODE( val < 0, true, 1, false, 0 ) ) %% 2 FROM agg ),"
	+"positives AS ( SELECT DECODE( val < 0, true, val * -1, false, val ) FROM agg )"
	+"SELECT decode(num_pos, 0, EXP ( SUM( LN( val ) ) ),"
	+" 1, -EXP ( SUM( LN( val ) ) ) ) FROM positives";
	
	/*
	 * Format params: Original table, maximum table, Original Table, Maximum table
	 * The result is prefereably computed in a WITH clause.
	 * Eg CTABLE:
	 * WITH maximum AS <MAXVAL2TABLES>
	 * WITH result AS <CTABLE>
	 * <ATTACHLASTZERO>
	 */
	public static final String ATTACHLASTZERO = "SELECT * FROM %s UNION ALL SELECT mrow AS row, mcol AS col, 0 as value FROM %s WHERE NOT EXISTS (SELECT * FROM %s INNER JOIN %s ON mrow = row AND mcol = col)";
	/*
	 * Format params: Table 1, Table 2, Weight matrix table
	 */
	public static final String CTABLE = "SELECT alias_a.value as row, alias_b.value as col, sum(alias_c.value) as value from %s alias_a"
+ " join %s alias_b on alias_a.row = alias_b.row AND alias_a.col = alias_b.col"
+ " join %s alias_c on alias_c.row = alias_a.row AND alias_c.col = alias_a.col group by alias_a.value, alias_b.value";
	/*
	 * Format params: Table 1, Table 2
	 */
	public static final String MAXVAL2TABLES = "SELECT max(alias_a.value) AS mrow, max(alias_b.value) AS mcol FROM %s alias_a, %s alias_b";
	/*
	 * Format params: Operation (< or >), value 1, value 2
	 */
	public static final String BINARYMAXMINOP = "decode(%s,true,%s,false,%s)";
	/**
	 * Format params: Variable to be printed
	 */
	public static final String PRINT = "RAISE NOTICE 'Message: %%', %s";
	/**
	 * Format params: operation, tables
	 */
	public static final String SIMPLESCALARSELECTFROM = "SELECT %s AS sval FROM %s";
	/**
	 * Format params: operation
	 */
	public static final String SIMPLESCALARSELECT = "SELECT %s AS sval";
	/**
	 * Format params: MAX/MIN, Table name
	 */
	public static final String UNARYMAXMIN = "SELECT %s(alias_a.value) AS sval FROM \"%s\" alias_a";
	/**
	 * Format params: Table name
	 */
	public static final String UNARYSUM = "SELECT SUM(alias_a.value) AS sval FROM \"%s\" alias_a";
	/**
	 * Format params: Table name
	 */
	public static final String UNARYROWSUM = "SELECT alias_a.row AS row, 1 AS col, SUM(alias_a.value) as value FROM \"%s\" alias_a GROUP BY alias_a.row";
	/**
	 * Format params: Table name
	 */
	public static final String UNARYCOLSUM = "SELECT 1 AS row, alias_a.col AS col, SUM(alias_a.value) as value FROM \"%s\" alias_a GROUP BY alias_a.col";
	/**
	 * Format params: Table name
	 */
	public static final String UNARYTRACE = "SELECT SUM(alias_a.value) AS sval FROM \"%s\" alias_a WHERE alias_a.col = alias_a.row";
	/**
	 * Format params: Variable name, operation
	 */
	public static final String DECLARE_VARIABLE = "%s %s;";
	/**
	 * Format params: Table name
	 */
	public static final String SELECTSTAR = "SELECT * FROM \"%s\"";
	/**
	 * Format params: < or >
	 */
	public static final String BINARYOP_MAXMIN_PART = "decode(coalesce(alias_a.value,0)%scoalesce(alias_b.value,0),true,coalesce(alias_a.value,0),false,coalesce(alias_b.value,0))";
	
	/*
	 * No format params
	 */
	public static final String BINARYMATRIXDIV_PART = "alias_a.value / decode(alias_b.value=0,true,null,false,alias_b.value)";
	/**
	 * Format params: Operation type
	 */
	public static final String BINARYOP_PART_COAL = "coalesce(alias_a.value, 0) %s coalesce(alias_b.value, 0)";
	/**
	 * Format params: Operation type
	 */
	public static final String BINARYOP_PART = "alias_a.value %s alias_b.value";
	/**
	 * Format params: Function name
	 */
	public static final String FUNCTIONOP_PART = "%s(coalesce(alias_a.value, 0), coalesce(alias_b.value, 0))";
	/**
	 * Format params: Operation, Table 1, Join type Table 2
	 */
	public static final String MATRIXMATRIXOP = "SELECT coalesce(alias_a.row, alias_b.row) AS row, coalesce(alias_a.col, alias_b.col) AS col, %s as value FROM \"%s\" alias_a %s \"%s\" alias_b ON alias_a.row = alias_b.row AND alias_a.col = alias_b.col";// WHERE alias_a.value != 0 AND alias_b.value != 0";
	/**
	 * Format params: Operation, Table, Second table or empty string
	 */
	public static final String MATRIXSCALAROP = "SELECT alias_a.row AS row, alias_a.col AS col, %s as value FROM \"%s\" alias_a %s";// WHERE alias_a.value != 0";
	/**
	 * Format params: Inner operation, Table 1, Join, Table 2
	 */
	public static final String AGGTRACEOP = "SELECT coalesce(alias_a.row, alias_b.row) AS row, coalesce(alias_a.col, alias_b.col) AS col, SUM(%s) as value FROM \"%s\" alias_a %s \"%s\" alias_b ON alias_a.row = alias_a.col AND alias_a.row = alias_b.col AND alias_a.col = alias_b.row";
	/**
	 * Format params: Inner Operations, Table 1, Table 2
	 */
	public static final String AGGSUMOP = "SELECT alias_a.row AS row, alias_b.col AS col, SUM(%s) as value FROM \"%s\" alias_a %s \"%s\" alias_b ON (alias_a.col = alias_b.row) GROUP BY alias_a.row, alias_b.col";
	/**
	 * Format params: Inner Operations, Table 1, Table 2
	 */
	public static final String SPLITAGGSUMOP = "SELECT alias_a.row AS row, alias_b.col AS col, SUM(%s) as value FROM \"%s\" alias_a %s \"%s\" alias_b ON (alias_a.col = alias_b.row) WHERE %s GROUP BY alias_a.row, alias_b.col";
	/**
	 * Format params: Outer Operation, Inner Operation, Table 1, Table 2
	 */
	public static final String AGGBINOP = "SELECT alias_a.row, alias_b.col, %s(%s) as value FROM \"%s\" alias_a %s \"%s\" alias_b ON (alias_a.col = alias_b.row) GROUP BY alias_a.row, alias_b.col";
	/**
	 * Format params: Table name
	 */
	public static final String TRANSPOSEOP = "SELECT alias_a.row AS col, alias_a.col AS row, alias_a.value as value FROM \"%s\" alias_a";
	/**
	 * Format params: Table name
	 */
	public static final String DIAG_M2VOP = "SELECT alias_a.row AS row, 1 AS col, alias_a.value as value FROM \"%s\" alias_a WHERE alias_a.row = alias_a.col";
	/**
	 * Format params: Table name
	 */
	public static final String DIAG_V2M = "SELECT alias_a.row AS row, alias_a.row AS col, alias_a.value as value FROM \"%s\" alias_a";
	/**
	 * Format params: Operation
	 */
	public static final String UNARYSCALAROP = "SELECT %s AS sval";
	/**
	 * Format params: Table name
	 */
	public static final String CASTASSCALAROP = "SELECT max(alias_a.value) AS sval FROM \"%s\" alias_a";
	/**
	 * Format params: Function name, Table name, function name
	 */
	public static final String UNARYFUNCOP = "SELECT alias_a.row as row, alias_a.col AS col, %s(alias_a.value) as value FROM \"%s\" alias_a";
	/**
	 * Format params: Unary operation, Table name, unary operation
	 */
	public static final String UNARYOP = "SELECT alias_a.row AS row, alias_a.col AS col, %salias_a.value FROM \"%s\" alias_a";
	/**
	 * Format params: Table name
	 */
	public static final String UNARYNOT = "SELECT alias_a.row AS row, alias_a.col AS col, 1 as value FROM \"%s\" alias_a WHERE alias_a.value == 0";
	/**
	 * Format params: Table name, k, Table name
	 */
	//Note: Aggregates cannot be nested without subquery
	public static final String BINCENTRALMOMENT = "SELECT (1.0 / COUNT(*)) * SUM((value - (SELECT AVG(value) FROM %s))^%s) FROM %s";
	/**
	 * Format params: Table name 1, Table name 2, Table name 1, Table name 2
	 */
	//Note: Aggregates cannot be nested without subquery
	public static final String BINCOVARIANCE = "SELECT (1.0 / COUNT(*)) * SUM((alias_a.value - (SELECT AVG(value) FROM %s)) * (alias_b.value - (SELECT AVG(value) FROM %s))) FROM %s alias_a JOIN %s alias_b ON alias_a.col = alias_b.col AND alias_a.row = alias_b.row";
	/**
	 * Format params: one-cell table with scalar value
	 */
	public static final String SELECTSCALAR = "SELECT sval FROM %s";
	/**
	 * This enumeration specifies what kind of SQL is generated by a SQLLops
	 */
	public enum GENERATES 
	{ 
		/**
		 * Does not generate any SQL, e.g. a literal or a read(...)
		 */
		NONE,
		/**
		 * Generates a stored procedure call
		 */
		PROC,
		/**
		 * Generates SQL but is a temporary table within a statement
		 */
		SQL,
		/**
		 * Generates SQL that creates a new table that is not written at end of block
		 */
		DML,
		/**
		 * Is a literal such as a number
		 */
		//LITERAL,
		/**
		 * Generates SQL and that creates a new table and overwrites
		 * an existing one that has the same name
		 * Is NOT cleaned up after the program finishes
		 */
		DML_PERSISTENT,
		/**
		 * Generates SQL and that creates a new table and overwrites
		 * an existing one that has the same name
		 * Is cleaned up after the program finishes
		 */
		DML_TRANSIENT,
		/**
		 * Prints a variable
		 */
		PRINT
	};
	
	public static String OpOp2ToString(OpOp2 op)
	{
		String opr = Hops.HopsOpOp2String.get(op);
		
		//Some exceptions:
		if(op == OpOp2.NOTEQUAL)
			opr = "<>";
		if(op == OpOp2.CONCAT)
			opr = "||";
		return opr;
	}
	
	public static String addQuotes(String s)
	{
		return String.format("\"%s\"", s);
	}
	
	public static String addCoalesce(String s)
	{
		return String.format("coalesce(%s)", s);
	}
	
	private ArrayList<SQLLops> _inputs;
	private ArrayList<SQLLops> _outputs;
	
	private VISIT_STATUS _visited = VISIT_STATUS.NOTVISITED;
	private GENERATES _flag;
	private String _sql;
	private String _tableName;
	private DataType _dataType;
	private ValueType _valueType;
	private SQLLopProperties _properties;
	private int id = 0;
	
	public int getId() {
		return id;
	}

	public SQLLopProperties get_properties() {
		return _properties;
	}

	public void set_properties(SQLLopProperties properties) {
		_properties = properties;
	}

	public DataType get_dataType() {
		return _dataType;
	}

	public void set_dataType(DataType dataType) {
		_dataType = dataType;
	}

	public ValueType get_valueType() {
		return _valueType;
	}

	public void set_valueType(ValueType valueType) {
		_valueType = valueType;
	}

	public ArrayList<SQLLops> getInputs() {
		return _inputs;
	}
	public ArrayList<SQLLops> getOutputs() {
		return _outputs;
	}
	public GENERATES get_flag() {
		return _flag;
	}
	public void set_flag(GENERATES flag) {
		_flag = flag;
	}
	public String get_sql() {
		return _sql;
	}
	public void set_sql(String sql) {
		_sql = sql;
	}
	public String get_tableName() {
		return _tableName;
	}
	
	public void set_visited(VISIT_STATUS visited) {
		_visited = visited;
	}
	
	public VISIT_STATUS get_visited() {
		return _visited;
	}
	
	public void resetVisitStatus() {
		if (this.get_visited() == Hops.VISIT_STATUS.NOTVISITED)
			return;
		for(SQLLops l : this.getInputs())
			l.resetVisitStatus();
		this.set_visited(Hops.VISIT_STATUS.NOTVISITED);
	}
	
	/**
	 * This function overwrites the unique name!
	 * @param tableName
	 */
	public void set_tableName(String tableName) {
		_tableName = tableName;
	}
	
	public int get_depth()
	{
		int max = 0;
		for(SQLLops child : this.getInputs())
		{
			if(child.get_flag() != GENERATES.DML)
				max = Math.max(max, (child.get_depth() + 1));
		}
		return max;
	}
	
	public int get_width()
	{
		int total = 0;
		if(this.getInputs().size() == 0)
			return 1;
		for(SQLLops child : this.getInputs())
		{
			if(child.get_flag() != GENERATES.DML)
				total += child.get_width();
		}
		return total;
	}
	
	public boolean treeContains(String name)
	{
		if(this.get_tableName().equals(name))
			return true;
		
		for(SQLLops child : this.getInputs())
		{
			if(child.treeContains(name))
				return true;
		}
		return false;
	}
	
	public static SQLLops getDensifySQLLop(String name, SQLLops input)
	{
		String qIn = addQuotes(input.get_tableName());
		
		SQLLops max = new SQLLops("maxrowcol", GENERATES.SQL, input, ValueType.DOUBLE, DataType.MATRIX);
		max.set_properties(new SQLLopProperties());
		max.set_sql("SELECT max(row) AS mrow, max(col) AS mcol FROM " + qIn);
		
		SQLLops lop = new SQLLops(name, GENERATES.SQL, max, ValueType.DOUBLE, DataType.MATRIX);
		lop.set_sql("SELECT zeromatrix.row AS row, zeromatrix.col AS col, coalesce(alias_a.value, 0) AS value"
				+ " FROM ( \"ZeroMatrix\" zeromatrix LEFT JOIN "
				+ qIn + " alias_a ON alias_a.row = zeromatrix.row AND alias_a.col = zeromatrix.col ) "+
				"INNER JOIN \"" + max.get_tableName() + "\" maxrowcol ON " +
				"zeromatrix.col <= maxrowcol.mcol AND zeromatrix.row <= maxrowcol.mrow");
		SQLLopProperties prop = new SQLLopProperties();
		prop.setAggType(AGGREGATIONTYPE.MAX);
		prop.setJoinType(JOINTYPE.LEFTJOIN);
		prop.setOpString("Densification");
		lop.set_properties(prop);
		return lop;
	}
}
