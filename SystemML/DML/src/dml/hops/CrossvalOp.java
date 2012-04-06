package dml.hops;

import dml.lops.CrossvalLop;
import dml.lops.Lops;
import dml.lops.LopProperties.ExecType;
import dml.parser.MetaLearningFunctionParameters;
import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.sql.sqllops.SQLLops;
import dml.utils.HopsException;

public class CrossvalOp extends Hops {

	MetaLearningFunctionParameters _params ;

	public CrossvalOp(String l, DataType dt, ValueType vt, PartitionOp input, MetaLearningFunctionParameters params) {
		super(Hops.Kind.CrossvalOp, l, dt, vt);
		getInput().add(input) ;
		input.getParent().add(this) ;
		this._params = params ;
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += "CV";
		return s;
	}

	public void printMe() throws HopsException {
		if (get_visited() != VISIT_STATUS.DONE) {
			super.printMe();
			System.out.println("  CrossvalOp: " + "\n");
			for (Hops h : getInput()) {
				h.printMe();
			}
			;
		}
		set_visited(VISIT_STATUS.DONE);
	}

	public Lops constructLops() throws HopsException {
		if(get_lops() == null) {
			Lops pLop = getInput().get(0).constructLops() ;
			CrossvalLop cvlop = new CrossvalLop(pLop, _params) ;
			set_lops(cvlop) ;
		}
		return get_lops() ;
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected ExecType optFindExecType() throws HopsException {
		// TODO Auto-generated method stub
		return null;
	}
}
