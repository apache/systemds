package dml.hops;

import dml.lops.Lops;
import dml.lops.PartitionLop;
import dml.meta.PartitionParams;
import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.sql.sqllops.SQLLops;
import dml.utils.HopsException;

public class PartitionOp extends Hops {
	PartitionParams pp ;
	
	public PartitionOp(String l, DataType dt, ValueType vt, PartitionParams pp, DataOp input) {
		super(Hops.Kind.PartitionOp, l, dt, vt);
		getInput().add(input) ;
		input.getParent().add(this) ;
		
		this.pp = pp ;
	}

	@Override
	public String getOpString() {
		return pp.toString();
	}
	
	public void printMe() throws HopsException {
		if (get_visited() != VISIT_STATUS.DONE) {
			super.printMe();
			System.out.println("  Partition: " + pp.toString() + "\n");
			for (Hops h : getInput()) {
				h.printMe();
			}
		}
		set_visited(VISIT_STATUS.DONE);
	}

	@Override
	public Lops constructLops() throws HopsException {
		if(get_lops() == null) {
			Lops dataLop = getInput().get(0).constructLops();
			PartitionLop pLop = new PartitionLop(pp, dataLop, get_dataType(), get_valueType()) ;
			pLop.getOutputParameters().setDimensions(getInput().get(0).get_dim1(),
					getInput().get(0).get_dim2(), getInput().get(0).get_rows_per_block(), 
					getInput().get(0).get_cols_per_block()) ;
			set_lops(pLop) ;
		}
		return get_lops() ;
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		// TODO Auto-generated method stub
		return null;
	}
}
