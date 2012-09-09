package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.PartitionLop;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.meta.PartitionParams;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LopsException;


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
	public Lops constructLops() throws HopsException, LopsException {
		if(get_lops() == null) {
			Lops dataLop = getInput().get(0).constructLops();
			PartitionLop pLop = new PartitionLop(pp, dataLop, get_dataType(), get_valueType()) ;
			pLop.getOutputParameters().setDimensions(getInput().get(0).get_dim1(),
					getInput().get(0).get_dim2(),  
					getInput().get(0).get_rows_in_block(), 
					getInput().get(0).get_cols_in_block(),
					getInput().get(0).getNnz()) ;
			pLop.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			set_lops(pLop) ;
		}
		return get_lops() ;
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public boolean allowsAllExecTypes()
	{
		return false;
	}

	@Override
	protected ExecType optFindExecType() throws HopsException {
		// TODO Auto-generated method stub
		return null;
	}
}
