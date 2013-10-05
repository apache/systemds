/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.lops.CrossvalLop;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.sql.sqllops.SQLLops;


public class CrossvalOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private CrossvalOp() {
		//default constructor for clone
	}
	
	public CrossvalOp(String l, DataType dt, ValueType vt, PartitionOp input) {
		super(Hop.Kind.CrossvalOp, l, dt, vt);
		getInput().add(input) ;
		input.getParent().add(this) ;
	}

	@Override
	public String getOpString() {
		String s = new String("");
		s += "CV";
		return s;
	}

	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (get_visited() != VISIT_STATUS.DONE) {
				super.printMe();
				LOG.debug("  CrossvalOp: ");
				for (Hop h : getInput()) {
					h.printMe();
				}
			}
			set_visited(VISIT_STATUS.DONE);
		}
	}

	public Lop constructLops() throws HopsException, LopsException {
		if(get_lops() == null) {
			Lop pLop = getInput().get(0).constructLops() ;
			CrossvalLop cvlop = new CrossvalLop(pLop) ;
			cvlop.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
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
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		return OptimizerUtils.INVALID_SIZE;
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		return OptimizerUtils.INVALID_SIZE;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
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
	
	@Override
	public void refreshSizeInformation()
	{
		// TODO modify whenever CL/EL integrated into the optimizer
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		CrossvalOp ret = new CrossvalOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		// TODO modify whenever CL/EL integrated into the optimizer
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		return false;
	}
}
