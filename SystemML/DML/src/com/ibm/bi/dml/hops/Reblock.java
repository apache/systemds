package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.ReBlock;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;
import com.ibm.bi.dml.utils.HopsException;


public class Reblock extends Hops {

	// Constructor that adds a Reblock Hop *AFTER* a hop (e.g. Read Hop) to
	// produce a block size
	public Reblock(Hops inp, int rows_per_block, int cols_per_block) {
		super(Kind.Reblock, inp.get_name(), inp.get_dataType(), inp
				.get_valueType());

		set_dim1(inp.get_dim1());
		set_dim2(inp.get_dim2());
		
		set_rows_in_block(rows_per_block);
		set_cols_in_block(cols_per_block);

		getParent().addAll(inp.getParent());
		getInput().add(0, inp);

		// fix parents of input
		inp.getParent().clear();
		inp.getParent().add(this);

		// fix input of parents
		for (Hops p : getParent()) {
			for (int i = 0; i < p.getInput().size(); i++) {
				if (p.getInput().get(i) == inp) {
					p.getInput().set(i, this);
				}
			}
		}
		computeMemEstimate();
	}

	// Constructor that adds a Reblock Hop *BEFORE* a hop to create its block
	// size(e.g. Write Hop)

	public Reblock(Hops par) {
		super(Kind.Reblock, par.get_name(), par.get_dataType(), par
				.get_valueType());

		set_dim1(par.get_dim1());
		set_dim2(par.get_dim2());
		
		set_rows_in_block(par.get_rows_in_block());
		set_cols_in_block(par.get_cols_in_block());

		getParent().add(par);
		getInput().addAll(par.getInput());

		// fix inputs of parent
		par.getInput().clear();
		par.getInput().add(this);

		// fix parents of input
		for (Hops in : getInput()) {
			while (in.getParent().contains(par)) {
				in.getParent().remove(par);
				in.getParent().add(this);
			}
		}
		computeMemEstimate();
	}

	@Override
	public Lops constructLops() throws HopsException {
		if (get_lops() == null) {

			try {
				ExecType et = optFindExecType();
				if ( et == ExecType.MR ) {
					ReBlock reblock = new ReBlock(
						getInput().get(0).constructLops(),
						get_rows_in_block(), get_cols_in_block(), get_dataType(), get_valueType());
					reblock.getOutputParameters().setDimensions(get_dim1(),
							get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
		
					reblock.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
					
					set_lops(reblock);
				}
				else 
					throw new HopsException(this.printErrorLocation() + "In Reblock Hop, Invalid ExecType (" + et + ") for Reblock. \n");
			} catch ( Exception e ) {
				throw new HopsException(this.printErrorLocation() + "In Reblock Hop, error constructing Lops -- \n" + e);
			}
		}
		return get_lops();
	}

	public void printMe() throws HopsException {
		if (get_visited() != VISIT_STATUS.DONE) {
			super.printMe();
			for (Hops h : getInput()) {
				h.printMe();
			}
		}
		set_visited(VISIT_STATUS.DONE);
	}
	
	@Override
	public String getOpString() {
		return "reblock";
	}

	@Override
	public SQLLops constructSQLLOPs() throws HopsException {
		if(this.getInput().size() != 1)
			throw new HopsException(this.printErrorLocation() + "Reblock needs one input");
		
		GENERATES flag = determineGeneratesFlag();
		
		//Just pass the input SQLLops, there is no reblocking in Netezza,
		//However, we have to adjust the flag
		Hops input = this.getInput().get(0);
		SQLLops lop = input.constructSQLLOPs();
		
		if(lop.get_flag() != GENERATES.NONE)
			lop.set_flag(flag);
		
		this.set_sqllops(lop);
		return this.get_sqllops();
	}
	
	@Override
	public boolean allowsAllExecTypes()
	{
		return false;
	}

	@Override
	public double computeMemEstimate() {
		
		/* This method should never be invoked while deciding CP vs. MR */
		
		_outputMemEstimate = getInput().get(0).getOutputSize();
		_memEstimate = getInputOutputSize();
		return _memEstimate;
	}
	

	@Override
	protected ExecType optFindExecType() throws HopsException {
		if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE )
			throw new HopsException(this.printErrorLocation() + "In Reblock Hop, REBLOCKing is an invalid operation when execution mode = SINGLE_NODE \n");
		
		if( _etypeForced != null & _etypeForced != ExecType.MR ) 			
			throw new HopsException(this.printErrorLocation() + "In Reblock Hop, REBLOCKing is an invalid operation when execution mode = SINGLE_NODE \n");
		
		// Reblock operation always gets executed in MR. 
		// It may not be meaningful to perform it in CP.
		_etype = ExecType.MR;
		return _etype;
	}

}
