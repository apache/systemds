/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.ReBlock;
import com.ibm.bi.dml.lops.CSVReBlock;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;


public class ReblockOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private ReblockOp() {
		//default constructor for clone
	}
	
	// Constructor that adds a Reblock Hop *AFTER* a hop (e.g. Read Hop) to
	// produce a block size
	public ReblockOp(Hop inp, int rows_per_block, int cols_per_block) {
		super(Kind.Reblock, inp.get_name(), inp.get_dataType(), inp
				.get_valueType());

		set_dim1(inp.get_dim1());
		set_dim2(inp.get_dim2());
		
		set_rows_in_block(rows_per_block);
		set_cols_in_block(cols_per_block);

		setNnz(inp.getNnz());
		
		getParent().addAll(inp.getParent());
		getInput().add(0, inp);

		// fix parents of input
		inp.getParent().clear();
		inp.getParent().add(this);

		// fix input of parents
		for (Hop p : getParent()) {
			for (int i = 0; i < p.getInput().size(); i++) {
				if (p.getInput().get(i) == inp) {
					p.getInput().set(i, this);
				}
			}
		}
	}

	// Constructor that adds a Reblock Hop *BEFORE* a hop to create its block
	// size(e.g. Write Hop)

	public ReblockOp(Hop par) {
		super(Kind.Reblock, par.get_name(), par.get_dataType(), par
				.get_valueType());

		set_dim1(par.get_dim1());
		set_dim2(par.get_dim2());
		
		set_rows_in_block(par.get_rows_in_block());
		set_cols_in_block(par.get_cols_in_block());
		
		setNnz(par.getNnz());

		getParent().add(par);
		getInput().addAll(par.getInput());

		// fix inputs of parent
		par.getInput().clear();
		par.getInput().add(this);

		// fix parents of input
		for (Hop in : getInput()) {
			while (in.getParent().contains(par)) {
				in.getParent().remove(par);
				in.getParent().add(this);
			}
		}
	}
	
	@Override
	public Lop constructLops() 
		throws HopsException, LopsException 
	{
		if (get_lops() == null) {

			try {
				ExecType et = optFindExecType();
				if ( et == ExecType.MR ) {
					Hop input = getInput().get(0);
					
					// Create the reblock lop according to the format of the input hop
					if ( input.getKind() == Kind.DataOp 
							&& ((DataOp)input).get_dataop() == DataOpTypes.PERSISTENTREAD
							&& ((DataOp)input).getFormatType() == FileFormatTypes.CSV ) {
						// NOTE: only persistent reads can have CSV format
						CSVReBlock rcsv = new CSVReBlock(
								getInput().get(0).constructLops(),
								get_rows_in_block(), get_cols_in_block(), get_dataType(), get_valueType());
							rcsv.getOutputParameters().setDimensions(get_dim1(),
									get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
				
							rcsv.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
							
							set_lops(rcsv);
					}
					else {
						ReBlock reblock = new ReBlock(
							getInput().get(0).constructLops(),
							get_rows_in_block(), get_cols_in_block(), get_dataType(), get_valueType());
						reblock.getOutputParameters().setDimensions(get_dim1(),
								get_dim2(), get_rows_in_block(), get_cols_in_block(), getNnz());
			
						reblock.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
						
						set_lops(reblock);
					}
				}
				else 
					throw new HopsException(this.printErrorLocation() + "In Reblock Hop, Invalid ExecType (" + et + ") for Reblock. \n");
			} catch ( Exception e ) {
				throw new HopsException(this.printErrorLocation() + "In Reblock Hop, error constructing Lops -- \n" , e);
			}
		}
		
		return get_lops();
	}

	public void printMe() throws HopsException {
		if (get_visited() != VISIT_STATUS.DONE) {
			super.printMe();
			for (Hop h : getInput()) {
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
		Hop input = this.getInput().get(0);
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
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		return 0;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		long[] ret = null;
		
		Hop input = getInput().get(0);
		MatrixCharacteristics mc = memo.getAllInputStats(input);
		if( mc != null ) 
			ret = new long[]{mc.get_rows(), mc.get_cols(), mc.getNonZeros()};
		
		return ret;
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

	@Override
	public void refreshSizeInformation()
	{
		Hop input1 = getInput().get(0);
		set_dim1( input1.get_dim1() );
		set_dim2( input1.get_dim2() );
		setNnz( input1.getNnz() );
	}

	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		ReblockOp ret = new ReblockOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes

		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( that._kind!=Kind.Reblock )
			return false;
		
		ReblockOp that2 = (ReblockOp)that;	
		return (   _rows_in_block == that2._rows_in_block
				&& _cols_in_block == that2._cols_in_block
				&& getInput().get(0) == that2.getInput().get(0));
	}
}
