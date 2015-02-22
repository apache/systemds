/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.CSVReBlock;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.sql.sqllops.SQLLops;
import com.ibm.bi.dml.sql.sqllops.SQLLops.GENERATES;


public class ReblockOp extends Hop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private ReblockOp() {
		//default constructor for clone
	}
	
	// Constructor that adds a Reblock Hop *AFTER* a hop (e.g. Read Hop) to
	// produce a block size
	public ReblockOp(Hop inp, int rows_per_block, int cols_per_block) {
		super(Kind.ReblockOp, inp.getName(), inp.getDataType(), inp
				.getValueType());

		setDim1(inp.getDim1());
		setDim2(inp.getDim2());
		
		setRowsInBlock(rows_per_block);
		setColsInBlock(cols_per_block);

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
		super(Kind.ReblockOp, par.getName(), par.getDataType(), par
				.getValueType());

		setDim1(par.getDim1());
		setDim2(par.getDim2());
		
		setRowsInBlock(par.getRowsInBlock());
		setColsInBlock(par.getColsInBlock());
		
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
		if (getLops() == null) {

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
								getRowsInBlock(), getColsInBlock(), getDataType(), getValueType());
							rcsv.getOutputParameters().setDimensions(getDim1(),
									getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
				
							rcsv.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
							
							setLops(rcsv);
					}
					else //TEXT / BINARYBLOCK / BINARYCELL  
					{
						setLops( getInput().get(0).constructLops() );
						setRequiresReblock(true);
						
						// construct and set reblock lop as current root lop
						constructAndSetReblockLopIfRequired();
					}
				}
				else 
					throw new HopsException(this.printErrorLocation() + "In Reblock Hop, Invalid ExecType (" + et + ") for Reblock. \n");
			} catch ( Exception e ) {
				throw new HopsException(this.printErrorLocation() + "In Reblock Hop, error constructing Lops -- \n" , e);
			}
		}
		
		return getLops();
	}

	public void printMe() throws HopsException {
		if (getVisited() != VisitStatus.DONE) {
			super.printMe();
			for (Hop h : getInput()) {
				h.printMe();
			}
		}
		setVisited(VisitStatus.DONE);
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
		
		this.setSqlLops(lop);
		return this.getSqlLops();
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
			ret = new long[]{mc.getRows(), mc.getCols(), mc.getNonZeros()};
		
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
		setDim1( input1.getDim1() );
		setDim2( input1.getDim2() );
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
		if( that._kind!=Kind.ReblockOp )
			return false;
		
		ReblockOp that2 = (ReblockOp)that;	
		return (   _rows_in_block == that2._rows_in_block
				&& _cols_in_block == that2._cols_in_block
				&& getInput().get(0) == that2.getInput().get(0));
	}
}
