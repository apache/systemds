package com.ibm.bi.dml.runtime.matrix.io;

import java.util.HashMap;

import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.utils.CacheAssignmentException;
import com.ibm.bi.dml.utils.CacheException;

public class MatrixBlock extends MatrixBlockDSM
{
	
	private MatrixObjectNew envelope = null;

	public MatrixBlock(int i, int j, boolean sparse1) {
		super(i, j, sparse1);
	}

	public MatrixBlock() {
		super();
	}
	
	public MatrixBlock(HashMap<CellIndex, Double> map) {
		super(map);
	}
	
	public static MatrixBlock randOperations(int rows, int cols, double sparsity, double min, double max, long seed)
	{
		MatrixBlock m = null;
		if ( sparsity < SPARCITY_TURN_POINT ) {
			m = new MatrixBlock(rows,cols,true);
			m.getRandomSparseMatrix(rows, cols, sparsity, min, max, seed);
		}
		else {
			m = new MatrixBlock(rows, cols, false);
			m.getRandomDenseMatrix(rows, cols, min, max, seed);
		}
		return m;
	}
	
	/*@Override
	public long getObjectSizeInMemory ()
	{
		return 0 + super.getObjectSizeInMemory ();
	}*/
	
	public MatrixObjectNew getEnvelope ()
	{
		return envelope;
	}
	
	public void clearEnvelope ()
	{
		envelope = null;
	}
	
	/**
	 * 
	 * @param newEnvelope
	 * @throws CacheAssignmentException if this matrix has already been assigned
	 *     to some other envelope.
	 */
	public void setEnvelope (MatrixObjectNew newEnvelope)
		throws CacheAssignmentException
	{
		if (envelope != null && envelope != newEnvelope)
			throw new CacheAssignmentException("MatrixBlock cannot be assigned to multiple envelopes.");
		envelope = newEnvelope; 
	}
	
	@Override
	public void finalize()
	{
		try 
		{
			if( envelope != null )
				envelope.attemptEviction( this );
		} 
		catch (CacheException e)
		{
			e.printStackTrace();
		}
		
	}
}
