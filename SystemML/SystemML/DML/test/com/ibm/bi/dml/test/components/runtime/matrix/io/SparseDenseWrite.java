/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.components.runtime.matrix.io;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.DataOutput;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.runtime.controlprogram.caching.CacheDataOutput;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.test.utils.TestUtils;


public class SparseDenseWrite 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private int _rows = 5132;
	private int _cols = 5079;
	
	private double _sparsity = 0.5d;
				
	@Test
	public void testMatrixSparse2DenseWrite() 
	{	
		try 
		{
			//generate sparse matrix
			double[][] matrix = TestUtils.generateTestMatrix(_rows, _cols, 0, 1, _sparsity, 7);
			MatrixBlock mb = new MatrixBlock(_rows, _cols, true);
			for( int i=0; i<_rows; i++ )
				for( int j=0; j<_cols; j++ )
					mb.quickSetValue(i, j, matrix[i][j]);
			
			//serialize matrix into dense representation
			byte[] bbuff = new byte[(int)mb.getExactSizeOnDisk()];
			
			for( int i=0; i<50; i++ ){ //performance after JIT
				Timing time = new Timing(true);
				DataOutput dout = new CacheDataOutput(bbuff);
				mb.write(dout);
				System.out.println("Serialized matrix (sparse to dense) in "+time.stop()+"ms");
			}
			//deserialize matrix
			ByteArrayInputStream bis = new ByteArrayInputStream(bbuff);
			DataInputStream din = new DataInputStream(bis); 
			MatrixBlock mb2 = new MatrixBlock();
			mb2.readFields(din);
			double[][] matrix2 = DataConverter.convertToDoubleMatrix(mb2);
			
			
			/* NOTE: this formulation prevents sparse-dense write via exam sparsity
			 
			//serialize matrix into dense representation
			ByteBuffer bbuff = new ByteBuffer((int)mb.getExactSizeOnDisk());
			
			Timing time = new Timing(true);
			bbuff.serializeMatrix(mb);
			System.out.println("Serialized matrix in "+time.stop()+"ms");
			
			//deserialize matrix
			MatrixBlock mb2 = bbuff.deserializeMatrix();
			double[][] matrix2 = DataConverter.convertToDoubleMatrix(mb2);
			
			*/
			
			//compare matrices
			for( int i=0; i<_rows; i++ )
				for( int j=0; j<_cols; j++ )
					if( matrix[i][j]!=matrix2[i][j] )
						Assert.fail("Wrong value i="+i+", j="+j+", value1="+matrix[i][j]+", value2="+matrix2[i][j]);
		
		} 
		catch (Exception e) 
		{
			throw new RuntimeException(e);
		}
	}
}
