package dml.test.components.runtime.matrix;

import static org.junit.Assert.*;

import org.junit.Test;

import dml.runtime.matrix.MetaData;
import dml.runtime.matrix.ReblockMR;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.test.BinaryMatrixCharacteristics;
import dml.test.utils.TestUtils;

public class ReblockMRTest {

    @Test
    public void testRunJobOneInput() {
    	
    /*	String infile="testReblock.in";
    	String outfile="testReblock.out";
    	int rows=10;
		int cols=10;
		double min=1;
		double max=5;
		double sparsity=0.5;
		long seed=1;
		double[][] m=TestUtils.generateTestMatrix(rows, cols, min, max, sparsity, seed);
		TestUtils.writeTestMatrix(infile, m);
    	
		int numReducers=3;
		int replication=1;
		int bnr=2;
		int bnc=2;
		
		try {
			MetaData[] md=ReblockMR.runJob(new String[]{infile}, new InputInfo[]{InputInfo.TextCellInputInfo}, 
					new long[]{rows}, new long[]{cols}, new int[]{1}, new int[]{1}, 
					"", "rblk 0 0 "+bnr+" "+bnc, "", 
					numReducers, replication, new byte[]{0}, new byte[]{0}, new String[]{outfile}, 
					new OutputInfo[]{OutputInfo.TextCellOutputInfo}).getMetaData();
			//BinaryMatrixCharacteristics out=TestUtils.readBlocksFromSequenceFile(outfile, bnr, bnc);
			//TestUtils.compareMatrices(m, out.getValues(), rows, cols, 0.0001);
			TestUtils.compareFilesInDifferentOrder(infile, outfile, rows, cols, 0.0001);
		} catch (Exception e) {
			fail(e.toString());
		}*/
    }
    
    @Test
    public void testRunJobMultipleInputs() {
    	
    /*	String infile1="testReblock.in1";
    	String infile2="testReblock.in2";
    	String outfile1="testReblock.out1";
    	String outfile2="testReblock.out2";
    	int rows=10;
		int cols=10;
		double min=1;
		double max=5;
		double sparsity=0.5;
		long seed=1;
		double[][] m1=TestUtils.generateTestMatrix(rows, cols, min, max, sparsity, seed);
		double[][] m2=TestUtils.generateTestMatrix(rows, cols, min, max, sparsity, seed);
		TestUtils.writeTestMatrix(infile1, m1);
		TestUtils.writeTestMatrix(infile2, m2);
    	
		int numReducers=3;
		int replication=1;
		int bnr=2;
		int bnc=2;
		
		try {
			MetaData[] md=ReblockMR.runJob(new String[]{infile1, infile2}, new InputInfo[]{InputInfo.TextCellInputInfo, InputInfo.TextCellInputInfo}, 
					new long[]{rows, rows}, new long[]{cols, cols}, new int[]{1, 1}, new int[]{1, 1}, 
					"", "rblk 0 0 "+bnr+" "+bnc+",rblk 1 1 "+bnr+" "+bnc, "", 
					numReducers, replication, new byte[]{0, 1}, new byte[]{0,0}, new String[]{outfile1, outfile2}, 
					new OutputInfo[]{OutputInfo.TextCellOutputInfo, OutputInfo.BinaryBlockOutputInfo}).getMetaData();
			TestUtils.compareFilesInDifferentOrder(infile1, outfile1, rows, cols, 0.0001);
			BinaryMatrixCharacteristics out=TestUtils.readBlocksFromSequenceFile(outfile2, bnr, bnc);
			TestUtils.compareMatrices(m2, out.getValues(), rows, cols, 0.0001);
			
		} catch (Exception e) {
			fail(e.toString());
		}*/
    }
    
    @Test
    public void testRunJobNoMixedInputFormats() {
    	
    /*	String infile1="testReblock.in1";
    	String infile2="testReblock.in2";
    	int rows=10;
		int cols=10;
		double min=1;
		double max=5;
		double sparsity=0.5;
		long seed=1;
		double[][] m=TestUtils.generateTestMatrix(rows, cols, min, max, sparsity, seed);
		TestUtils.writeTestMatrix(infile1, m);
    	
		int numReducers=3;
		int replication=1;
		int bnr=2;
		int bnc=2;
		
		try {
			MetaData[] md=ReblockMR.runJob(new String[]{infile1}, new InputInfo[]{InputInfo.TextCellInputInfo}, 
					new long[]{rows}, new long[]{cols}, new int[]{1}, new int[]{1}, 
					"", "rblk 0 0 "+bnr+" "+bnc, "", 
					numReducers, replication, new byte[]{0}, new byte[]{0}, new String[]{infile2}, 
					new OutputInfo[]{OutputInfo.TextCellOutputInfo}).getMetaData();
			TestUtils.compareFilesInDifferentOrder(infile1, infile2, rows, cols, 0.0001);
		} catch (Exception e) {
			fail(e.toString());
		}
		
		bnr=3;
		bnc=3;
		String outfile1="testReblock.out1";
    	String outfile2="testReblock.out2";
		try {
			MetaData[] md=ReblockMR.runJob(new String[]{infile1, infile2}, new InputInfo[]{InputInfo.TextCellInputInfo, InputInfo.BinaryBlockInputInfo}, 
					new long[]{rows, rows}, new long[]{cols, cols}, new int[]{1, 1}, new int[]{1, 1}, 
					"", "rblk 0 0 "+bnr+" "+bnc+",rblk 1 1 "+bnr+" "+bnc, "", 
					numReducers, replication, new byte[]{0, 1}, new byte[]{0,0}, new String[]{outfile1, outfile2}, 
					new OutputInfo[]{OutputInfo.TextCellOutputInfo, OutputInfo.BinaryBlockOutputInfo}).getMetaData();
			TestUtils.compareFilesInDifferentOrder(infile1, outfile1, rows, cols, 0.0001);
			BinaryMatrixCharacteristics out=TestUtils.readBlocksFromSequenceFile(outfile2, bnr, bnc);
			TestUtils.compareMatrices(m, out.getValues(), rows, cols, 0.0001);
			
		} catch (Exception e) {
			System.out.println("Error expected: don't supported sequenceFile and textFile mixed for reblock");
		}*/
    }

    public void print(double[][] m, int rows, int cols)
    {
    	for(int i=0; i<rows; i++)
    	{
    		for(int j=0; j<cols; j++)
    			System.out.print(m[i][j]+"\t");
    		System.out.println();
    	}
    }
}
