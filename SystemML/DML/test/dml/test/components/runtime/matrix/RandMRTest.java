package dml.test.components.runtime.matrix;

import org.junit.Test;

import dml.runtime.matrix.RandMR;
import dml.runtime.matrix.io.OutputInfo;
import dml.test.BinaryMatrixCharacteristics;
import dml.test.utils.TestUtils;

public class RandMRTest {

    @Test
    public void testRunJob() {
    	long[] numRows = new long[] { 10, 50 };
    	long[] numCols = new long[] { 10, 50 };
    	int[] blockRowSize = new int[] { 2, 10000 };
    	int[] blockColSize = new int[] { 2, 10000 };
    	double[] minValue = new double[] { -1, 10 };
    	double[] maxValue = new double[] { 1, 20 };
    	double[] sparsity = new double[] { 1, 0.1 };
    	String[] pdf = new String[] { "uniform", "uniform" };
    	int replication = 1;
    	String[] inputs = new String[] { "temp_randmrtest_in_0", "temp_randmrtest_in_1" };
    	String[] outputs = new String[] { "temp_randmrtest_out_0", "temp_randmrtest_out_1" };
    	OutputInfo[] outputInfos = new OutputInfo[] { OutputInfo.BinaryBlockOutputInfo,
    			OutputInfo.BinaryBlockOutputInfo };
    	String instructionsInMapper = "Rand 0 2 min=-1 max=1 sparsity=1 pdf=uniform," +
    			"Rand 1 3 min=10 max=20 sparsity=0.1 pdf=uniform";
    	byte[] resultIndexes = new byte[] { 2, 3 };
    	byte[] resultDimsUnknown = new byte[] {0,0};
    	
    	try {
			RandMR.runJob(numRows, numCols, blockRowSize, blockColSize, minValue, maxValue, sparsity, pdf, replication,
					inputs, outputs, outputInfos, instructionsInMapper, resultIndexes, resultDimsUnknown);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		BinaryMatrixCharacteristics randMatrix = TestUtils.readBlocksFromSequenceFile("temp_randmrtest_out_0",
				blockRowSize[0], blockColSize[0]);
		TestUtils.checkMatrix(randMatrix, numRows[0], numCols[0], minValue[0], maxValue[0]);
		randMatrix = TestUtils.readBlocksFromSequenceFile("temp_randmrtest_out_1",
				blockRowSize[1], blockColSize[1]);
		TestUtils.checkMatrix(randMatrix, numRows[1], numCols[1], minValue[1], maxValue[1]);
		
		TestUtils.removeTemporaryFiles();
    }

}
