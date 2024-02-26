package org.apache.sysds.test.component.sparse;

import org.apache.sysds.test.AutomatedTestBase;


import org.apache.sysds.runtime.data.SparseBlockDCSR;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCOO;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

import java.util.Iterator;
import java.util.Arrays;

public class SandBox extends AutomatedTestBase {

    private final static int rows = 8; //324
    private final static int cols = 6; //100
    private final static int rlPartial = 134;
    private final static double sparsity1 = 0.1;
    private final static double sparsity2 = 0.2;
    private final static double sparsity3 = 0.3;

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
    }



    @Test
    public void executeTest(){
        testMethod(SparseBlock.Type.COO, false);
    }
    private void testMethod( SparseBlock.Type btype, boolean partial)
    {
        try {
            //data generation
            //double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 8765432);
            double[][] A = {{0, 0, 0, 0, 0, 0}, // Row 1: All zeros
                    {10, 20, 0, 0, 0, 0}, // Row 2: Non-zero
                    {0, 0, 0, 0, 0, 0}, // Row 3: All zeros
                    {0, 30, 0, 40, 0, 0}, // Row 4: Non-zero
                    {0, 0, 0, 0, 0, 0}, // Row 5: All zeros
                    {9, 3, 50, 60, 70, 0}, // Row 6: Non-zero
                    {0, 0, 0, 0, 0, 0}, // Row 7: All zeros
                    {0, 0, 0, 0, 0, 80} // Row 8: Non-zero
            };

            double[][] B = {{0, 0, 0, 0, 0, 0}, // Row 1: All zeros
                    {10, 20, 0, 0, 0, 0}, // Row 2: Non-zero
                    {0, 0, 0, 0, 0, 0}, // Row 3: All zeros
                    {0, 30, 0, 40, 0, 0}, // Row 4: Non-zero
                    {0, 0, 0, 0, 0, 0}, // Row 5: All zeros
                    {0, 0, 50, 60, 70, 0}, // Row 6: Non-zero
                    {0, 0, 0, 0, 0, 75}, // Row 7: All zeros
                    {0, 0, 0, 0, 0, 80} // Row 8: Non-zero
            };

            double[][] C = {{0, 0, 0, 0, 0, 0}, // Row 1: All zeros
                    {0, 0, 0, 0, 0, 0}, // Row 2: Non-zero
                    {0, 0, 0, 0, 0, 0}, // Row 3: All zeros
                    {0, 0, 0, 0, 0, 0}, // Row 4: Non-zero
                    {0, 10, 10, 10, 10, 0}, // Row 5: All zeros
                    {0, 0, 0, 0, 0, 0}, // Row 6: Non-zero
                    {0, 0, 0, 0, 0, 0}, // Row 7: All zeros
                    {0, 0, 0, 0, 0, 0} // Row 8: Non-zero
            };

            //init sparse block
            SparseBlock sblock = null;
            MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(C);
            SparseBlock srtmp = mbtmp.getSparseBlock();
            switch(btype) {
                case MCSR:
                    sblock = new SparseBlockMCSR(srtmp);
                    break;
                case CSR:
                    sblock = new SparseBlockCSR(srtmp);
                    break;
                case COO:
                    sblock = new SparseBlockCOO(srtmp);
                    break;
                case DCSR:
                    sblock = new SparseBlockDCSR(srtmp);
                    break;
            }
            //System.out.println(SparseBlock.Type.DCSR.getClass().getName());
            //Assert.assertTrue(1==1);
            //System.out.println(sblock.size());

            Iterator<Integer> iter = sblock.getIteratorNonZeroRows(0,8);
            //System.out.println(iter.next());
            //int nextRow = -1;
            while(iter.hasNext()){
                //nextRow = iter.next();
                System.out.println(iter.next());
            }
            //System.out.println(iter.next());
            //System.out.println(nextRow);


        }

        catch(Exception ex) {
            ex.printStackTrace();
            throw new RuntimeException(ex);
        }
    }
}
