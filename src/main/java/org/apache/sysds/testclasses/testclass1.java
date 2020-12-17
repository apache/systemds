package org.apache.sysds.testclasses;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public class testclass1 {

    static int MAX_SIZE = 4;

    public static void main(String[] args) {
        System.out.println("starting java test");
        //UtilFunctions.processData(x);


        FrameBlock f = new FrameBlock(4, Types.ValueType.fromExternalString("STRING"));
        int[] col1 = new int[MAX_SIZE];
        int[] col2 = new int[MAX_SIZE];
        int[] col3 = new int[MAX_SIZE];
        for(int i = 0; i < MAX_SIZE; i++)  {
            col1[i] = i;
            col2[i] = i + 10;
            col3[i] = i + 100;
        }

        f.appendColumn(col1);
        f.appendColumn(col2);
        f.appendColumn(col3);
        FrameBlock new_block = f.map("");


        System.out.println("finish testing");

    }

}
