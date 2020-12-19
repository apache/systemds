package org.apache.sysds.testclasses;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public class testclass1 {

    static int MAX_SIZE = 4;

    public static void main(String[] args) {
        System.out.println("starting java test");
        //UtilFunctions.processData(x);

        FrameBlock f = new FrameBlock();
        f.appendColumn(new String[]{"1","2","3","4","5"});
        f.appendColumn(new String[]{"1","1","1","1","2"});
        f.appendColumn(new String[]{"Hllo","asdasd","asd","asd","asd"});

        UtilFunctions.calculateAttributeTypes(f);


        System.out.println("finish testing");

    }

}
