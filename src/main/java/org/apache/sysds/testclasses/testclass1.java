package org.apache.sysds.testclasses;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public class testclass1 {

    static int MAX_SIZE = 4;

    public static void main(String[] args) {
        FrameBlock f = new FrameBlock();
        f.appendColumn(new String[]{"77","77","89","89","43", "xx", "123"});
        f.appendColumn(new String[]{"100","xx","4","34","321", "xx", "123"});
        f.appendColumn(new String[]{"Hllo","asdasd","3asd","asd","asd", "xx", "123"});
        f.appendColumn(new String[]{"44","3","235","52","weg", "xx", "123"});

        UtilFunctions.calculateAttributeTypes(f);
    }

}
