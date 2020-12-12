package org.apache.sysds.testclasses;

import org.apache.sysds.runtime.util.UtilFunctions;

public class testclass1 {

    public static void main(String[] args) {
        System.out.println("starting java test");
        String x = new String("hello");
        UtilFunctions.processData(x);
    }
}
