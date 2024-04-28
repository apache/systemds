package org.apache.sysds.api.ropt.old_impl;

public class CloudOptimizerUtils {

    public static long toB( long mb )
    {
        return 1024 * 1024 * mb;
    }

    public static long toMB( long b )
    {
        return b / (1024 * 1024);
    }
}
