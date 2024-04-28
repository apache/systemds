package org.apache.sysds.api.ropt;

public class CloudOptimizerUtils {
    public static final double MEM_FACTOR = 1.5;

    public static long toB( long mb )
    {
        return 1024 * 1024 * mb;
    }

    public static long toMB( long b )
    {
        return b / (1024 * 1024);
    }
}
