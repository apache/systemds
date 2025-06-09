package org.apache.sysds.runtime.transform.decode;
import org.apache.sysds.common.Types.ValueType;

/**
 * helper class f
 */
public class ColumnInput {
    public ColumnBlock columnBlock;

    //
    public ValueType[] schema;
    public ColumnDecoder decoder;
}
