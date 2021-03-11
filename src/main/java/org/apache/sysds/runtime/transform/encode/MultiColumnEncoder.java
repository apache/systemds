package org.apache.sysds.runtime.transform.encode;

import org.apache.sysds.runtime.matrix.data.FrameBlock;

import java.util.List;

public class MultiColumnEncoder {

    private List<Encoder> _encoders = null;
    private FrameBlock _meta = null;


    public MultiColumnEncoder(List<Encoder> encoders){
        _encoders = encoders;
    }

}
