package org.apache.sysds.runtime.transform.tokenize;

import org.apache.sysds.runtime.matrix.data.FrameBlock;

import java.io.Serializable;

public interface TokenizerPre extends Serializable {
    Tokenizer.DocumentsToTokenList tokenizePre(FrameBlock in);
}
