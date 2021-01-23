package org.apache.sysds.runtime.transform.tokenize;

import org.apache.sysds.runtime.matrix.data.FrameBlock;

import java.io.Serializable;

public interface TokenizerPost extends Serializable {
    FrameBlock tokenizePost(Tokenizer.DocumentsToTokenList tl, FrameBlock out);
}
