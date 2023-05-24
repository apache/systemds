package org.apache.sysds.runtime.compress.readers;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.utils.DblArray;

public class ReaderColumnSelectionDenseDeltaSingleBlock extends ReaderColumnSelection{
    //TODO Confirm specifics and requirements for the class
    protected ReaderColumnSelectionDenseDeltaSingleBlock(IColIndex colIndexes, int rl, int ru) {
        super(colIndexes, rl, ru);
        throw new NotImplementedException();
    }

    @Override
    protected DblArray getNextRow() {
        throw new NotImplementedException();
    }
}
