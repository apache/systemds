package org.apache.sysds.runtime.compress.colgroup.scheme;

import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDCLZW;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public abstract class DDCLZWScheme extends DDCScheme {
    // TODO: private int nUnique; Zu Datenspezifisch, überhaupt sinnvoll

    protected DDCLZWScheme(IColIndex cols) {
        super(cols);
    }

    public static DDCLZWScheme create(ColGroupDDCLZW g) {
        return g.getNumCols() == 1 ? new DDCLZWSchemeSC(g) : new DDCLZWSchemeMC(g);
    }

    public static DDCLZWScheme create(IColIndex cols) {
        return cols.size() == 1 ? new DDCLZWSchemeSC(cols) : new DDCLZWSchemeMC(cols);
    }

}
