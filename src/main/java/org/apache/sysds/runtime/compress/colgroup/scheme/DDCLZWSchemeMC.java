package org.apache.sysds.runtime.compress.colgroup.scheme;

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDCLZW;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.ACount;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
import org.apache.sysds.runtime.compress.utils.DoubleCountHashMap;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;

public class DDCLZWSchemeMC extends DDCLZWScheme {
    //private DDCSchemeMC ddcscheme;
    private final DblArray emptyRow;

    private final DblArrayCountHashMap map;

    private DDCLZWSchemeMC(IColIndex cols, DblArrayCountHashMap map) {
        super(cols);
        this.map = map;
        this.emptyRow = new DblArray(new double[cols.size()]);
    }

    protected DDCLZWSchemeMC(ColGroupDDCLZW g) {
        super(g.getColIndices());
        this.lastDict = g.getDictionary();
        final MatrixBlock mbDict = lastDict.getMBDict(this.cols.size()).getMatrixBlock();
        final int dictRows = mbDict.getNumRows();
        final int dictCols = mbDict.getNumColumns();

        // Read the mapping data and materialize map.
        map = new DblArrayCountHashMap(dictRows * 2);
        final ReaderColumnSelection r = ReaderColumnSelection.createReader(mbDict, //
                ColIndexFactory.create(dictCols), false, 0, dictRows);

        DblArray d = null;
        while((d = r.nextRow()) != null)
            map.increment(d);

        emptyRow = new DblArray(new double[dictCols]);
    }

    protected DDCLZWSchemeMC(IColIndex cols) {
        super(cols);
        final int nCol = cols.size();
        this.map = new DblArrayCountHashMap(4);
        this.emptyRow = new DblArray(new double[nCol]);
    }

    @Override
    protected AColGroup encodeV(MatrixBlock data, IColIndex columns) {
        final int nRow = data.getNumRows();
        final ReaderColumnSelection reader = ReaderColumnSelection.createReader(//
                data, columns, false, 0, nRow);
        return encode(data, reader, nRow, columns);
    }

    @Override
    protected AColGroup encodeVT(MatrixBlock data, IColIndex columns) {
        final int nRow = data.getNumColumns();
        final ReaderColumnSelection reader = ReaderColumnSelection.createReader(//
                data, columns, true, 0, nRow);
        return encode(data, reader, nRow, columns);
    }

    private AColGroup encode(MatrixBlock data, ReaderColumnSelection reader, int nRow, IColIndex columns) {
        final AMapToData d = MapToFactory.create(nRow, map.size());
        DblArray cellVals;
        ACount<DblArray> emptyIdx = map.getC(emptyRow);
        if(emptyIdx == null) {

            while((cellVals = reader.nextRow()) != null) {
                final int row = reader.getCurrentRowIndex();

                final int id = map.getId(cellVals);
                d.set(row, id);

            }
        }
        else {
            int r = 0;
            while((cellVals = reader.nextRow()) != null) {
                final int row = reader.getCurrentRowIndex();
                if(row != r) {
                    while(r < row)
                        d.set(r++, emptyIdx.id);
                }
                final int id = map.getId(cellVals);
                d.set(row, id);
                r++;
            }
            while(r < nRow)
                d.set(r++, emptyIdx.id);
        }
        if(lastDict == null || lastDict.getNumberOfValues(columns.size()) != map.size())
            lastDict = DictionaryFactory.create(map, columns.size(), false, data.getSparsity());
        return ColGroupDDCLZW.create(columns, lastDict, d, null);

    }


    @Override
    protected ICLAScheme updateV(MatrixBlock data, IColIndex columns) {
        final int nRow = data.getNumRows();
        final ReaderColumnSelection reader = ReaderColumnSelection.createReader(//
                data, columns, false, 0, nRow);
        return update(data, reader, nRow, columns);
    }

    private ICLAScheme update(MatrixBlock data, ReaderColumnSelection reader, int nRow, IColIndex columns) {
        DblArray d = null;
        int r = 0;
        while((d = reader.nextRow()) != null) {
            final int cr = reader.getCurrentRowIndex();
            if(cr != r) {
                map.increment(emptyRow, cr - r);
                r = cr;
            }
            map.increment(d);
            r++;
        }
        if(r < nRow)
            map.increment(emptyRow, nRow - r - 1);

        return this;
    }


    @Override
    protected ICLAScheme updateVT(MatrixBlock data, IColIndex columns) {
        final int nRow = data.getNumColumns();
        final ReaderColumnSelection reader = ReaderColumnSelection.createReader(//
                data, columns, true, 0, nRow);
        return update(data, reader, nRow, columns);
    }

    @Override
    protected Pair<ICLAScheme, AColGroup> tryUpdateAndEncodeT(MatrixBlock data, IColIndex columns) {
        final int nRow = data.getNumColumns();
        final ReaderColumnSelection reader = ReaderColumnSelection.createReader(//
                data, columns, true, 0, nRow);
        return tryUpdateAndEncode(data, reader, nRow, columns);
    }

    private Pair<ICLAScheme, AColGroup> tryUpdateAndEncode(MatrixBlock data, ReaderColumnSelection reader, int nRow,
                                                           IColIndex columns) {
        final AMapToData d = MapToFactory.create(nRow, map.size());
        int max = d.getUpperBoundValue();

        DblArray cellVals;
        ACount<DblArray> emptyIdx = map.getC(emptyRow);
        if(emptyIdx == null) {
            while((cellVals = reader.nextRow()) != null) {
                final int row = reader.getCurrentRowIndex();
                final int id = map.increment(cellVals);
                if(id > max)
                    throw new DMLCompressionException("Failed update and encode with " + max + " possible values");
                d.set(row, id);
            }
        }
        else {
            int r = 0;
            while((cellVals = reader.nextRow()) != null) {
                final int row = reader.getCurrentRowIndex();
                if(row != r) {
                    map.increment(emptyRow, row - r);
                    while(r < row)
                        d.set(r++, emptyIdx.id);
                }
                final int id = map.increment(cellVals);
                if(id > max)
                    throw new DMLCompressionException(
                            "Failed update and encode with " + max + " possible values" + map + " " + map.size());
                d.set(row, id);
                r++;
            }
            if(r < nRow)

                map.increment(emptyRow, nRow - r);
            while(r < nRow)
                d.set(r++, emptyIdx.id);
        }
        if(lastDict == null || lastDict.getNumberOfValues(columns.size()) != map.size())
            lastDict = DictionaryFactory.create(map, columns.size(), false, data.getSparsity());

        AColGroup g = ColGroupDDCLZW.create(columns, lastDict, d, null);
        ICLAScheme s = this;
        return new Pair<>(s, g);
    }
    @Override
    public ACLAScheme clone() {
        return new DDCLZWSchemeMC(cols, map.clone());
    }

    @Override
    protected final Object getMap() {
        return map;
    }
}
