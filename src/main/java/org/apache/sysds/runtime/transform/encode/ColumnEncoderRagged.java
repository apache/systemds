package org.apache.sysds.runtime.transform.encode;

import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.common.Types.ValueType;

import java.util.HashMap;
import java.util.Map;

/**
 * Encodes a column using ragged array/dictionary representation to optimize memory usage.
 * Stores unique values in a dictionary and replaces occurrences with indices.
 */
public class ColumnEncoderRagged extends ColumnEncoder {
    private static final long serialVersionUID = 2291732648968734088L;
    
    // Dictionary storage
    private Object[] _dict;
    private int _dictSize;
    private int _nullIndex = -1;
    
    // Reverse mapping for fast lookups
    private transient Map<Object, Integer> _valueToIndex;

    private static final String[] DEFAULT_NA_STRINGS = new String[]{"NA", "NaN", ""};
    
    public ColumnEncoderRagged() {
        super(-1); // ID will be set during construction
    }   
    
    public ColumnEncoderRagged(int colID) {
        super(colID);
    }

    @Override
    protected TransformType getTransformType() {
        return TransformType.RAGGED;
    }

    // Helper method to check NA values
    private boolean isNAValue(String val) {
        if(val == null) return true;
        for(String na : DEFAULT_NA_STRINGS) {
            if(val.equals(na)) return true;
        }
        return false;
    }

    @Override
    public void build(CacheBlock<?> in) {
        if (!(in instanceof FrameBlock))
            throw new IllegalArgumentException("Ragged encoding only supports FrameBlock input");
            
        FrameBlock fin = (FrameBlock) in;
        if (_colID < 1 || _colID > fin.getNumColumns())
            throw new IllegalArgumentException("Invalid column ID: " + _colID);
        
        _valueToIndex = new HashMap<>();
        _dict = new String[Math.min(1024, fin.getNumRows())];
        _dictSize = 0;
        
        for (int i = 0; i < fin.getNumRows(); i++) {
            Object valObj = fin.get(i, _colID - 1);
            // Convert all values to strings safely
            String val = (valObj != null) ? valObj.toString() : null;
            
            if (isNAValue(val)) {
                if (_nullIndex == -1) {
                    _nullIndex = _dictSize;
                    _dict[_dictSize++] = null;
                }
                continue;
            }
            
            if (!_valueToIndex.containsKey(val)) {
                if (_dictSize == _dict.length) {
                    String[] newDict = new String[_dict.length * 2];
                    System.arraycopy(_dict, 0, newDict, 0, _dictSize);
                    _dict = newDict;
                }
                _dict[_dictSize] = val;
                _valueToIndex.put(val, _dictSize);
                _dictSize++;
            }
        }
    }

    @Override
public MatrixBlock apply(CacheBlock<?> in, MatrixBlock out, int outputCol) {
    // Validate input type
    if (!(in instanceof FrameBlock)) {
        throw new IllegalArgumentException("Ragged encoding only supports FrameBlock input");
    }
    
    FrameBlock fin = (FrameBlock) in;
    final int numRows = fin.getNumRows();
    
    // Create new matrix if needed
    if (out == null) {
        out = new MatrixBlock(numRows, outputCol + 1, false);
    }
    
    // Encode each value
    for (int i = 0; i < numRows; i++) {
        String val = fin.get(i, _colID - 1).toString();
        int index = isNAValue(val) ? _nullIndex : _valueToIndex.getOrDefault(val, _nullIndex);
        
        // Use the standard set method
        out.set(i, outputCol, (double) index);
    }
    
    return out;
}

    @Override
    public double[] getCodeCol(CacheBlock<?> in, int outputCol, int rowStart, double[] tmp) {
        if (!(in instanceof FrameBlock))
            throw new IllegalArgumentException("Ragged encoding only supports FrameBlock input");
        FrameBlock fin = (FrameBlock) in;

        if (tmp == null)
            tmp = new double[fin.getNumRows() - rowStart];
        
        for (int i = rowStart; i < fin.getNumRows(); i++) {
            String val = fin.get(i, _colID - 1).toString();
            tmp[i - rowStart] = isNAValue(val) ? _nullIndex : _valueToIndex.getOrDefault(val, _nullIndex);
        }
        return tmp;
    }

    @Override
    public double getCode(CacheBlock<?> in, int row) {
        if (!(in instanceof FrameBlock))
            throw new IllegalArgumentException("Ragged encoding only supports FrameBlock input");
        FrameBlock fin = (FrameBlock) in;

        String val = fin.get(row, _colID - 1).toString();
        return isNAValue(val) ? _nullIndex : _valueToIndex.getOrDefault(val, _nullIndex);
    }

    @Override
    public FrameBlock getMetaData(FrameBlock out) {
        if (out == null)
            out = new FrameBlock(1, ValueType.STRING);
        
        // Store dictionary in meta frame
        out.ensureAllocatedColumns(_dictSize);
        for (int i = 0; i < _dictSize; i++) {
            out.set(i, 0, _dict[i]);
        }
        
        return out;
    }

    @Override
    public void initMetaData(FrameBlock meta) {
        if (meta == null || meta.getNumRows() == 0)
            return;
        
        // Reconstruct dictionary from meta data
        _dictSize = meta.getNumRows();
        _dict = new Object[_dictSize];
        _valueToIndex = new HashMap<>();
        
        for (int i = 0; i < _dictSize; i++) {
            _dict[i] = meta.get(i, 0);
            if (_dict[i] == null) {
                _nullIndex = i;
            } else {
                _valueToIndex.put(_dict[i], i);
            }
        }
    }

    // Other required methods with default implementations
    @Override public void allocateMetaData(FrameBlock meta) {}
    @Override public void prepareBuildPartial() {}
    @Override public void buildPartial(FrameBlock in) { build(in); }
    @Override public void updateIndexRanges(long[] beginDims, long[] endDims, int offset) {}

    // Additional helper methods
    public Object[] getDictionary() {
        return _dict;
    }
    
    public int getDictionarySize() {
        return _dictSize;
    }
    
    public int getNullIndex() {
        return _nullIndex;
    }
}