/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.data;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.util.SortUtils;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;

import static java.util.stream.IntStream.range;

// TODO: Do we need to be able to handle completely empty matrices? This will cause errors right now
// TODO: Maybe add a custom constructor because it does not make sense to initialize with rlen
public class SparseBlockDCSR extends SparseBlock
{
    private static final long serialVersionUID = 1922673753466164244L; // TODO: Generate properly
    private static final Log LOG = LogFactory.getLog(SparseBlockDCSR.class.getName());

    private int[] _rowidx = null; // row index array (size: >=
    private int[] _rowptr = null; //
    private int[] _colidx = null; // column index array (size: >=nnz)

    //private int[] _ptr = null;       //row pointer array (size: rlen+1)
    //private int[] _indexes = null;   //column index array (size: >=nnz)
    private double[] _values = null; //value array (size: >=nnz)
    private int _size = 0;           //actual nnz
    private int _rlen = 0;           //number of rows
    private int _nnzr = 0;           // number of nonzero rows

    // TODO: Is it a good idea to store the last pointer to potentially save binary searches?
    // TODO: Especially when we iteratively want to get the next nnz column, now we always have to do a binary search
    // TODO: However, this will complicate the code a bit and add 8 bytes of overhead and I don't know if this will actually speed up things in most use cases (as binary search is already O(log n))
    //private int lastRowPtr = -1;    // Last row pointer to make certain operations more efficient
    //private int lastColPtr = -1;    // Last column pointer to make certain operations more efficient

    public SparseBlockDCSR(int rlen) {
        this(rlen, INIT_CAPACITY);
    }

    public SparseBlockDCSR(int rlen, int capacity) {
        //TODO: This allocates too much space (we care about number of non-empty rows)
        LOG.warn("Allocating a DCSR-block using row-length. This will lead to significant overhead!");
        LOG.warn("If you want to initialize a sparse block using rlen, choose SparseBlockCSR instead!");

        _rowidx = new int[rlen];
        _rowptr = new int[rlen + 1];
        _colidx = new int[capacity];
        _values = new double[capacity];
        _rlen = rlen;
        _size = 0;
        _nnzr = 0;
    }

    public SparseBlockDCSR(int rlen, int capacity, int size, int nnzr){
        //TODO: This allocates too much space (we care about number of non-empty rows)
        LOG.warn("Allocating a DCSR-block using row-length. This will lead to significant overhead!");
        LOG.warn("If you want to initialize a sparse block using rlen, choose SparseBlockCSR instead!");

        _rowidx = new int[rlen];
        _rowptr = new int[rlen + 1];
        _colidx = new int[capacity];
        _values = new double[capacity];
        _rlen = rlen;
        _size = size;
        _nnzr = nnzr;
    }

    public SparseBlockDCSR(int[] rowIdx, int[] rowPtr, int[] colIdx, double[] values, int rlen, int nnz, int nnzr){
        _rowidx = rowIdx;
        _rowptr = rowPtr;
        _colidx = colIdx;
        _values = values;
        _rlen = rlen;
        _size = nnz;
        _nnzr = nnzr;
    }

    /**
     * Copy constructor sparse block abstraction.
     *
     * @param sblock sparse block to copy
     */
    public SparseBlockDCSR(SparseBlock sblock)
    {
        long size = sblock.size();
        if( size > Integer.MAX_VALUE )
            throw new RuntimeException("SparseBlockDCSR supports nnz<=Integer.MAX_VALUE but got "+size);

        //special case SparseBlockDCSR
        if( sblock instanceof SparseBlockDCSR ) {
            SparseBlockDCSR ocsr = (SparseBlockDCSR)sblock;
            _rowidx = Arrays.copyOf(ocsr._rowidx, ocsr._nnzr);
            _rowptr = Arrays.copyOf(ocsr._rowptr, ocsr._nnzr+1);
            _colidx = Arrays.copyOf(ocsr._colidx, ocsr._size);
            _values = Arrays.copyOf(ocsr._values, ocsr._size);
            _rlen = ocsr._rlen;
            _nnzr = ocsr._nnzr;
            _size = ocsr._size;
        }
        else if( sblock instanceof SparseBlockCSR ) {
            // TODO: Is CSR always sorted (confused by strict check and sorting functionality)
            // More efficient conversion from CSR to DCSR
            int rlen = sblock.numRows();

            SparseBlockCSR ocsr = (SparseBlockCSR)sblock;
            _rowidx = range(0, rlen).filter(rowIdx -> !sblock.isEmpty(rowIdx)).toArray();
            _rowptr = new int[_rowidx.length + 1];
            _colidx = Arrays.copyOf(ocsr.indexes(), (int)ocsr.size());
            _values = Arrays.copyOf(ocsr.values(), (int)ocsr.size());
            _rlen = rlen;
            _nnzr = _rowidx.length;
            _size = (int)ocsr.size();

            int pos = 0;

            for (int rowptr = 0; rowptr < _rowidx.length; rowptr++) {
                pos += sblock.size(_rowidx[rowptr]);
                _rowptr[rowptr+1] = pos;
            }
        }
        //general case SparseBlock
        else {
            int rlen = sblock.numRows();

            _rowidx = range(0, rlen).filter(rowIdx -> !sblock.isEmpty(rowIdx)).toArray();
            _rowptr = new int[_rowidx.length + 1];
            _colidx = new int[(int)size];
            _values = new double[(int)size];
            _rlen = rlen;
            _nnzr = _rowidx.length;
            _size = (int)size;

            int pos = 0;

            for ( int rowIdx : _rowidx ) {
                int apos = sblock.pos(rowIdx);
                int alen = sblock.size(rowIdx);
                int[] aix = sblock.indexes(rowIdx);
                double[] avals = sblock.values(rowIdx);
                System.arraycopy(aix, apos, _colidx, pos, alen);
                System.arraycopy(avals, apos, _values, pos, alen);
                pos += alen;
                _rowptr[rowIdx+1] = pos;
            }
        }
    }

    /**
     * Copy constructor old sparse row representation.
     * @param rows array of sparse rows
     * @param nnz number of non-zeroes
     */
    /*public SparseBlockDCSR(SparseRow[] rows, int nnz)
    {
        int rlen = rows.length;

        _ptr = new int[rlen+1]; //ix0=0
        _indexes = new int[nnz];
        _values = new double[nnz];
        _size = nnz;

        for( int i=0, pos=0; i<rlen; i++ ) {
            if( rows[i]!=null && !rows[i].isEmpty() ) {
                int alen = rows[i].size();
                int[] aix = rows[i].indexes();
                double[] avals = rows[i].values();
                System.arraycopy(aix, 0, _indexes, pos, alen);
                System.arraycopy(avals, 0, _values, pos, alen);
                pos += alen;
            }
            _ptr[i+1]=pos;
        }
    }*/

    /**
     * Copy constructor for COO representation
     *
     * @param rows number of rows
     * @param rowInd	row indices
     * @param colInd	column indices
     * @param values	non zero values
     */
    /*public SparseBlockDCSR(int rows, int[] rowInd, int[] colInd, double[] values) {
        int nnz = values.length;
        _ptr = new int[rows+1];
        _indexes = Arrays.copyOf(colInd, colInd.length);
        _values = Arrays.copyOf(values, values.length);
        _size = nnz;

        //single-pass construction of row pointers
        int rlast = 0;
        for(int i=0; i<nnz; i++) {
            int r = rowInd[i];
            if( rlast < r )
                Arrays.fill(_ptr, rlast+1, r+1, i);
            rlast = r;
        }
        Arrays.fill(_ptr, rlast+1, numRows()+1, nnz);
    }*/

    /**
     * Copy constructor for given array of column indexes, which
     * identifies rows by position and implies values of 1.
     *
     * @param rows number of rows
     * @param nnz number of non-zeros
     * @param colInd column indexes
     */
    /*public SparseBlockDCSR(int rows, int nnz, int[] colInd) {
        _ptr = new int[rows+1];
        _indexes = (rows==nnz) ? colInd : new int[nnz];
        _values = new double[nnz];
        Arrays.fill(_values, 1);
        _size = nnz;

        //single-pass construction of row pointers
        //and copy of column indexes if necessary
        for(int i=0, pos=0; i<rows; i++) {
            if( colInd[i] >= 0 ) {
                if( rows > nnz )
                    _indexes[pos] = colInd[i];
                pos++;
            }
            _ptr[i+1] = pos;
        }
    }*/

    /**
     * Initializes the CSR sparse block from an ordered input
     * stream of ultra-sparse ijv triples.
     *
     * @param nnz number of non-zeros to read
     * @param in data input stream of ijv triples, ordered by ij
     * @throws IOException if deserialization error occurs
     */
    /*public void initUltraSparse(int nnz, DataInput in)
            throws IOException
    {
        //allocate space if necessary
        if( _values.length < nnz )
            resize(newCapacity(nnz));

        //read ijv triples, append and update pointers
        int rlast = 0;
        for(int i=0; i<nnz; i++) {
            int r = in.readInt();
            if( rlast < r )
                Arrays.fill(_ptr, rlast+1, r+1, i);
            rlast = r;
            _indexes[i] = in.readInt();
            _values[i] = in.readDouble();
        }
        Arrays.fill(_ptr, rlast+1, numRows()+1, nnz);

        //update meta data
        _size = nnz;
    }*/

    /**
     * Initializes the CSR sparse block from an ordered input
     * stream of sparse rows (rownnz, jv-pairs*).
     *
     * @param rlen number of rows
     * @param nnz number of non-zeros to read
     * @param in data input stream of sparse rows, ordered by i
     * @throws IOException if deserialization error occurs
     */
    /*public void initSparse(int rlen, int nnz, DataInput in)
            throws IOException
    {
        //allocate space if necessary
        if( _values.length < nnz )
            resize(newCapacity(nnz));

        //read sparse rows, append and update pointers
        _ptr[0] = 0;
        for( int r=0, pos=0; r<rlen; r++ ) {
            int lnnz = in.readInt();
            for( int j=0; j<lnnz; j++, pos++ ) {
                _indexes[pos] = in.readInt();
                _values[pos] = in.readDouble();
            }
            _ptr[r+1] = pos;
        }

        //update meta data
        _size = nnz;
    }*/

    /**
     * Get the estimated in-memory size of the sparse block in CSR
     * with the given dimensions w/o accounting for overallocation.
     *
     * @param nrows number of rows
     * @param ncols number of columns
     * @param sparsity sparsity ratio
     * @return memory estimate
     */
    public static long estimateSizeInMemory(long nrows, long ncols, double sparsity) {
        double lnnz = Math.max(INIT_CAPACITY, Math.ceil(sparsity*nrows*ncols));

        //32B overhead per array, int arr in nrows, int/double arr in nnz
        double size = 16;                                    // Memory overhead of the object
        size += 4 + 4 + 4 + 4;                                  // 3x int field + 0 (padding not necessary)
        size += MemoryEstimates.intArrayCost(nrows);         // rowidx array (row indices)
        size += MemoryEstimates.intArrayCost(nrows+1); // rowptr array (row pointers)
        size += MemoryEstimates.intArrayCost((long) lnnz);   // colidx array (column indexes)
        size += MemoryEstimates.doubleArrayCost((long) lnnz);// values array (non-zero values)

        //robustness for long overflows
        return (long) Math.min(size, Long.MAX_VALUE);
    }

    ///////////////////
    //SparseBlock implementation

    @Override
    public void allocate(int r) {
        //do nothing everything preallocated
    }

    @Override
    public void allocate(int r, int nnz) {
        //do nothing everything preallocated
    }

    @Override
    public void allocate(int r, int ennz, int maxnnz) {
        //do nothing everything preallocated
    }

    @Override
    public void compact(int r) {
        //do nothing everything preallocated
    }

    @Override
    public int numRows() {
        return _rlen;
    }

    @Override
    public boolean isThreadSafe() {
        return false;
    }

    @Override
    public boolean isContiguous() {
        return true;
    }

    @Override
    public boolean isAllocated(int r) {
        return true;
    }

    @Override
    public void reset() {
        // TODO: Not tested
        if( _size > 0 ) {
            _size = 0;
            _nnzr = 0;
            _rlen = 0;
        }
    }

    @Override
    public void reset(int ennz, int maxnnz) {
        // TODO: Not tested
        if( _size > 0 ) {
            _size = 0;
            _nnzr = 0;
            _rlen = 0;
        }
    }

    @Override
    public void reset(int r, int ennz, int maxnnz) {
        // TODO: Not tested
        deleteIndexRange(r, 0, Integer.MAX_VALUE);
    }

    @Override
    public long size() {
        return _size;
    }

    @Override
    public int size(int r) {
        int idx = Arrays.binarySearch(_rowidx, 0, _nnzr, r);

        if (idx < 0)
            return 0;

        return _rowptr[idx+1] - _rowptr[idx];
    }

    @Override
    public long size(int rl, int ru) {
        int lowerIdx = Arrays.binarySearch(_rowidx, 0, _nnzr, rl);

        if (lowerIdx < 0)
            lowerIdx = -lowerIdx - 1;

        int upperIdx = Arrays.binarySearch(_rowidx, lowerIdx, _nnzr, ru);

        if (upperIdx < 0)
            upperIdx = -upperIdx - 1;

        return _rowptr[upperIdx] - _rowptr[lowerIdx];
    }

    @Override
    public long size(int rl, int ru, int cl, int cu) {
        long nnz = 0;

        int lRowIdx = Arrays.binarySearch(_rowidx, 0, _nnzr, rl);
        if (lRowIdx < 0)
            lRowIdx = -lRowIdx - 1;

        int uRowIdx = Arrays.binarySearch(_rowidx, lRowIdx, _nnzr, ru);
        if (uRowIdx < 0)
            uRowIdx = -uRowIdx - 1;

        for (int rowIdx = lRowIdx; rowIdx < uRowIdx; rowIdx++) {
            int clIdx = Arrays.binarySearch(_colidx, _rowptr[rowIdx], _rowptr[rowIdx+1], cl);
            if (clIdx < 0)
                clIdx = -clIdx - 1;

            int cuIdx = Arrays.binarySearch(_colidx, clIdx, _rowptr[rowIdx+1], cu);
            if (cuIdx < 0)
                cuIdx = -cuIdx - 1;

            nnz += cuIdx - clIdx;
        }
        return nnz;
    }

    @Override
    public boolean isEmpty(int r) {
        return size(r) == 0;
    }

    @Override
    public int[] indexes(int r) {
        return _colidx;
    }

    @Override
    public double[] values(int r) {
        return _values;
    }

    @Override
    public int pos(int r) {
        int idx = Arrays.binarySearch(_rowidx, 0, _nnzr, r);

        if (idx < 0)
            idx = Math.max(-idx - 2, 0);

        return _rowptr[idx];
    }

    @Override
    public boolean set(int r, int c, double v) {
        int rowIndex = Arrays.binarySearch(_rowidx, 0, _nnzr, r);
        boolean rowExists = rowIndex >= 0;

        if (!rowExists) {
            // Nothing to do
            if (v == 0)
                return false;

            int rowInsertionIndex = -rowIndex - 1;
            insertRow(rowInsertionIndex, r, _rowptr[rowInsertionIndex]);
            incrRowPtr(rowInsertionIndex+1);
            insertCol(_rowptr[rowInsertionIndex], c, v);
            return true;
        }

        int pos = _rowptr[rowIndex];
        int len = _rowptr[rowIndex+1] - pos;
        int index = Arrays.binarySearch(_colidx, pos, pos+len, c);
        boolean colExists = index >= 0;

        if (v != 0) {
            if (colExists) {
                _values[index] = v;
                return false;
            }

            // Insert a new column into an existing row
            insertCol(-index-1, c, v);
            incrRowPtr(rowIndex+1);

            return true;
        }

        if (!colExists)
            return false;

        // If there is only one entry in the row, we have to remove the entire row
        if (len == 1) {
            deleteRow(rowIndex);
            rowIndex--;
        }

        // remove the column
        incrRowPtr(rowIndex+1, -1);
        deleteCol(index);

        return true;
    }

    @Override
    public boolean add(int r, int c, double v) {
        // TODO: Not tested
        // TODO: Make more efficient to save binary searches
        double oldValue = get(r, c);

        if (v == 0)
            return false;

        return set(r, c, oldValue + v);
    }

    @Override
    public void set(int r, SparseRow row, boolean deep) {
        int newRowSize = row.size();

        int rowIndex = Arrays.binarySearch(_rowidx, 0, _nnzr, r);
        boolean rowExists = rowIndex >= 0;

        if (!rowExists) {
            // Nothing to do
            if (newRowSize == 0)
                return;

            int rowInsertionIndex = -rowIndex - 1;
            insertRow(rowInsertionIndex, r, _rowptr[rowInsertionIndex]);
            incrRowPtr(rowInsertionIndex+1, newRowSize);
            insertCols(_rowptr[rowInsertionIndex], row.indexes(), row.values());
            return;
        }

        int pos = _rowptr[rowIndex];
        int oldRowSize = _rowptr[rowIndex+1] - pos;

        if (newRowSize == 0) {
            // Delete row
            deleteRow(rowIndex);
            incrRowPtr(rowIndex, -oldRowSize);
            deleteCols(pos, oldRowSize);
            return;
        }

        incrRowPtr(rowIndex+1, newRowSize-oldRowSize);
        insertCols(pos, row.indexes(), row.values(), oldRowSize);
    }

    @Override
    public void append(int r, int c, double v) {
        // TODO: Maybe more efficient implementation? We could save two binary searches (row and column)
        set(r, c, v);
    }

    @Override
    public void setIndexRange(int r, int cl, int cu, double[] v, int vix, int vlen) {
        // TODO: Isn't lnnz always vlen? Or isn't this guaranteed (according to docs)? Otherwise remove this computation
        int lnnz = UtilFunctions.computeNnz(v, vix, vlen);

        if (lnnz == 0) {
            deleteIndexRange(r, cl, cu);
            return;
        }

        int rowIdx = Arrays.binarySearch(_rowidx, 0, _nnzr, r);

        if (rowIdx < 0) {
            rowIdx = -rowIdx - 1;
            insertRow(rowIdx, r, _rowptr[rowIdx]);
        }

        int rowStart = _rowptr[rowIdx];
        int rowEnd = _rowptr[rowIdx+1];

        int clIdx = Arrays.binarySearch(_colidx, rowStart, rowEnd, cl);
        if (clIdx < 0)
            clIdx = -clIdx - 1;

        int cuIdx = Arrays.binarySearch(_colidx, clIdx, rowEnd, cu);
        if (cuIdx < 0)
            cuIdx = -cuIdx - 1;

        int oldnnz = cuIdx - clIdx;

        allocateCols(clIdx, lnnz, oldnnz);
        incrRowPtr(rowIdx+1, lnnz - oldnnz);

        int insertionIndex = clIdx;

        for (int i = vix; i < vix+vlen; i++) {
            if (v[i] != 0) {
                _colidx[insertionIndex] = cl + i - vix;
                _values[insertionIndex] = v[i];
                insertionIndex++;
            }
        }
    }

    @Override
    public void setIndexRange(int r, int cl, int cu, double[] v, int[] vix, int vpos, int vlen) {
        // TODO: Not tested, mostly copied from previous function

        if (vlen == 0) {
            deleteIndexRange(r, cl, cu);
            return;
        }

        int rowIdx = Arrays.binarySearch(_rowidx, 0, _nnzr, r);

        if (rowIdx < 0) {
            rowIdx = -rowIdx - 1;
            insertRow(rowIdx, r, _rowptr[rowIdx]);
        }

        int rowStart = _rowptr[rowIdx];
        int rowEnd = _rowptr[rowIdx+1];

        int clIdx = Arrays.binarySearch(_colidx, rowStart, rowEnd, cl);
        if (clIdx < 0)
            clIdx = -clIdx - 1;

        int cuIdx = Arrays.binarySearch(_colidx, clIdx, rowEnd, cu);
        if (cuIdx < 0)
            cuIdx = -cuIdx - 1;

        int oldnnz = cuIdx - clIdx;

        allocateCols(clIdx, vlen, oldnnz);
        incrRowPtr(rowIdx+1, vlen - oldnnz);

        int insertionIndex = clIdx;

        for (int i = vpos; i < vpos+vlen; i++) {
            if (v[i] != 0) {
                _colidx[insertionIndex] = cl - vix[i];
                _values[insertionIndex] = v[i];
                insertionIndex++;
            }
        }
    }

    // TODO: Is it OK if I leave this CSR-specific method out? Haven't looked at it
    /**
     * Inserts a sorted row-major array of non-zero values into the row and column
     * range [rl,ru) and [cl,cu). Note: that this is a CSR-specific method to address
     * performance issues due to repeated re-shifting on update-in-place.
     *
     * @param rl  lower row index, starting at 0, inclusive
     * @param ru  upper row index, starting at 0, exclusive
     * @param cl  lower column index, starting at 0, inclusive
     * @param cu  upper column index, starting at 0, exclusive
     * @param v   right-hand-side dense block
     * @param vix right-hand-side dense block index
     * @param vlen right-hand-side dense block value length
     */
    /*
    public void setIndexRange(int rl, int ru, int cl, int cu, double[] v, int vix, int vlen) {

        //step 1: determine output nnz
        int nnz = _size - (int)size(rl, ru, cl, cu);
        if( v != null )
            nnz += UtilFunctions.computeNnz(v, vix, vlen);

        //step 2: reallocate if necessary
        if( _values.length < nnz )
            resize(nnz);

        //step 3: insert and overwrite index range
        //total shift can be negative or positive and w/ internal skew

        //step 3a: forward pass: compact (delete index range)
        int pos = pos(rl);
        for( int r=rl; r<ru; r++ ) {
            int rpos = pos(r);
            int rlen = size(r);
            _ptr[r] = pos;
            for( int k=rpos; k<rpos+rlen; k++ )
                if( _indexes[k]<cl || cu<=_indexes[k] ) {
                    _indexes[pos] = _indexes[k];
                    _values[pos++] = _values[k];
                }
        }
        shiftLeftByN(pos(ru), pos(ru)-pos);
        decrPtr(ru, pos(ru)-pos);

        //step 3b: backward pass: merge (insert index range)
        int tshift1 = nnz - _size; //always non-negative
        if( v == null || tshift1==0 ) //early abort
            return;
        shiftRightByN(pos(ru), tshift1);
        incrPtr(ru, tshift1);
        pos = pos(ru)-1;
        int clen2 = cu-cl;
        for( int r=ru-1; r>=rl; r-- ) {
            int rpos = pos(r);
            int rlen = size(r) - tshift1;
            //copy lhs right
            int k = -1;
            for( k=rpos+rlen-1; k>=rpos && _indexes[k]>=cu; k-- ) {
                _indexes[pos] = _indexes[k];
                _values[pos--] = _values[k];
            }
            //copy rhs
            int voff = vix + (r-rl) * clen2;
            for( int k2=clen2-1; k2>=0 & vlen>voff; k2-- )
                if( v[voff+k2] != 0 ) {
                    _indexes[pos] = cl + k2;
                    _values[pos--] = v[voff+k2];
                    tshift1--;
                }
            //copy lhs left
            for( ; k>=rpos; k-- ) {
                _indexes[pos] = _indexes[k];
                _values[pos--] = _values[k];
            }
            _ptr[r] = pos+1;
        }
    }*/

    // TODO: Is it OK if I leave this CSR-specific method out? Haven't looked at it
    /**
     * Inserts a sparse block into the row and column range [rl,ru) and [cl,cu).
     * Note: that this is a CSR-specific method to address  performance issues
     * due to repeated re-shifting on update-in-place.
     *
     * @param rl  lower row index, starting at 0, inclusive
     * @param ru  upper row index, starting at 0, exclusive
     * @param cl  lower column index, starting at 0, inclusive
     * @param cu  upper column index, starting at 0, exclusive
     * @param sb  right-hand-side sparse block
     */
    /*public void setIndexRange(int rl, int ru, int cl, int cu, SparseBlock sb) {
        // TODO: Implement
        throw new NotImplementedException();

        //step 1: determine output nnz
        int nnz = (int) (_size - size(rl, ru, cl, cu)
                + ((sb!=null) ? sb.size() : 0));

        //step 2: reallocate if necessary
        if( _values.length < nnz )
            resize(nnz);

        //step 3: insert and overwrite index range (backwards)
        //total shift can be negative or positive and w/ internal skew

        //step 3a: forward pass: compact (delete index range)
        int pos = pos(rl);
        for( int r=rl; r<ru; r++ ) {
            int rpos = pos(r);
            int rlen = size(r);
            _ptr[r] = pos;
            for( int k=rpos; k<rpos+rlen; k++ )
                if( _indexes[k]<cl || cu<=_indexes[k] ) {
                    _indexes[pos] = _indexes[k];
                    _values[pos++] = _values[k];
                }
        }
        shiftLeftByN(pos(ru), pos(ru)-pos);
        decrPtr(ru, pos(ru)-pos);

        //step 3b: backward pass: merge (insert index range)
        int tshift1 = nnz - _size; //always non-negative
        if( sb == null || tshift1==0 ) //early abort
            return;
        shiftRightByN(pos(ru), tshift1);
        incrPtr(ru, tshift1);
        pos = pos(ru)-1;
        for( int r=ru-1; r>=rl; r-- ) {
            int rpos = pos(r);
            int rlen = size(r) - tshift1;
            //copy lhs right
            int k = -1;
            for( k=rpos+rlen-1; k>=rpos && _indexes[k]>=cu; k-- ) {
                _indexes[pos] = _indexes[k];
                _values[pos--] = _values[k];
            }
            //copy rhs
            int r2 = r-rl;
            int r2pos = sb.pos(r2);
            for( int k2=r2pos+sb.size(r2)-1; k2>=r2pos; k2-- ) {
                _indexes[pos] = cl + sb.indexes(r2)[k2];
                _values[pos--] = sb.values(r2)[k2];
                tshift1--;
            }
            //copy lhs left
            for( ; k>=rpos; k-- ) {
                _indexes[pos] = _indexes[k];
                _values[pos--] = _values[k];
            }
            _ptr[r] = pos+1;
        }
    }*/

    @Override
    public void deleteIndexRange(int r, int cl, int cu) {
        int rowIdx = Arrays.binarySearch(_rowidx, 0, _nnzr, r);
        if( rowIdx < 0 ) //nothing to delete
            return;

        int nnz = _rowptr[rowIdx+1] - _rowptr[rowIdx];

        int start = Arrays.binarySearch(_colidx, _rowptr[rowIdx], _rowptr[rowIdx+1], cl);
        if (start < 0)
            start = -start-1;

        int end = Arrays.binarySearch(_colidx, start, _rowptr[rowIdx+1], cu);
        if( end < 0 ) //delete all remaining
            end = -end-1;

        if (end-start <= 0) // Nothing to delete
            return;

        if (nnz == end-start) {
            deleteRow(rowIdx);
            rowIdx--;
        }

        //overlapping array copy (shift rhs values left)
        System.arraycopy(_colidx, end, _colidx, start, _size-end);
        System.arraycopy(_values, end, _values, start, _size-end);
        _size -= (end-start);

        incrRowPtr(rowIdx+1, start-end);
    }

    @Override
    public void sort() {
        for( int i=0; i < _rowidx.length; i++ )
            sortFromRowIndex(i);
    }

    @Override
    public void sort(int r) {
        int rowIdx = Arrays.binarySearch(_rowidx, 0, _nnzr, r);

        if (rowIdx >= 0)
            sortFromRowIndex(rowIdx);
    }

    private void sortFromRowIndex(int rowIndex) {
        int pos = _rowptr[rowIndex];
        int len = _rowptr[rowIndex+1] - pos;
        if( !SortUtils.isSorted(pos, pos+len, _colidx) )
            SortUtils.sortByIndex(pos, pos+len, _colidx, _values);
    }

    @Override
    public double get(int r, int c) {
        int rowIndex = Arrays.binarySearch(_rowidx, 0, _nnzr, r);

        if (rowIndex < 0)
            return 0;

        int pos = _rowptr[rowIndex];
        int len = _rowptr[rowIndex+1] - pos;

        //search for existing col index in [pos,pos+len)
        int index = Arrays.binarySearch(_colidx, pos, pos+len, c);
        return (index >= 0) ? _values[index] : 0;
    }

    @Override
    public SparseRow get(int r) {
        if( isEmpty(r) )
            return new SparseRowScalar();
        int pos = pos(r);
        int len = size(r);

        SparseRowVector row = new SparseRowVector(len);
        System.arraycopy(_colidx, pos, row.indexes(), 0, len);
        System.arraycopy(_values, pos, row.values(), 0, len);
        row.setSize(len);
        return row;
    }

    @Override
    public Iterator<IJV> getIterator() {
        // TODO: Can be implemented more efficiently by walking through _rowidx etc.
        return super.getIterator();
    }

    @Override
    public int posFIndexLTE(int r, int c) {
        int rowIdx = Arrays.binarySearch(_rowidx, 0, _nnzr, r);

        if (rowIdx < 0)
            return -1;

        int colIdx = Arrays.binarySearch(_colidx, _rowptr[rowIdx], _rowptr[rowIdx+1], c);

        if (colIdx < 0)
            colIdx = -colIdx - 2;

        // There is no element smaller or equal in this row
        if (colIdx < _rowptr[rowIdx])
            return -1;

        return colIdx;
    }

    @Override
    public final int posFIndexGTE(int r, int c) {
        int rowIdx = Arrays.binarySearch(_rowidx, 0, _nnzr, r);

        if (rowIdx < 0)
            return -1;

        int colIdx = Arrays.binarySearch(_colidx, _rowptr[rowIdx], _rowptr[rowIdx+1], c);

        if (colIdx < 0)
            colIdx = -colIdx - 1;

        // There is no element greater or equal in this row
        if (colIdx >= _rowptr[rowIdx+1])
            return -1;

        return colIdx - _rowptr[rowIdx];
    }

    @Override
    public int posFIndexGT(int r, int c) {
        int rowIdx = Arrays.binarySearch(_rowidx, 0, _nnzr, r);

        if (rowIdx < 0)
            return -1;

        int colIdx = Arrays.binarySearch(_colidx, _rowptr[rowIdx], _rowptr[rowIdx+1], c) + 1;

        if (colIdx >= 0)
            colIdx++;
        else
            colIdx = -colIdx - 1;

        // There is no element greater or equal in this row
        if (colIdx >= _rowptr[rowIdx+1])
            return -1;

        return colIdx;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("SparseBlockCSR: rlen=");
        sb.append(numRows());
        sb.append(", nnz=");
        sb.append(size());
        sb.append("\n");
        final int rowDigits = (int)Math.max(Math.ceil(Math.log10(numRows())),1) ;
        for(int rowIdx = 0; rowIdx < _rowidx.length; rowIdx++) {
            // append row
            final int row = _rowidx[rowIdx];
            final int pos = _rowptr[rowIdx];
            final int len = _rowptr[rowIdx+1] - pos;

            sb.append(String.format("%0"+rowDigits+"d ", row));
            for(int j = pos; j < pos + len; j++) {
                if(_values[j] == (long) _values[j])
                    sb.append(String.format("%"+rowDigits+"d:%d", _colidx[j], (long)_values[j]));
                else
                    sb.append(String.format("%"+rowDigits+"d:%s", _colidx[j], Double.toString(_values[j])));
                if(j + 1 < pos + len)
                    sb.append(" ");
            }
            sb.append("\n");
        }

        return sb.toString();
    }

    /*private void debugData(boolean debugArrays) {
        LOG.error("MDEBUG: ===== NEXT =====");
        if (debugArrays) {
            LOG.error("MDEBUG: ROWIDX: " + Arrays.toString(_rowidx));
            LOG.error("MDEBUG: ROWPTR: " + Arrays.toString(_rowptr));
            LOG.error("MDEBUG: COLIDX: " + Arrays.toString(_colidx));
            LOG.error("MDEBUG: VALS: " + Arrays.toString(_values));
        }
        LOG.error("MDEBUG: NUM_ROWS: " + _rlen);
        LOG.error("MDEBUG: NNZR: " + _nnzr);
        LOG.error("MDEBUG: NNZ: " + _size);
    }*/

    @Override
    public boolean checkValidity(int rlen, int clen, long nnz, boolean strict) {
        //1. correct meta data
        if ( rlen < 0 || clen < 0 ) {
            throw new RuntimeException("Invalid block dimensions: "+rlen+" "+clen);
        }

        //2. correct array lengths
        if (_size != nnz && _rowptr.length != _rowidx.length + 1 && _values.length < nnz && _colidx.length < nnz ) {
            throw new RuntimeException("Incorrect array lengths.");
        }

        //3. non-decreasing row pointers
        // TODO: Check if only if strict check applies here (I don't think so); I did not quite understand when CSR Arrays can be unsorted (binary searches wouldn't work in that case)
        for ( int i=1; i <_rowidx.length; i++ ) {
            if (_rowidx[i-1] > _rowidx[i])
                throw new RuntimeException("Row indices are decreasing at row: " + i
                        + ", with indices " + _rowidx[i-1] + " > " +_rowidx[i]);
        }

        for (int i = 1; i < _rowptr.length; i++ ) {
            if (_rowptr[i - 1] > _rowptr[i]) {
                throw new RuntimeException("Row pointers are decreasing at row: " + i
                        + ", with pointers " + _rowptr[i-1] + " > " +_rowptr[i]);
            }
        }

        //4. sorted column indexes per row
        for ( int rowIdx = 0; rowIdx < _rowidx.length; rowIdx++ ) {
            int apos = _rowidx[rowIdx];
            int alen = _rowidx[rowIdx+1] - apos;

            for( int k = apos + 1; k < apos + alen; k++)
                if( _colidx[k-1] >= _colidx[k] )
                    throw new RuntimeException("Wrong sparse row ordering: "
                            + k + " " + _colidx[k-1] + " " + _colidx[k]);

            for( int k=apos; k<apos+alen; k++ )
                if( _values[k] == 0 )
                    throw new RuntimeException("Wrong sparse row: zero at "
                            + k + " at col index " + _colidx[k]);
        }

        //5. non-existing zero values
        for( int i=0; i<_size; i++ ) {
            if( _values[i] == 0 ) {
                throw new RuntimeException("The values array should not contain zeros."
                        + " The " + i + "th value is "+_values[i]);
            }
        }

        //6. a capacity that is no larger than nnz times resize factor.
        int capacity = _values.length;
        if(capacity > nnz*RESIZE_FACTOR1 ) {
            throw new RuntimeException("Capacity is larger than the nnz times a resize factor."
                    + " Current size: "+capacity+ ", while Expected size:"+nnz*RESIZE_FACTOR1);
        }

        return true;
    }

    @Override //specialized for CSR
    public boolean contains(double pattern) {
        boolean NaNpattern = Double.isNaN(pattern);
        double[] vals = _values;
        int len = _size;
        for(int i=0; i<len; i++)
            if(vals[i]==pattern || (NaNpattern && Double.isNaN(vals[i])))
                return true;
        return false;
    }

    ///////////////////////////
    // private helper methods

    private int newCapacity(int minsize) {
        //compute new size until minsize reached
        double tmpCap = Math.max(_values.length, 1);
        while( tmpCap < minsize ) {
            tmpCap *= (tmpCap <= 1024) ?
                    RESIZE_FACTOR1 : RESIZE_FACTOR2;
        }
        return (int)Math.min(tmpCap, Integer.MAX_VALUE);
    }

    private void deleteRow(int rowIdx) {
        System.arraycopy(_rowidx, rowIdx + 1, _rowidx, rowIdx, _nnzr-rowIdx-1);
        System.arraycopy(_rowptr, rowIdx + 1, _rowptr, rowIdx, _nnzr-rowIdx);
        _nnzr--;
    }

    private void insertRow(int ix, int row, int rowPtr) {
        if (_nnzr >= _rowidx.length) {
            resizeAndInsertRow(ix, row, rowPtr);
            return;
        }

        System.arraycopy(_rowidx, ix, _rowidx, ix+1, _nnzr-ix);
        System.arraycopy(_rowptr, ix, _rowptr, ix+1, _nnzr-ix+1);
        _rowidx[ix] = row;
        _rowptr[ix] = rowPtr;
        _nnzr++;
    }

    private void resizeAndInsertRow(int ix, int row, int rowPtr) {
        //compute new size
        int newCap = newCapacity(_rowidx.length+1);

        int[] oldrowidx = _rowidx;
        int[] oldrowptr = _rowptr;
        _rowidx = new int[newCap];
        _rowptr = new int[newCap+1];

        //copy lhs values to new array
        System.arraycopy(oldrowidx, 0, _rowidx, 0, ix);
        System.arraycopy(oldrowptr, 0, _rowptr, 0, ix);

        //copy rhs values to new array
        System.arraycopy(oldrowidx, ix, _rowidx, ix+1, _nnzr-ix);
        System.arraycopy(oldrowptr, ix, _rowptr, ix+1, _nnzr-ix+1);

        _rowidx[ix] = row;
        _rowptr[ix] = rowPtr;

        _nnzr++;
    }

    private void deleteCol(int ix) {
        // Without removing row
        //overlapping array copy (shift rhs values left by 1)
        System.arraycopy(_colidx, ix+1, _colidx, ix, _size-ix-1);
        System.arraycopy(_values, ix+1, _values, ix, _size-ix-1);
        _size--;
    }

    private void insertCol(int ix, int c, double v) {
        // Without inserting row
        if (_size >= _colidx.length) {
            resizeAndInsertCol(ix, c, v);
            return;
        }

        System.arraycopy(_colidx, ix, _colidx, ix+1, _size-ix);
        System.arraycopy(_values, ix, _values, ix+1, _size-ix);

        _colidx[ix] = c;
        _values[ix] = v;
        _size++;
    }

    private void deleteCols(int ix, int len) {
        insertCols(ix, new int[0], new double[0], len, 0, 0);
    }

    private void insertCols(int ix, int[] cols, double[] vals) {
        insertCols(ix, cols, vals, 0, 0, vals.length);
    }

    private void insertCols(int ix, int[] cols, double[] vals, int overwriteNum) {
        insertCols(ix, cols, vals, overwriteNum, 0, vals.length);
    }

    private void insertCols(int ix, int[] cols, double[] vals, int overwriteNum, int vix, int vlen) {
        // Without inserting row
        if (_size + vlen - overwriteNum > _colidx.length) {
            resizeAndInsertCols(ix, cols, vals, overwriteNum, vix, vlen);
            return;
        }

        allocateCols(ix, vlen, overwriteNum);

        System.arraycopy(cols, vix, _colidx, ix, vlen);
        System.arraycopy(vals, vix, _values, ix, vlen);
    }

    private void resizeAndInsertCols(int ix, int[] cols, double[] vals, int overwriteNum, int vix, int vlen) {
        resizeAndAllocateCols(ix, vlen, overwriteNum);

        //copy new vals into row
        System.arraycopy(cols, vix, _colidx, ix, vlen);
        System.arraycopy(vals, vix, _values, ix, vlen);
    }

    private void allocateCols(int ix, int numCols) {
        allocateCols(ix, numCols, 0);
    }

    private void allocateCols(int ix, int numCols, int overwriteNum) {
        if (numCols == 0)
            return;

        if (_size + numCols - overwriteNum > _colidx.length) {
            resizeAndAllocateCols(ix, numCols, overwriteNum);
            return;
        }

        System.arraycopy(_colidx, ix+overwriteNum, _colidx, ix+numCols, _size-ix-overwriteNum);
        System.arraycopy(_values, ix+overwriteNum, _values, ix+numCols, _size-ix-overwriteNum);
        _size += numCols - overwriteNum;
    }

    private void resizeAndAllocateCols(int ix, int numCols, int overwriteNum) {
        //compute new size
        int newCap = newCapacity(_size + numCols - overwriteNum);

        int[] oldcolidx = _colidx;
        double[] oldvalues = _values;
        _colidx = new int[newCap];
        _values = new double[newCap];

        //copy lhs values to new array
        System.arraycopy(oldcolidx, 0, _colidx, 0, ix);
        System.arraycopy(oldvalues, 0, _values, 0, ix);

        //copy rhs values to new array
        System.arraycopy(oldcolidx, ix + overwriteNum, _colidx, ix+numCols, _size-ix-overwriteNum);
        System.arraycopy(oldvalues, ix + overwriteNum, _values, ix+numCols, _size-ix-overwriteNum);

        _size += numCols - overwriteNum;
    }

    private void resizeAndInsertCol(int ix, int c, double v) {
        //compute new size
        int newCap = newCapacity(_values.length+1);

        int[] oldcolidx = _colidx;
        double[] oldvalues = _values;
        _colidx = new int[newCap];
        _values = new double[newCap];

        //copy lhs values to new array
        System.arraycopy(oldcolidx, 0, _colidx, 0, ix);
        System.arraycopy(oldvalues, 0, _values, 0, ix);

        //copy rhs values to new array
        System.arraycopy(oldcolidx, ix, _colidx, ix+1, _size-ix);
        System.arraycopy(oldvalues, ix, _values, ix+1, _size-ix);

        //insert new value
        _colidx[ix] = c;
        _values[ix] = v;

        _size++;
    }

    private void incrRowPtr(int rowIndex) {
        incrRowPtr(rowIndex, 1);
    }

    private void incrRowPtr(int rowIndex, int cnt) {

        for( int i = rowIndex; i < _nnzr + 1; i++ )
            _rowptr[i] += cnt;
    }
}
