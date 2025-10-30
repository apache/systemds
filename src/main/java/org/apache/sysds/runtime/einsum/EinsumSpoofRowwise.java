package org.apache.sysds.runtime.einsum;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;
import org.apache.sysds.runtime.codegen.SpoofRowwise;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;

public final class EinsumSpoofRowwise extends SpoofRowwise {
    private final int _ABCount;
    private final int _BCount;
    private final int _ACount;
    private final int _ZCount;
    private final int _AZCount;
    private final int _ZSize;

    private final int _uptoBCumCount;
    private final int _uptoZCumCount;

    private final EOpNodeFuse.EinsumRewriteType _EinsumRewriteType;

    public EinsumSpoofRowwise(EOpNodeFuse.EinsumRewriteType einsumRewriteType, RowType rowType, long constDim2, boolean tb1, int reqVectMem, int abCount, int bCount, int aCount, int zCount, int azCount, int zSize) {
        super(rowType, constDim2, tb1, reqVectMem);
        _ABCount = abCount;
        _BCount = bCount;
        _uptoBCumCount = bCount+ abCount;
        _ACount = aCount;
        _ZCount = zCount;
        _uptoZCumCount = _uptoBCumCount + aCount;
        _AZCount = azCount;
        _EinsumRewriteType = einsumRewriteType;
        _ZSize = zSize;
    }
    protected void genexec(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len, long grix, int rix) {
        switch (_EinsumRewriteType) {
            case AB_BA_B_A__AB -> genexec_AB(a,ai,b,scalars,c,ci,len,grix,rix);
            case AB_BA_B_A__B -> genexec_B(a,ai,b,scalars,c,ci,len,grix,rix);
            case AB_BA_B_A__A -> genexec_A_or_(a,ai,b,scalars,c,ci,len,grix,rix);
            case AB_BA_B_A__ -> genexec_A_or_(a,ai,b,scalars,c,ci,len,grix,rix);
            case AB_BA_B_A_AZ__Z -> {
                double[] temp = {0};
                genexec_A_or_(a,ai,b,scalars,temp,0,len,grix,rix);
                LibMatrixMult.vectMultiplyAdd(temp[0], b[_uptoZCumCount].values(rix), c, _ZSize*rix,0, _ZSize);
            }
            case AB_BA_B_A_AZ__BZ -> {
                double[] temp = new double[len];
                genexec_B(a,ai,b,scalars,temp,0,len,grix,rix);
                LibSpoofPrimitives.vectOuterMultAdd(temp, b[_uptoZCumCount].values(rix), c,0, _ZSize*rix, 0,  len,_ZSize);
            }
            case AB_BA_B_A_AZ__ZB -> {
                double[] temp = new double[len];
                genexec_B(a,ai,b,scalars,temp,0,len,grix,rix);

                LibSpoofPrimitives.vectOuterMultAdd(b[_uptoZCumCount].values(rix),temp , c,_ZSize*rix,0, 0, _ZSize, len);
            }
//            case AB_BA_B_XB_BX_A_XA_AX_AZ_AC__CZ ->
            default -> throw new NotImplementedException();
        }
    }
    protected void genexec_AB(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len, long grix, int rix) {
        int bi = 0;
        double[] TMP1 = null;
        if (_ABCount != 0){
            TMP1 = LibSpoofPrimitives.vectMultWrite(a,b[bi++].values(rix),ai,ai,len);
            while (bi < _ABCount) {
                if(_ACount == 0 && _BCount == 0 && bi == _ABCount-1) {
                    LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(rix), c, 0, ai, ci, len);
                }else {
                    LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(rix), TMP1, 0, ai, 0, len);
                }
            }
        }

        if(_BCount > 0 && TMP1 == null) {
            TMP1 = LibSpoofPrimitives.vectMultWrite(a,b[bi++].values(0),ai,0,len);
        }
        while(bi < _uptoBCumCount) {
            if (_ACount == 0 && bi == _uptoBCumCount - 1) {
                LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(0), c, 0, 0, ci, len);
            } else {
                LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(0), TMP1, 0, 0, 0, len);
            }
        }

        if(_ACount == 1) {
            LibMatrixMult.vectMultiplyWrite(b[bi].values(0)[rix],TMP1,c,0,ci,len);
        }
    }

    protected void genexec_B(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len, long grix, int rix) {
        int bi = 0;
        double[] TMP1 = null;
        if (_ABCount != 0){
            TMP1 = LibSpoofPrimitives.vectMultWrite(a,b[bi++].values(rix),ai,ai,len);
            while (bi < _ABCount) {
                if(_ACount == 0 && _BCount == 0 && bi == _ABCount-1) {
                    LibMatrixMult.vectMultiplyAdd(TMP1, b[bi++].values(rix), c, 0, ai, 0, len);
                }else {
                    LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(rix), TMP1, 0, ai, 0, len);
                }
            }
        }

        if(_BCount > 0 && TMP1 == null) {
            TMP1 = LibSpoofPrimitives.vectMultWrite(a,b[bi++].values(0),ai,0,len);
        }
        while(bi < _uptoBCumCount) {
            if (_ACount == 0 && bi == _uptoBCumCount - 1) {
                LibMatrixMult.vectMultiplyAdd(TMP1, b[bi++].values(0), c, 0, 0, 0, len);
            } else {
                LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(0), TMP1, 0, 0, 0, len);
            }
        }

        if(_ACount == 1) {
            LibMatrixMult.vectMultiplyAdd(b[bi].values(0)[rix],TMP1,c,0,0,len);
        }
    }

    protected void genexec_A_or_(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len, long grix, int rix) {
        int bi = 0;
        double[] TMP1 = null;
        double TMP2 = 0;
        if (_ABCount == 0 && _BCount == 0){
            TMP2 = LibSpoofPrimitives.dotProduct(a,b[bi++].values(rix),ai,ai,len);
        }
        else if (_ABCount != 0){
            TMP1 = LibSpoofPrimitives.vectMultWrite(a,b[bi++].values(rix),ai,ai,len);
            while (bi < _ABCount) {
                if(_BCount == 0 && bi == _ABCount - 1) {
                    TMP2 = LibSpoofPrimitives.dotProduct(TMP1,b[bi++].values(rix),0,ai,len);
                }else {
                    LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(rix), TMP1, 0, ai, 0, len);
                }
            }
        }

        if(_BCount == 1 && TMP1 == null) {
            TMP2 = LibSpoofPrimitives.dotProduct(a,b[bi++].values(0),ai,0,len);
        }
        else if(_BCount > 0 && TMP1 == null) {
            TMP1 = LibSpoofPrimitives.vectMultWrite(a,b[bi++].values(0),ai,0,len);
        }
        while(bi < _uptoBCumCount) {
            if(bi == _uptoBCumCount -1){
                TMP2 = LibSpoofPrimitives.dotProduct(TMP1,b[bi++].values(0),0,0,len);
            }
            else {
                LibMatrixMult.vectMultiplyWrite(TMP1, b[bi++].values(0), TMP1, 0, 0, 0, len);
            }
        }

        if(_ACount == 1) {
            TMP2 *= b[bi].values(0)[rix];
        }
        if (_EinsumRewriteType == EOpNodeFuse.EinsumRewriteType.AB_BA_B_A__A) c[ci] = TMP2;
        else c[0] += TMP2;
    }

    protected void genexec(double[] avals, int[] aix, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int alen, int len, long grix, int rix) {
		throw new RuntimeException("Sparse fused einsum not implemented");
    }
}
