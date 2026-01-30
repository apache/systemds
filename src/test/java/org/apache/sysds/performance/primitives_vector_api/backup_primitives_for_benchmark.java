package org.apache.sysds.performance.primitives_vector_api;

import org.apache.sysds.runtime.matrix.data.LibMatrixMult;



import java.util.Arrays;

import org.apache.commons.math3.util.FastMath;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.functionobjects.BitwAnd;
import org.apache.sysds.runtime.functionobjects.IntegerDivide;
import org.apache.sysds.runtime.functionobjects.Modulus;
import org.apache.sysds.runtime.matrix.data.LibMatrixDNN;
import org.apache.sysds.runtime.matrix.data.LibMatrixDNN.PoolingType;
import org.apache.sysds.runtime.matrix.data.LibMatrixDNNIm2Col;
import org.apache.sysds.runtime.matrix.data.LibMatrixDNNPooling;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorMask;


public class backup_primitives_for_benchmark {

    // Vector API initializations
	private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
	private static final int vLen = SPECIES.length();

    public static double[] allocVector(int len, boolean reset) {
		return allocVector(len, reset, 0);
	}
	
	protected static double[] allocVector(int len, boolean reset, double resetVal) {
		VectorBuffer buff = memPool.get();
		
		//find next matching vector in ring buffer or
		//allocate new vector if required
		double[] vect = buff.next(len);
		if( vect == null )
			vect = new double[len];
		
		//reset vector if required
		if( reset )
			Arrays.fill(vect, resetVal);
		return vect;
	}
    private static class VectorBuffer {
		private static final int MAX_SIZE = 512*1024; //4MB
		private final double[][] _data;
		private int _pos;
		private int _len1;
		private int _len2;
		
		public VectorBuffer(int num, int len1, int len2) {
			//best effort size restriction since large intermediates
			//not necessarily used (num refers to the total number)
			len1 = Math.min(len1, MAX_SIZE);
			len2 = Math.min(len2, MAX_SIZE);
			//pre-allocate ring buffer
			int lnum = (len2>0 && len1!=len2) ? 2*num : num;
			_data = new double[lnum][];
			for( int i=0; i<num; i++ ) {
				if( lnum > num ) {
					_data[2*i] = new double[len1];
					_data[2*i+1] = new double[len2];
				}
				else {
					_data[i] = new double[len1];
				}
			}
			_pos = -1;
			_len1 = len1;
			_len2 = len2;
		}
		public double[] next(int len) {
			if( _len1!=len && _len2!=len )
				return null;
			do {
				_pos = (_pos+1>=_data.length) ? 0 : _pos+1;
			} while( _data[_pos].length!=len );
			return _data[_pos];
		}
		@SuppressWarnings("unused")
		public boolean isReusable(int num, int len1, int len2) {
			int lnum = (len2>0 && len1!=len2) ? 2*num : num;
			return (_len1 == len1 && _len2 == len2
				&& _data.length == lnum);
		}
	}
    private static ThreadLocal<VectorBuffer> memPool = new ThreadLocal<>() {
		@Override protected VectorBuffer initialValue() { return new VectorBuffer(0,0,0); }
	};

    public static void scalarvectDivAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  a[j] / bval;
	}

	public static void vectDivAdd(double[] a, double bval, double[] c, int ai, int ci, int len) { 
		final double inv = 1.0 / bval; 
		final DoubleVector vinv = DoubleVector.broadcast(SPECIES, inv); 
		int i = 0; final int upperBound = SPECIES.loopBound(len); 

		//unrolled vLen-block (for better instruction-level parallelism) 
		for (; i < upperBound; i += vLen) { 
			DoubleVector va = DoubleVector.fromArray(SPECIES, a, ai + i); 
			DoubleVector vc = DoubleVector.fromArray(SPECIES, c, ci + i); 
			vc = vc.add(va.mul(vinv)); vc.intoArray(c, ci + i); 
		} 
		
		//rest, not aligned to vLen-blocks 
		for (; i < len; i++) { 
			c[ci + i] += a[ai + i] * inv;
		} 
	}

    public static void scalarvectDivAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] +=  bval / a[j];
	}

	public static void vectDivAdd(double bval, double[] a, double[] c, int ai, int ci, int len) {
		int i = 0;
		int upperBound = SPECIES.loopBound(len);
		DoubleVector vb = DoubleVector.broadcast(SPECIES, bval);

		//unrolled vLen-block  (for better instruction-level parallelism)
		for (; i < upperBound; i += vLen) {
			DoubleVector va = DoubleVector.fromArray(SPECIES, a, ai + i);
			DoubleVector vc = DoubleVector.fromArray(SPECIES, c, ci + i);
			vc = vc.add(vb.div(va));
			vc.intoArray(c, ci + i);
		}

		//rest, not aligned to vLen-blocks	
		for (;i<len;i++){
			c[ci+i] += bval/a[ai+i];
		}
	}

    public static void scalarvectDivAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += a[j] / bval;
	}

	// not in use: vector api implementation slower than scalar loop version
	public static void vectDivAdd(double[] a, double bval, double[] c, int[] aix, int ai, int ci, int alen, int len) {

		final double inv = 1.0 / bval;
		int i = 0;
		int upperBound = SPECIES.loopBound(alen);
		DoubleVector vinv = DoubleVector.broadcast(SPECIES, inv);

		//unrolled vLen-block  (for better instruction-level parallelism)
		for (; i < upperBound; i += vLen) {
			DoubleVector va = DoubleVector.fromArray(SPECIES, a, ai + i);
			DoubleVector vcontrib = va.mul(vinv);

			// scatter-add lane-by-lane
			for (int lane = 0; lane < vLen; lane++) {
				int idx = ci + aix[ai + i + lane];
				c[idx] += vcontrib.lane(lane);
			}
		}

		//rest, not aligned to vLen-blocks
		for(; i<alen; i++){
			c[ci + aix[ai + i]] += a[ai + i] * inv;
		}
	}

    public static void scalarvectDivAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		for( int j = ai; j < ai+alen; j++ )
			c[ci + aix[j]] += bval / a[j];
	}

	// not in use: vector api implementation slower than scalar loop version
	public static void vectDivAdd(double bval, double[] a, double[] c, int[] aix, int ai, int ci, int alen, int len) {
		int i = 0;
		int upperBound = SPECIES.loopBound(alen);
		DoubleVector vb = DoubleVector.broadcast(SPECIES, bval);

		//unrolled vLen-block  (for better instruction-level parallelism)
		for (; i < upperBound; i += vLen) {
			DoubleVector va = DoubleVector.fromArray(SPECIES, a, ai + i);
			DoubleVector vcontrib = vb.div(va);

			// scatter-add lane-by-lane
			for (int lane = 0; lane < vLen; lane++) {
				int idx = ci + aix[ai + i + lane];
				c[idx] += vcontrib.lane(lane);
			}	
		}
		//rest, not aligned to vLen-blocks
		for (; i<alen; i++){
			c[ci + aix[ai + i]] += bval / a[ai +i];
		}
	}
    public static double[] scalarvectDivWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++)
			c[j] = a[ai+j] / bval;
		return c;
	}

	// not in use: vector api implementation slower than scalar loop version
	public static double[] vectDivWrite(double[] a, double bval, int ai, int len) {
		double[] c = allocVector(len, false);
		final double inv = 1.0 / bval;
		final DoubleVector vinv = DoubleVector.broadcast(SPECIES, inv);
		int i = 0;
		int upper = SPECIES.loopBound(len);

		//unrolled vLen-block  (for better instruction-level parallelism)
		for (; i < upper; i += vLen) {
			DoubleVector va = DoubleVector.fromArray(SPECIES, a, ai + i);
			va.mul(vinv).intoArray(c, i);
		}

		//rest, not aligned to vLen-blocks
		for (; i < len; i++) {
			c[i] = a[ai + i] * inv;
		}
		return c;
	}
    public static double[] scalarvectDivWrite(double bval, double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++)
			c[j] = bval / a[ai + j];
		return c;
	}

	// not in use: vector api implementation slower than scalar loop version
	public static double[] vectDivWrite(double bval, double[] a, int ai, int len) {
		double[] c = allocVector(len, false);
		final DoubleVector vb = DoubleVector.broadcast(SPECIES, bval);
		int i = 0;
		int upper = SPECIES.loopBound(len);

		//unrolled vLen-block  (for better instruction-level parallelism)
		for (; i < upper; i += vLen) {
			DoubleVector va = DoubleVector.fromArray(SPECIES, a, ai + i);
			vb.div(va).intoArray(c, i);
		}

		//rest, not aligned to vLen-blocks
		for (; i<len; i++){
			c[i] = bval / a[ai + i];
		}
		return c;
	}
    public static double[] scalarvectDivWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		for( int j = 0; j < len; j++)
			c[j] = a[ai + j] / b[bi + j];
		return c;
	}

	// not in use: vector api implementation slower than scalar loop version
	public static double[] vectDivWrite(double[] a, double[] b, int ai, int bi, int len) {
		double[] c = allocVector(len, false);
		int i = 0;
		int upper = SPECIES.loopBound(len);

		//unrolled vLen-block  (for better instruction-level parallelism)
		for (; i < upper; i += vLen) {
			DoubleVector va = DoubleVector.fromArray(SPECIES, a, ai + i);
			DoubleVector vb = DoubleVector.fromArray(SPECIES, b, bi + i);
			va.div(vb).intoArray(c, i);
		}

		//rest, not aligned to vLen-blocks
		for(; i <len; i++){
			c[i] = a[ai + i] / b[bi + i];
		}
		return c;
	}
    public static double scalarrowMaxsVectMult(double[] a, double[] b, int ai, int bi, int len) {
		double val = Double.NEGATIVE_INFINITY;
		int j=0;
		for( int i = ai; i < ai+len; i++ )
			val = Math.max(a[i]*b[j++], val);
		return val;
	}

	public static double rowMaxsVectMult(double[] a, double[] b, int ai, int bi, int len) {
		double maxVal = Double.NEGATIVE_INFINITY;
	
		int i = 0;
		int upper = SPECIES.loopBound(len);
	
		DoubleVector vmax = DoubleVector.broadcast(SPECIES, Double.NEGATIVE_INFINITY);

		//unrolled vLen-block  (for better instruction-level parallelism)
		for (; i < upper; i += vLen) {
			DoubleVector va = DoubleVector.fromArray(SPECIES, a, ai + i);
			DoubleVector vb = DoubleVector.fromArray(SPECIES, b, bi + i);
			DoubleVector prod = va.mul(vb);
			vmax = vmax.max(prod);
		}
	
		maxVal = vmax.reduceLanes(VectorOperators.MAX);
	
		//rest, not aligned to vLen-blocks
		for (; i < len; i++) {
			maxVal = Math.max(maxVal, a[ai + i] * b[bi + i]);
		}
	
		return maxVal;
	}
    // note: parameter bi unused
	public static double scalarrowMaxsVectMult(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double val = Double.NEGATIVE_INFINITY;
		for( int i = ai; i < ai+len; i++ )
			val = Math.max(a[i]*b[aix[i]], val);
		return val;
	}

	// not in use: vector api implementation slower than scalar loop version
	public static double rowMaxsVectMult(double[] a, double[] b, int[] aix, int ai, int bi, int len) {
		double scalarMax = Double.NEGATIVE_INFINITY;

		int i = 0;
		int upperBound = SPECIES.loopBound(len);
		DoubleVector vmax = DoubleVector.broadcast(SPECIES, Double.NEGATIVE_INFINITY);

		//unrolled vLen-block  (for better instruction-level parallelism)
		for (; i < upperBound; i += SPECIES.length()) {
			DoubleVector va = DoubleVector.fromArray(SPECIES, a, ai + i);
			DoubleVector vb = DoubleVector.fromArray(SPECIES, b, 0, aix, ai + i);
			DoubleVector prod = va.mul(vb);
			vmax = vmax.max(prod);
		}
		scalarMax = Math.max(scalarMax, vmax.reduceLanes(VectorOperators.MAX));

		//rest, not aligned to vLen-blocks
		for (; i < len; i++) {
			double prod = a[ai + i] * b[aix[ai + i]];
			if (prod > scalarMax)
				scalarMax = prod;
		}
		return scalarMax;
    }
	

    public static double scalarvectSum(double[] a, int ai, int len) { 
		double val = 0;
		final int bn = len%8;
		
		//compute rest
		for( int i = ai; i < ai+bn; i++ )
			val += a[ i ];
		
		//unrolled 8-block (for better instruction-level parallelism)
		for( int i = ai+bn; i < ai+len; i+=8 ) {
			//read 64B cacheline of a, compute cval' = sum(a) + cval
			val += a[ i+0 ] + a[ i+1 ] + a[ i+2 ] + a[ i+3 ]
			     + a[ i+4 ] + a[ i+5 ] + a[ i+6 ] + a[ i+7 ];
		}
		
		//scalar result
		return val; 
	} 
	
	public static double vectSum(double[] a, int ai, int len) {
        double sum = 0d;
        int i = 0;

        DoubleVector acc = DoubleVector.zero(SPECIES);
        int upperBound = SPECIES.loopBound(len);

		//unrolled vLen-block  (for better instruction-level parallelism)
        for (; i < upperBound; i += SPECIES.length()) {
            DoubleVector v = DoubleVector.fromArray(SPECIES, a, ai + i);
            acc = acc.add(v);
        }
        sum += acc.reduceLanes(VectorOperators.ADD);

        //rest, not aligned to vLen-blocks
        for (; i < len; i++) {
            sum += a[ai + i];
        }
        return sum;
    }
    public static double scalarvectMax(double[] a, int ai, int len) { 
		double val = Double.NEGATIVE_INFINITY;
		for( int i = ai; i < ai+len; i++ )
			val = Math.max(a[i], val);
		return val; 
	} 

	public static double vectMax(double[] a, int ai, int len) {
		int i = 0;
		int upperBound = SPECIES.loopBound(len);
		DoubleVector vmax = DoubleVector.broadcast(SPECIES, Double.NEGATIVE_INFINITY);
	
		//unrolled vLen-block  (for better instruction-level parallelism)
		for (; i < upperBound; i += vLen) {
			DoubleVector v = DoubleVector.fromArray(SPECIES, a, ai + i);
			vmax = vmax.max(v);
		}
		double maxVal = vmax.reduceLanes(VectorOperators.MAX);

		//rest, not aligned to vLen-blocks	
		for(;i<len;i++){
			maxVal = Math.max(a[ai + i],maxVal);
		}
		return maxVal;
	}
    public static double scalarvectCountnnz(double[] a, int ai, int len) { 
		int count = 0;
		for( int i = ai; i < ai+len; i++ )
			count += (a[i] != 0) ? 1 : 0;
		return count;
	} 
	public static double vectCountnnz(double[] a, int ai, int len) {	
		int count = 0;
		int i = 0;
		int upperBound = SPECIES.loopBound(len);
		DoubleVector vzero = DoubleVector.zero(SPECIES);
	
		//unrolled vLen-block  (for better instruction-level parallelism)
		for (; i < upperBound; i += vLen) {
			DoubleVector v = DoubleVector.fromArray(SPECIES, a, ai + i);
			VectorMask<Double> nz = v.compare(VectorOperators.NE, vzero);
			count += nz.trueCount();
		}
	
		//rest, not aligned to vLen-blocks	
		for(;i<len;i++){
			count += (a[i] != 0) ? 1 : 0;
		}
		return count;
	}
    public static void scalarvectEqualAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		for( int j = ai; j < ai+len; j++, ci++)
			c[ci] += (a[j] == bval) ? 1 : 0;
	}
	public static void vectEqualAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
		int i = 0;
		int upper = SPECIES.loopBound(len);
		final DoubleVector bVec   = DoubleVector.broadcast(SPECIES, bval);
		final DoubleVector ones   = DoubleVector.broadcast(SPECIES, 1.0);
		final DoubleVector zeros  = DoubleVector.zero(SPECIES);

		//unrolled vLen-block  (for better instruction-level parallelism)
		for (; i < upper; i += vLen) {
			DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai + i);
			DoubleVector cVec = DoubleVector.fromArray(SPECIES, c, ci + i);

			VectorMask<Double> eq = aVec.compare(VectorOperators.EQ, bVec);

			DoubleVector inc = zeros.blend(ones, eq);

			cVec.add(inc).intoArray(c, ci + i);
		}

		//rest, not aligned to vLen-blocks
		for (; i < len; i++) {
			c[ci + i] += (a[ai + i] == bval) ? 1.0 : 0.0;
			}
		}
    public static double[] scalarvectEqualWrite(double[] a, double bval, int ai, int len) {
            double[] c = allocVector(len, false);
            for( int j = 0; j < len; j++, ai++)
                c[j] = (a[ai] == bval) ? 1 : 0;
            return c;
        }
        public static double[] vectEqualWrite(double[] a, double bval, int ai, int len) {
            double[] c = allocVector(len, false);
            int i = 0;
            int upper = SPECIES.loopBound(len);
            DoubleVector vb = DoubleVector.broadcast(SPECIES, bval);
            DoubleVector zeros = DoubleVector.zero(SPECIES);
            DoubleVector ones = DoubleVector.broadcast(SPECIES, 1.0);
        
            //unrolled vLen-block  (for better instruction-level parallelism)
            for (; i < upper; i += vLen) {
                DoubleVector va = DoubleVector.fromArray(SPECIES, a, ai + i);
                var mask = va.compare(VectorOperators.EQ, vb);
                DoubleVector out = zeros.blend(ones, mask);
                out.intoArray(c, i);
            }
        
            //rest, not aligned to vLen-blocks
            for (; i < len; i++) {
                c[i] = (a[ai + i] == bval) ? 1 : 0;
            }
            return c;
        }
            public static double[] scalarvectEqualWrite(double[] a, double[] b, int ai, int bi, int len) {
                double[] c = allocVector(len, false);
                for( int j = 0; j < len; j++, ai++, bi++)
                    c[j] = (a[ai] == b[bi]) ? 1 : 0;
                return c;
            }
        
    public static double[] vectEqualWrite(double[] a, double[] b, int ai, int bi, int len) {
                double[] c = allocVector(len, false);
                final DoubleVector ones  = DoubleVector.broadcast(SPECIES, 1.0);
                final DoubleVector zeros = DoubleVector.zero(SPECIES);
                int i = 0;
                int upper = SPECIES.loopBound(len);
        
                //unrolled vLen-block  (for better instruction-level parallelism)
                for (; i < upper; i += vLen) {
                    DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai + i);
                    DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, bi + i);
                    VectorMask<Double> eq = aVec.compare(VectorOperators.EQ, bVec);
                    DoubleVector out = zeros.blend(ones, eq);
        
                    out.intoArray(c, i);
                }
        
                   //rest, not aligned to vLen-blocks
                for (; i < len; i++) {
                    c[i] = (a[ai + i] == b[bi + i]) ? 1.0 : 0.0;
                }
                return c;
            }
            public static double[] vectNotequalWrite(double[] a, double[] b, int ai, int bi, int len) {
                double[] c = allocVector(len, false);
                for( int j = 0; j < len; j++, ai++, bi++)
                    c[j] = (a[ai] != b[bi]) ? 1 : 0;
                return c;
            }
        
            // not in use: vector api implementation slower than scalar loop version
public static double[] vectNotequalWrite_vector_api(double[] a, double[] b, int ai, int bi, int len) {
                double[] c = allocVector(len, false);
                final DoubleVector ones  = DoubleVector.broadcast(SPECIES, 1.0);
                final DoubleVector zeros = DoubleVector.zero(SPECIES);
                int i = 0;
                int upper = SPECIES.loopBound(len);
                
                //unrolled vLen-block  (for better instruction-level parallelism)
                for (; i < upper; i += vLen) {
                    DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai + i);
                    DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, bi + i);
        
                    VectorMask<Double> ne = aVec.compare(VectorOperators.NE, bVec);
                    DoubleVector out = zeros.blend(ones, ne);
        
                    out.intoArray(c, i);
                }
        
                //rest, not aligned to vLen-blocks
                for (; i < len; i++) {
                    c[i] = (a[ai + i] != b[bi + i]) ? 1.0 : 0.0;
                }
                return c;
                }


                public static void scalarvectLessAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
                    for( int j = ai; j < ai+len; j++, ci++)
                        c[ci] += (a[j] < bval) ? 1 : 0;
                }
    public static void vectLessAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
                    final DoubleVector bVec  = DoubleVector.broadcast(SPECIES, bval);
                    final DoubleVector ones  = DoubleVector.broadcast(SPECIES, 1.0);
                    final DoubleVector zeros = DoubleVector.zero(SPECIES);
            
                    int i = 0;
                    int upper = SPECIES.loopBound(len);
            
                    //unrolled vLen-block  (for better instruction-level parallelism)
                    for (; i < upper; i += vLen) {
                        DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai + i);
                        DoubleVector cVec = DoubleVector.fromArray(SPECIES, c, ci + i);
            
                        VectorMask<Double> lt = aVec.compare(VectorOperators.LT, bVec);
                        DoubleVector inc = zeros.blend(ones, lt);
            
                        cVec.add(inc).intoArray(c, ci + i);
                    }
            
                    //rest, not aligned to vLen-blocks
                    for (; i < len; i++) {
                        c[ci + i] += (a[ai + i] < bval) ? 1.0 : 0.0;
                        }
                    }


    public static double[] scalarvectLessWrite(double[] a, double bval, int ai, int len) {
                        double[] c = allocVector(len, false);
                        for( int j = 0; j < len; j++, ai++)
                            c[j] = (a[ai] < bval) ? 1 : 0;
                        return c;
                    }
                
                
    public static double[] vectLessWrite(double[] a, double bval, int ai, int len) {
                        double[] c = allocVector(len, false);
                        final DoubleVector bVec  = DoubleVector.broadcast(SPECIES, bval);
                        final DoubleVector ones  = DoubleVector.broadcast(SPECIES, 1.0);
                        final DoubleVector zeros = DoubleVector.zero(SPECIES);
                
                        int i = 0;
                        int upper = SPECIES.loopBound(len);
                
                        //unrolled vLen-block  (for better instruction-level parallelism)
                        for (; i < upper; i += vLen) {
                            DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai + i);
                
                            VectorMask<Double> lt = aVec.compare(VectorOperators.LT, bVec);
                            DoubleVector out = zeros.blend(ones, lt);
                
                            out.intoArray(c, i);
                        }
                
                        //rest, not aligned to vLen-blocks
                        for (; i < len; i++) {
                            c[i] = (a[ai + i] < bval) ? 1.0 : 0.0;
                        }
                
                        return c;
                    }

                    public static double[] scalarvectLessWrite(double[] a, double[] b, int ai, int bi, int len) {
                        double[] c = allocVector(len, false);
                        for( int j = 0; j < len; j++, ai++, bi++)
                            c[j] = (a[ai] < b[bi]) ? 1 : 0;
                        return c;
                    }
                
                    public static double[] vectLessWrite(double[] a, double[] b, int ai, int bi, int len) {
                        double[] c = allocVector(len, false);
                
                        final DoubleVector ones  = DoubleVector.broadcast(SPECIES, 1.0);
                        final DoubleVector zeros = DoubleVector.zero(SPECIES);
                
                        int i = 0;
                        int upper = SPECIES.loopBound(len);
                
                        //unrolled vLen-block  (for better instruction-level parallelism)
                        for (; i < upper; i += vLen) {
                            DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai + i);
                            DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, bi + i);
                
                            VectorMask<Double> lt = aVec.compare(VectorOperators.LT, bVec);
                            DoubleVector out = zeros.blend(ones, lt);
                
                            out.intoArray(c, i);
                        }
                
                        //rest, not aligned to vLen-blocks
                        for (; i < len; i++) {
                        c[i] = (a[ai + i] < b[bi + i]) ? 1.0 : 0.0;
                        }
                
                        return c;
                        }
                        public static void scalarvectLessequalAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
                            for( int j = ai; j < ai+len; j++, ci++)
                                c[ci] += (a[j] <= bval) ? 1 : 0;
                        }
                    
                        public static void vectLessequalAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
                            final DoubleVector bVec  = DoubleVector.broadcast(SPECIES, bval);
                            final DoubleVector ones  = DoubleVector.broadcast(SPECIES, 1.0);
                            final DoubleVector zeros = DoubleVector.zero(SPECIES);
                    
                            int i = 0;
                            int upper = SPECIES.loopBound(len);
                    
                            //unrolled vLen-block  (for better instruction-level parallelism)
                            for (; i < upper; i += vLen) {
                                DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai + i);
                                DoubleVector cVec = DoubleVector.fromArray(SPECIES, c, ci + i);
                    
                                VectorMask<Double> le = aVec.compare(VectorOperators.LE, bVec);
                                DoubleVector inc = zeros.blend(ones, le);
                    
                                cVec.add(inc).intoArray(c, ci + i);
                            }
                    
                            //rest, not aligned to vLen-blocks
                            for (; i < len; i++) {
                                c[ci + i] += (a[ai + i] <= bval) ? 1.0 : 0.0;
                            }
                            }
                            public static double[] scalarvectLessequalWrite(double[] a, double bval, int ai, int len) {
                                double[] c = allocVector(len, false);
                                for( int j = 0; j < len; j++, ai++)
                                    c[j] = (a[ai] <= bval) ? 1 : 0;
                                return c;
                            }
                            public static double[] vectLessequalWrite(double[] a, double bval, int ai, int len) {
                                double[] c = allocVector(len, false);
                                final DoubleVector bVec  = DoubleVector.broadcast(SPECIES, bval);
                                final DoubleVector ones  = DoubleVector.broadcast(SPECIES, 1.0);
                                final DoubleVector zeros = DoubleVector.zero(SPECIES);
                        
                                int i = 0;
                                int upper = SPECIES.loopBound(len);
                        
                                //unrolled vLen-block  (for better instruction-level parallelism)
                                for (; i < upper; i += vLen) {
                                    DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai + i);
                        
                                    VectorMask<Double> le = aVec.compare(VectorOperators.LE, bVec);
                                    DoubleVector out = zeros.blend(ones, le);
                        
                                    out.intoArray(c, i);
                                }
                        
                                //rest, not aligned to vLen-blocks
                                for (; i < len; i++) {
                                    c[i] = (a[ai + i] <= bval) ? 1.0 : 0.0;
                                }
                        
                                return c;
                            }
                            public static double[] scalarvectLessequalWrite(double[] a, double[] b, int ai, int bi, int len) {
                                double[] c = allocVector(len, false);
                                for( int j = 0; j < len; j++, ai++, bi++)
                                    c[j] = (a[ai] <= b[bi]) ? 1 : 0;
                                return c;
                            }
                        
                            public static double[] vectLessequalWrite(double[] a, double[] b, int ai, int bi, int len) {
                                double[] c = allocVector(len, false);
                        
                                final DoubleVector ones  = DoubleVector.broadcast(SPECIES, 1.0);
                                final DoubleVector zeros = DoubleVector.zero(SPECIES);
                        
                                int i = 0;
                                int upper = SPECIES.loopBound(len);
                        
                                //unrolled vLen-block  (for better instruction-level parallelism)
                                for (; i < upper; i += vLen) {
                                DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai + i);
                                DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, bi + i);
                        
                                VectorMask<Double> le = aVec.compare(VectorOperators.LE, bVec);
                                DoubleVector out = zeros.blend(ones, le);
                        
                                out.intoArray(c, i);
                                }
                        
                                //rest, not aligned to vLen-blocks
                                for (; i < len; i++) {
                                c[i] = (a[ai + i] <= b[bi + i]) ? 1.0 : 0.0;
                                }
                        
                                return c;
                                }
                                public static void scalarvectGreaterAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
                                    for( int j = ai; j < ai+len; j++, ci++)
                                        c[ci] += (a[j] > bval) ? 1 : 0;
                                }
                            
                                public static void vectGreaterAdd(double[] a, double bval, double[] c, int ai, int ci, int len) {
                                    final DoubleVector bVec  = DoubleVector.broadcast(SPECIES, bval);
                                    final DoubleVector ones  = DoubleVector.broadcast(SPECIES, 1.0);
                                    final DoubleVector zeros = DoubleVector.zero(SPECIES);
                            
                                    int i = 0;
                                    int upper = SPECIES.loopBound(len);
                            
                                    //unrolled vLen-block  (for better instruction-level parallelism)
                                    for (; i < upper; i += vLen) {
                                        DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai + i);
                                        DoubleVector cVec = DoubleVector.fromArray(SPECIES, c, ci + i);
                            
                                        VectorMask<Double> gt = aVec.compare(VectorOperators.GT, bVec);
                                        DoubleVector inc = zeros.blend(ones, gt);
                            
                                        cVec.add(inc).intoArray(c, ci + i);
                                    }
                            
                                    //rest, not aligned to vLen-blocks
                                    for (; i < len; i++) {
                                        c[ci + i] += (a[ai + i] > bval) ? 1.0 : 0.0;
                                    }
                                    }
                                    public static double[] scalarvectGreaterWrite(double[] a, double bval, int ai, int len) {
                                        double[] c = allocVector(len, false);
                                        for( int j = 0; j < len; j++, ai++)
                                            c[j] = (a[ai] > bval) ? 1 : 0;
                                        return c;
                                    }
                                    public static double[] vectGreaterWrite(double[] a, double bval, int ai, int len) {
                                        double[] c = allocVector(len, false);
                                        final DoubleVector bVec  = DoubleVector.broadcast(SPECIES, bval);
                                        final DoubleVector ones  = DoubleVector.broadcast(SPECIES, 1.0);
                                        final DoubleVector zeros = DoubleVector.zero(SPECIES);
                                
                                        int i = 0;
                                        int upper = SPECIES.loopBound(len);
                                
                                        //unrolled vLen-block  (for better instruction-level parallelism)
                                        for (; i < upper; i += vLen) {
                                            DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai + i);
                                
                                            VectorMask<Double> gt = aVec.compare(VectorOperators.GT, bVec);
                                            DoubleVector out = zeros.blend(ones, gt);
                                
                                            out.intoArray(c, i);
                                        }
                                
                                        //rest, not aligned to vLen-blocks
                                        for (; i < len; i++) {
                                            c[i] = (a[ai + i] > bval) ? 1.0 : 0.0;
                                        }
                                        return c;
                                    }
                                    public static void scalarvectMult2Add(double[] a, double[] c, int ai, int ci, int len) {
                                        for( int j = ai; j < ai+len; j++, ci++)
                                            c[ci] +=  a[j] + a[j];
                                    }
                                
                                    public static void vectMult2Add(double[] a, double[] c, int ai, int ci, int len) {
                                        LibMatrixMult.vectMultiplyAdd(2.0,a,c,ai,ci,len);
                                    }

                                    public static double[] scalarvectGreaterWrite(double[] a, double[] b, int ai, int bi, int len) {
                                        double[] c = allocVector(len, false);
                                        for( int j = 0; j < len; j++, ai++, bi++)
                                            c[j] = (a[ai] > b[bi]) ? 1 : 0;
                                        return c;
                                    }
                                
                                    // not in use: vector api implementation slower than scalar loop version
                                    public static double[] vectGreaterWrite(double[] a, double[] b, int ai, int bi, int len) {
                                        double[] c = allocVector(len, false);
                                        final DoubleVector ones  = DoubleVector.broadcast(SPECIES, 1.0);
                                        final DoubleVector zeros = DoubleVector.zero(SPECIES);
                                
                                        int i = 0;
                                        int upper = SPECIES.loopBound(len);
                                
                                        //unrolled vLen-block  (for better instruction-level parallelism)
                                        for (; i < upper; i += vLen) {
                                            DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, ai + i);
                                            DoubleVector bVec = DoubleVector.fromArray(SPECIES, b, bi + i);
                                
                                            VectorMask<Double> gt = aVec.compare(VectorOperators.GT, bVec);
                                            DoubleVector out = zeros.blend(ones, gt);
                                
                                            out.intoArray(c, i);
                                        }
                                
                                        //rest, not aligned to vLen-blocks
                                        for (; i < len; i++) {
                                            c[i] = (a[ai + i] > b[bi + i]) ? 1.0 : 0.0;
                                        }
                                        return c;
                                        }
    
}
