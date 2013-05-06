package com.ibm.bi.dml.runtime.util;

import java.util.Random;

import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;

/**
 * Utilities for sorting, primarily used for SparseRows.
 * 
 */
public class SortUtils 
{
	/**
	 * 
	 * @param start
	 * @param end
	 * @param indexes
	 * @return
	 */
	public static boolean isSorted(int start, int end, int[] indexes)
	{
		boolean ret = true;
		for( int i=start+1; i<end; i++ )
    		if( indexes[i]<indexes[i-1] ){
    			ret = false;
    			break;
    		}
		return ret;
	}
	
	/**
	 * In-place sort of two arrays, only indexes is used for comparison and values
	 * of same position are sorted accordingly. 
	 * 
	 * NOTE: This is a copy of IBM JDK Arrays.sort, extended for two related arrays.
	 * 
     * @param start
     * @param end
     * @param indexes
     * @param values
     */
    public static void sort(int start, int end, int[] indexes, double[] values) 
    {
        int tempIx;
        double tempVal;
        
        int length = end - start;
        if (length < 7) {
            for (int i = start + 1; i < end; i++) {
                for (int j = i; j > start && indexes[j - 1] > indexes[j]; j--) {
                    tempIx = indexes[j];
                    indexes[j] = indexes[j - 1];
                    indexes[j - 1] = tempIx;
                    tempVal = values[j];
                    values[j] = values[j - 1];
                    values[j - 1] = tempVal;
                }
            }
            return;
        }
        int middle = (start + end) / 2;
        if (length > 7) {
            int bottom = start;
            int top = end - 1;
            if (length > 40) {
                length /= 8;
                bottom = med3(indexes, bottom, bottom + length, bottom
                        + (2 * length));
                middle = med3(indexes, middle - length, middle, middle + length);
                top = med3(indexes, top - (2 * length), top - length, top);
            }
            middle = med3(indexes, bottom, middle, top);
        }
        int partionValue = indexes[middle];
        int a, b, c, d;
        a = b = start;
        c = d = end - 1;
        while (true) {
            while (b <= c && indexes[b] <= partionValue) {
                if (indexes[b] == partionValue) {
                    tempIx = indexes[a];
                    indexes[a] = indexes[b];
                    indexes[b] = tempIx;
                    tempVal = values[a];
                    values[a++] = values[b];
                    values[b] = tempVal;
                }
                b++;
            }
            while (c >= b && indexes[c] >= partionValue) {
                if (indexes[c] == partionValue) {
                    tempIx = indexes[c];
                    indexes[c] = indexes[d];
                    indexes[d] = tempIx;
                    tempVal = values[c];
                    values[c] = values[d];
                    values[d--] = tempVal;
                }
                c--;
            }
            if (b > c) {
                break;
            }
            tempIx = indexes[b];
            indexes[b] = indexes[c];
            indexes[c] = tempIx;
            tempVal = values[b];
            values[b++] = values[c];
            values[c--] = tempVal;
        }
        length = a - start < b - a ? a - start : b - a;
        int l = start;
        int h = b - length;
        while (length-- > 0) {
            tempIx = indexes[l];
            indexes[l] = indexes[h];
            indexes[h] = tempIx;
            tempVal = values[l];
            values[l++] = values[h];
            values[h++] = tempVal;
        }
        length = d - c < end - 1 - d ? d - c : end - 1 - d;
        l = b;
        h = end - length;
        while (length-- > 0) {
            tempIx = indexes[l];
            indexes[l] = indexes[h];
            indexes[h] = tempIx;
            tempVal = values[l];
            values[l++] = values[h];
            values[h++] = tempVal;
        }
        if ((length = b - a) > 0) {
            sort(start, start + length, indexes, values);
        }
        if ((length = d - c) > 0) {
            sort(end - length, end, indexes, values);
        }
    }
    
    /**
     * 
     * @param array
     * @param a
     * @param b
     * @param c
     * @return
     */
    private static int med3(int[] array, int a, int b, int c) 
    {
        int x = array[a], y = array[b], z = array[c];
        return x < y ? (y < z ? b : (x < z ? c : a)) : (y > z ? b : (x > z ? c
                : a));
    }
    
    
    public static void main(String[] args)
    {
    	int n = 10000000;
    	int[] indexes = new int[n];
    	double[] values = new double[n];
    	Random rand = new Random();
    	for( int i=0; i<n; i++ )
    	{
    		indexes[i] = rand.nextInt();
    		values[i] = rand.nextDouble();
    	}
    	
    	System.out.println("Running quicksort test ...");
    	Timing time = new Timing();
    	
    	time.start();   	
    	SortUtils.sort(0, indexes.length, indexes, values);    	
    	System.out.println("quicksort n="+n+" in "+time.stop()+"ms.");
    	
    	time.start();   	
    	boolean flag = SortUtils.isSorted(0, indexes.length, indexes);
    	System.out.println("check sorted n="+n+" in "+time.stop()+"ms, "+flag+".");
    }
}
