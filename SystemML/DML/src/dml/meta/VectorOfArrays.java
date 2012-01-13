package dml.meta;

import java.util.Vector;

public class VectorOfArrays {
	public Vector<long[]> thevec;	//each long[] is of length 1<<30 
	public long length;		//num rows/cols
	int width;		//num folds/iterns
	public VectorOfArrays() {
		thevec = null;
		length = width = 0;
	}
	public VectorOfArrays(long l, int w) {
		length = l;
		width = w;
		long numarrays = (length * width) / (1 << 30);	//typically, very few arrays, since each is 1bln!
		thevec = new Vector<long[]>();
		for(long i = 0; i < numarrays-1; i++) {
			thevec.add(new long[(1<<30)]);	//the system will possibly run of out memory anyways (each array is 4GB!)
		}
		int leftover = (int) ((length * width) - (numarrays * (1 << 30))); //cast wont cause problems
		thevec.add(new long[leftover]);	
	}
	public VectorOfArrays(VectorOfArrays that) {
		length = that.length;
		width = that.width;
		thevec = new VectorOfArrays(length, width).thevec;
		thevec = that.thevec;
	}

	public void set(long yindex, int xindex, long value) {
		long absolindex = yindex * width + xindex;	//since we store in row major order
		int arraynum = (int)(absolindex / (1 << 30));	//the cast wont cause overflows
		int leftover = (int) (absolindex - arraynum * (1 << 30));
		thevec.get(arraynum)[leftover] = value;
	}

	public long get(long yindex, int xindex) {
		long absolindex = yindex * width + xindex;	//since we store in row major order
		int arraynum = (int)(absolindex / (1 << 30));	//the cast wont cause overflows
		int leftover = (int) (absolindex - arraynum * (1 << 30));
		return thevec.get(arraynum)[leftover];
	}
}
