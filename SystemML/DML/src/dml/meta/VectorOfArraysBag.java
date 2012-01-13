package dml.meta;

import java.util.Vector;

//this handles repetitions in the entries, ie. each prevrowid can map to mult fut rowids
//unlike simple vectofarr of size MT, here, we have size (2+f)MT! we store morelocn indics for each fold; f/2 extra entries for reptns
public class VectorOfArraysBag {
	public Vector<long[]> thevec;	//each long[] is of length 1<<30 
	public long length;		//num rows/cols
	int width;		//num folds/iterns
	double frac;
	long availlocn;	//this gives the currently avail locn (entryid) for reptn entries
	public VectorOfArraysBag() {
		thevec = null;
		availlocn = length = width = 0;
		frac = 0;
	}
	public VectorOfArraysBag(long l, int w, double f) {
		length = l;
		width = w;
		frac = f;
		availlocn = l;	//by default the starting reptn entry goes from index l onwards
		long numarrays = ((long)(length * (2 + 2*f)) + 1) * width / (1 << 30);	//typically, very few arrays, since each is 1bln!
		thevec = new Vector<long[]>();
		for(long i = 0; i < numarrays-1; i++) {
			thevec.add(new long[(1<<30)]);	//the system will possibly run of out memory anyways (each array is 4GB!)
		}
		int leftover = (int) ((((long)(length * (2 + 2*f)) + 1)* width) - (numarrays * (1 << 30))); //cast wont cause problems
		if(leftover != 0) 
			thevec.add(new long[leftover]);	
	}
	public VectorOfArraysBag(VectorOfArraysBag that) {
		length = that.length;
		width = that.width;
		frac = that.frac;
		availlocn = that.availlocn;
		thevec = new VectorOfArraysBag(length, width, frac).thevec;
		thevec = that.thevec;
	}
	
	public static int arraynum(long absol) {
		return (int)(absol / (1 << 30));
	}
	public static int leftover(long absol) {
		return (int)(absol - arraynum(absol) * (1<<30));
	}

	public void set(long yindex, int xindex, long value) {	//here we use the locn indcs and extra reptn entries
		long absolindex = yindex * 2 * width + xindex;	//since we store in row major order, with morelocns alongside
		long absolindexindic = yindex * 2 * width + xindex + width;	//this is the morelocn indicator
		//check locnindic; ! otherwise, go to that entryid!
		if(thevec.get(arraynum(absolindexindic))[leftover(absolindexindic)] == 0) { //this is the first time this prevrowid's value is being set
			thevec.get(arraynum(absolindexindic))[leftover(absolindexindic)] = -1;	//end of the list for this prevrowid
			thevec.get(arraynum(absolindex))[leftover(absolindex)] = value;
		}
		else {	//so the morelocn points to an entryid in the reptn section; so, we need to loop till we find a -1, and then use next availlocn
			long indiclocn = thevec.get(arraynum(absolindexindic))[leftover(absolindexindic)];
			long entryid = yindex;
			while(indiclocn != -1) {
				entryid = indiclocn;
				indiclocn = thevec.get(arraynum(indiclocn * 2 * width + xindex + width))[leftover(indiclocn * 2 * width + xindex + width)];
			} //entryid at end points to last elem of list; use availlocn to add after this
			thevec.get(arraynum(entryid * 2 * width + xindex + width))[leftover(entryid * 2 * width + xindex + width)] = availlocn;
			thevec.get(arraynum(availlocn * 2 * width + xindex))[leftover(availlocn * 2 * width + xindex)] = value;
			thevec.get(arraynum(availlocn * 2 * width + xindex + width))[leftover(availlocn * 2 * width + xindex + width)] = -1; //new end of list
			availlocn++;
		}
	}

	public Vector<Long> get(long yindex, int xindex) {	//we return a vector of futrowids
		Vector<Long> retvals = new Vector<Long>();
		long absolindex = yindex * 2 * width + xindex;	//since we store in row major order, with morelocns alongside
		long absolindexindic = yindex * 2 * width + xindex + width;	//this is the morelocn indicator
		long indiclocn = thevec.get(arraynum(absolindexindic))[leftover(absolindexindic)];
		if(indiclocn == 0) {	//no entry for this prevrowid at all!
			return retvals;
		}
		else {
			retvals.add(thevec.get(arraynum(absolindex))[leftover(absolindex)]);
			while(indiclocn != -1) {	//when indiclocn becomes -1, we have reached end of the list, and stop
				long anentry = thevec.get(arraynum(indiclocn * 2 * width + xindex))[leftover(indiclocn * 2 * width + xindex)];
				retvals.add(anentry);
				indiclocn = thevec.get(arraynum(indiclocn * 2 * width + xindex + width))[leftover(indiclocn * 2 * width + xindex + width)];
			}
			return retvals;
		}
	}
}
