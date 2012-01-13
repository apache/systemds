/*
 *  This software may only be used by you under license from AT&T Corp.
 *  ("AT&T").  A copy of AT&T's Source Code Agreement is available at
 *  AT&T's Internet website having the URL:
 *  <http://www.research.att.com/sw/tools/graphviz/license/source.html>
 *  If you received this software without first entering into a license
 *  with AT&T, you have an infringing copy of this software and cannot use
 *  it without violating AT&T's intellectual property rights.
 */

package dml.utils.visualize.grappa.att;
import java.io.*;
import java.util.*;

/**
 * A class providing <I>sprintf</I> support.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a> and Rich Drechsler, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class GrappaSupportPrintf

    implements GrappaConstants

{

    ///////////////////////////////////////////////////////////////////////////
    //
    // GrappaSupportPrintf
    //
    ///////////////////////////////////////////////////////////////////////////

    /**
     * The familiar C-language sprintf rendered in Java and extended for
     * some Grappa types.
     *
     * @param args the first element of this array is a string giving the
     *             format of the returned string, the remaining elements
     *             are object to be formatted according to the format.
     * @return a string giving a formatted representation of the arguments.
     */
    public final static String
    sprintf(Object args[]) {

	PrintfParser	cvt;
	StringBuffer	prtbuf;
	char		format[];
	int		flen;
	int		argn;
	int		n;
	char		ch;
	boolean         flag;

	if(!(args[0] instanceof String)) {
	    throw new RuntimeException("initial argument must be format String");
	}

	argn = 0;
	format = ((String)args[argn++]).toCharArray();

	flen = format.length;
	prtbuf = new StringBuffer(2 * flen);
	cvt = new PrintfParser();

	for (n = 0; n < flen; ) {
	    if ((ch = format[n++]) == '%') {
		if ((n = cvt.parse(format, n)) < flen) {
		    switch (ch = format[n++]) {
		    case 'b':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			if (args[argn] instanceof GrappaBox)
			    flag = ((GrappaBox)args[argn]).isDimensioned();
			else
			    flag = true;
			if (args[argn] instanceof java.awt.geom.Rectangle2D)
			    cvt.buildBox(prtbuf, ((java.awt.geom.Rectangle2D)args[argn++]), false, flag);
			else throw new RuntimeException("argument " + argn + " should be a Rectangle2D");
			break;

		    case 'B':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			if (args[argn] instanceof GrappaBox)
			    flag = ((GrappaBox)args[argn]).isDimensioned();
			else
			    flag = true;
			if (args[argn] instanceof java.awt.geom.Rectangle2D)
			    cvt.buildBox(prtbuf, ((java.awt.geom.Rectangle2D)args[argn++]), true, flag);
			else throw new RuntimeException("argument " + argn + " should be a Rectangle2D");
			break;
		    case 'c':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			if (args[argn] instanceof Character)
			    cvt.buildChar(prtbuf, ((Character)args[argn++]).charValue());
			else throw new RuntimeException("argument " + argn + " should be a Character");
			break;

		    case 'd':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			if (args[argn] instanceof Number)
			    cvt.buildInteger(prtbuf, ((Number)args[argn++]).intValue());
			else throw new RuntimeException("argument " + argn + " should be a Number");
			break;

		    case 'o':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			if (args[argn] instanceof Character)
			    cvt.buildOctal(prtbuf, ((Character)args[argn++]).charValue());
			else if (args[argn] instanceof Number)
			    cvt.buildOctal(prtbuf, ((Number)args[argn++]).intValue());
			else throw new RuntimeException("argument " + argn + " should be a Character or Number");
			break;

		    case 'p':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			if (args[argn] instanceof java.awt.geom.Point2D)
			    cvt.buildPoint(prtbuf, ((java.awt.geom.Point2D)args[argn++]), false);
			else if (args[argn] instanceof java.awt.geom.Dimension2D)
			    cvt.buildSize(prtbuf, ((java.awt.geom.Dimension2D)args[argn++]), false);
			else throw new RuntimeException("argument " + argn + " should be a Point2D");
			break;

		    case 'P':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			if (args[argn] instanceof java.awt.geom.Point2D)
			    cvt.buildPoint(prtbuf, ((java.awt.geom.Point2D)args[argn++]), true);
			else if (args[argn] instanceof java.awt.geom.Dimension2D)
			    cvt.buildSize(prtbuf, ((java.awt.geom.Dimension2D)args[argn++]), true);
			else throw new RuntimeException("argument " + argn + " should be a Point2D");
			break;

		    case 'x':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			if (args[argn] instanceof Character)
			    cvt.buildHex(prtbuf, ((Character)args[argn++]).charValue(), false);
			else if (args[argn] instanceof Number)
			    cvt.buildHex(prtbuf, ((Number)args[argn++]).intValue(), false);
			else throw new RuntimeException("argument " + argn + " should be a Character or Number");
			break;

		    case 'X':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			if (args[argn] instanceof Character)
			    cvt.buildHex(prtbuf, ((Character)args[argn++]).charValue(), true);
			else if (args[argn] instanceof Number)
			    cvt.buildHex(prtbuf, ((Number)args[argn++]).intValue(), true);
			else throw new RuntimeException("argument " + argn + " should be a Character or Number");
			break;

		    case 'e':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			if (args[argn] instanceof Character)
			    cvt.buildExp(prtbuf, ((Character)args[argn++]).charValue(), false);
			else if (args[argn] instanceof Number)
			    cvt.buildExp(prtbuf, ((Number)args[argn++]).doubleValue(), false);
			else throw new RuntimeException("argument " + argn + " should be a Character or Number");
			break;

		    case 'E':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			if (args[argn] instanceof Character)
			    cvt.buildExp(prtbuf, ((Character)args[argn++]).charValue(), true);
			else if (args[argn] instanceof Number)
			    cvt.buildExp(prtbuf, ((Number)args[argn++]).doubleValue(), true);
			else throw new RuntimeException("argument " + argn + " should be a Character or Number");
			break;

		    case 'f':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			if (args[argn] instanceof Character)
			    cvt.buildFloat(prtbuf, ((Character)args[argn++]).charValue());
			else if (args[argn] instanceof Number)
			    cvt.buildFloat(prtbuf, ((Number)args[argn++]).doubleValue());
			else throw new RuntimeException("argument " + argn + " should be a Character or Number");
			break;

		    case 'g':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			if (args[argn] instanceof Character)
			    cvt.buildFlex(prtbuf, ((Character)args[argn++]).charValue(), false);
			else if (args[argn] instanceof Number)
			    cvt.buildFlex(prtbuf, ((Number)args[argn++]).doubleValue(), false);
			else throw new RuntimeException("argument " + argn + " should be a Character or Number");
			break;

		    case 'G':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			if (args[argn] instanceof Character)
			    cvt.buildFlex(prtbuf, ((Character)args[argn++]).charValue(), true);
			else if (args[argn] instanceof Number)
			    cvt.buildFlex(prtbuf, ((Number)args[argn++]).doubleValue(), true);
			else throw new RuntimeException("argument " + argn + " should be a Character or Number");
			break;

		    case 's':
			if (args.length <= argn) throw new RuntimeException("too few arguments for format");
			cvt.buildString(prtbuf, args[argn++].toString());
			break;

		    case '%':
			prtbuf.append('%');
			break;

		    default:
			// different compilers handle this different ways,
			// some just do the equivalent of prtbuf.append(ch),
			// but we will just ignore the unrecognized format
			break;
		    }
		} else prtbuf.append(ch);
	    } else if (ch == '\\') {
		if (n < flen) {
		    switch (ch = format[n++]) {
		    case 'b':
			prtbuf.append('\b');
			break;
		    case 'f':
			prtbuf.append('\f');
			break;
		    case 'n':
			prtbuf.append('\n');
			break;
		    case 'r':
			prtbuf.append('\r');
			break;
		    case 't':
			prtbuf.append('\t');
			break;
		    case 'u':
			if ((n+3) < flen) {
			    if (
				GrappaSupport.isdigit(format[n])
				&&
				GrappaSupport.isdigit(format[n+1])
				&&
				GrappaSupport.isdigit(format[n+2])
				&&
				GrappaSupport.isdigit(format[n+3])
				) {
				int uni = (int)format[n+3]+16*(int)format[n+2]+256*(int)format[n+1]+4096*(int)format[n];
				prtbuf.append((char)uni);
				n += 4;
			    } else prtbuf.append('u');
			} else prtbuf.append('u');
			break;
		    case '"':
			prtbuf.append('\"');
			break;
		    case '\'':
			prtbuf.append('\'');
			break;
		    case '\\':
			prtbuf.append('\\');
			break;
		    case '0':
		    case '1':
		    case '2':
		    case '3':
		    case '4':
		    case '5':
		    case '6':
		    case '7':
		    case '8':
		    case '9':
			// need to fix this, assumes 3 digit octals
			if ((n+1) < flen) {
			    if (
				GrappaSupport.isdigit(format[n])
				&&
				GrappaSupport.isdigit(format[n+1])
				) {
				int oct = (int)format[n+1]+8*(int)format[n]+64*(int)ch;
				prtbuf.append((char)oct);
				n += 2;
			    } else prtbuf.append(ch);
			} else prtbuf.append(ch);
			break;
		    }
		} else prtbuf.append(ch);
	    } else prtbuf.append(ch);
	}

	return(prtbuf.toString());
    }

    ///////////////////////////////////////////////////////////////////////////
}

class PrintfParser

    implements GrappaConstants

{

    private boolean		alternate;
    private boolean		rightpad;
    private boolean		sign;
    private boolean		space;
    private boolean		zeropad;
    private boolean		trim;
    private int			precision;
    private int			width;
    private String		plus;
    private char		padding;
    private StringBuffer	scratch;

    ///////////////////////////////////////////////////////////////////////////
    //
    // Constructor
    //
    ///////////////////////////////////////////////////////////////////////////

    PrintfParser() {

	scratch = new StringBuffer();
    }

    ///////////////////////////////////////////////////////////////////////////
    //
    // printfParser
    //
    ///////////////////////////////////////////////////////////////////////////

    final int
    parse(char cfmt[]) {

	return(parse(cfmt, 0));
    }

    ///////////////////////////////////////////////////////////////////////////

    final int
    parse(char cfmt[], int n) {

	boolean	done;
	int	ch;

    //
    // Parse the conversion specification that starts at index n
    // in fmt.  Results are stored in the class variables and the
    // position of the character that stopped the parse is
    // returned to the caller.
    //

	alternate = false;
	rightpad = false;
	sign = false;
	space = false;
	zeropad = false;
	trim = false;

	for (done = false; n < cfmt.length && !done; n++) {
	    switch (cfmt[n]) {
		case '-': rightpad = true; break;
		case '+': sign = true; break;
		case ' ': space = true; break;
		case '0': zeropad = true; break;
		case '#': alternate = true; break;
		default: done = true; n--; break;
	    }
	}

	plus = (sign ? "+" : (space ? " " : ""));

	for (width = 0; n < cfmt.length && GrappaSupport.isdigit(ch = cfmt[n]); n++)
	    width = width*10 + (ch - '0');

	if (n < cfmt.length && cfmt[n] == '.') {
	    n++;
	    for (precision = 0; n < cfmt.length && GrappaSupport.isdigit(ch = cfmt[n]); n++)
		precision = precision*10 + (ch - '0');
	} else precision = -1;

	padding = (zeropad && !rightpad) ? '0' : ' ';

	return(n);
    }

    ///////////////////////////////////////////////////////////////////////////

    final StringBuffer
    buildChar(StringBuffer buf, int arg) {

	scratch.setLength(0);
	scratch.append((char)arg);
	return(strpad(buf, scratch.toString(), ' ', width, rightpad));
    }

    ///////////////////////////////////////////////////////////////////////////

    final StringBuffer
    buildExp(StringBuffer buf, double arg, boolean upper) {

	double		exp;
	double		base;
	double		val;
	int		sign;

	precision = (precision >= 0) ? precision : 6;

	val = arg;
	sign = (val >= 0) ? 1 : -1;
	val = (val < 0) ? -val : val;

	if (val >= 1) {
	    exp = Math.log(val)/LOG10;
	    base = Math.pow(10, exp - (int)exp);
	} else {
	    exp = Math.log(val/10)/LOG10;
	    base = Math.pow(10, exp - (int)exp + 1);
	}

	scratch.setLength(0);
	scratch.append(upper ? "E" : "e");
	scratch.append(exp > 0 ? '+' : '-');

	strpad(scratch, ("" + (int)(exp > 0 ? exp : -exp)), '0', 2, false);
	if (padding == '0' && precision >= 0)
	    padding = ' ';

	return(strpad(buf, doubleToString(sign*base, scratch.toString()), padding, width, rightpad));
    }

    ///////////////////////////////////////////////////////////////////////////

    final StringBuffer
    buildFlex(StringBuffer buf, double arg, boolean upper) {

	double		exp;
	double		val;
	double		ival;
	StringBuffer	retbuf;
	int		iexp;
	int		pr;

	trim = true;

	val = arg;
	ival = (int)arg;
	val = (val < 0) ? -val : val;


	if (val >= 1) {
	    exp = Math.log(val)/LOG10;
	} else {
	    exp = Math.log(val/10)/LOG10;
	}

	iexp = (int)exp;
	precision = (precision >= 0) ? --precision : 5;



	if (val == ival) {
	    if (alternate) {
		if (precision < 0 || iexp <= precision) {
		    precision -= iexp;
		    retbuf = buildFloat(buf, arg);
		} else retbuf = buildExp(buf, arg, upper);
	    } else {
		if (precision < 0 || iexp <= precision) {
		    precision = -1;
		    retbuf = buildInteger(buf, (int)arg);
		} else retbuf = buildExp(buf, arg, upper);
	    }
	} else if (iexp < -4 || iexp > precision)
	    retbuf = buildExp(buf, arg, upper);
	else retbuf = buildFloat(buf, arg);

	return(retbuf);
    }

    ///////////////////////////////////////////////////////////////////////////

    final StringBuffer
    buildPoint(StringBuffer buf, java.awt.geom.Point2D parg, boolean upper) {

	double[]		arg = { 0, 0 };
	double[]		exp = { 0, 0 };
	double[]		val = { 0, 0 };
	double[]		ival = { 0, 0 };
	int[]			iexp = { 0, 0 };
	StringBuffer		retbuf = null;
	int			orig_precision;
	int			pr;


	trim = true;

	arg[0] = parg.getX();
	arg[1] = (Grappa.negateStringYCoord?-parg.getY():parg.getY());
	val[0] = arg[0];
	val[1] = arg[1];
	orig_precision = precision;

	for(int i=0; i<2; i++) {
	    precision = orig_precision;
	    ival[i] = (int)val[i];
	    val[i] = (val[i] < 0) ? -val[i] : val[i];

	    if (val[i] >= 1) {
		exp[i] = Math.log(val[i])/LOG10;
	    } else {
		exp[i] = Math.log(val[i]/10)/LOG10;
	    }

	    iexp[i] = (int)exp[i];
	    precision = (precision >= 0) ? --precision : 5;

	    if (val[i] == ival[i]) {
		if (alternate) {
		    if (precision < 0 || iexp[i] <= precision) {
			precision -= iexp[i];
			retbuf = buildFloat(buf, arg[i]);
		    } else retbuf = buildExp(buf, arg[i], upper);
		} else {
		    if (precision < 0 || iexp[i] <= precision) {
			precision = -1;
			retbuf = buildInteger(buf, (long)arg[i]);
		    } else retbuf = buildExp(buf, arg[i], upper);
		}
	    } else if (iexp[i] < -4 || iexp[i] > precision)
		retbuf = buildExp(buf, arg[i], upper);
	    else retbuf = buildFloat(buf, arg[i]);

	    if(i == 0) {
		retbuf = retbuf.append(',');
		buf = retbuf;
	    }
	}

	return(retbuf);
    }

    ///////////////////////////////////////////////////////////////////////////

    final StringBuffer
    buildSize(StringBuffer buf, java.awt.geom.Dimension2D parg, boolean upper) {

	double[]		arg = { 0, 0 };
	double[]		exp = { 0, 0 };
	double[]		val = { 0, 0 };
	double[]		ival = { 0, 0 };
	int[]			iexp = { 0, 0 };
	StringBuffer		retbuf = null;
	int			orig_precision;
	int			pr;


	trim = true;

	arg[0] = parg.getWidth();
	arg[1] = parg.getHeight();
	val[0] = arg[0];
	val[1] = arg[1];
	orig_precision = precision;

	for(int i=0; i<2; i++) {
	    precision = orig_precision;
	    ival[i] = (int)val[i];
	    val[i] = (val[i] < 0) ? -val[i] : val[i];

	    if (val[i] >= 1) {
		exp[i] = Math.log(val[i])/LOG10;
	    } else {
		exp[i] = Math.log(val[i]/10)/LOG10;
	    }

	    iexp[i] = (int)exp[i];
	    precision = (precision >= 0) ? --precision : 5;

	    if (val[i] == ival[i]) {
		if (alternate) {
		    if (precision < 0 || iexp[i] <= precision) {
			precision -= iexp[i];
			retbuf = buildFloat(buf, arg[i]);
		    } else retbuf = buildExp(buf, arg[i], upper);
		} else {
		    if (precision < 0 || iexp[i] <= precision) {
			precision = -1;
			retbuf = buildInteger(buf, (long)arg[i]);
		    } else retbuf = buildExp(buf, arg[i], upper);
		}
	    } else if (iexp[i] < -4 || iexp[i] > precision)
		retbuf = buildExp(buf, arg[i], upper);
	    else retbuf = buildFloat(buf, arg[i]);

	    if(i == 0) {
		retbuf = retbuf.append(',');
		buf = retbuf;
	    }
	}

	return(retbuf);
    }

    ///////////////////////////////////////////////////////////////////////////

    final StringBuffer
    buildBox(StringBuffer buf, java.awt.geom.Rectangle2D parg, boolean upper, boolean dimensioned) {

	double[]		arg = { 0, 0, 0, 0 };
	double[]		exp = { 0, 0, 0, 0 };
	double[]		val = { 0, 0, 0, 0 };
	double[]		ival = { 0, 0, 0, 0 };
	int[]			iexp = { 0, 0, 0, 0 };
	StringBuffer		retbuf = null;
	int			orig_precision;
	int			pr;


	trim = true;

	if(!dimensioned) {
	    arg[0] = parg.getX();
	    arg[1] = parg.getY();
	    arg[2] = arg[0] + arg[2];
	    arg[3] = arg[1] + arg[3];
	    arg[1] = (Grappa.negateStringYCoord?-arg[1]:arg[1]);
	    arg[3] = (Grappa.negateStringYCoord?-arg[3]:arg[3]);
	} else {
	    arg[0] = parg.getX();
	    arg[1] = (Grappa.negateStringYCoord?-parg.getY():parg.getY());
	    arg[2] = parg.getWidth();
	    arg[3] = parg.getHeight();
	}
	val[0] = arg[0];
	val[1] = arg[1];
	val[2] = arg[2];
	val[3] = arg[3];
	orig_precision = precision;

	for(int i=0; i<4; i++) {
	    precision = orig_precision;
	    ival[i] = (int)val[i];
	    val[i] = (val[i] < 0) ? -val[i] : val[i];

	    if (val[i] >= 1) {
		exp[i] = Math.log(val[i])/LOG10;
	    } else {
		exp[i] = Math.log(val[i]/10)/LOG10;
	    }

	    iexp[i] = (int)exp[i];
	    precision = (precision >= 0) ? --precision : 5;

	    if (val[i] == ival[i]) {
		if (alternate) {
		    if (precision < 0 || iexp[i] <= precision) {
			precision -= iexp[i];
			retbuf = buildFloat(buf, arg[i]);
		    } else retbuf = buildExp(buf, arg[i], upper);
		} else {
		    if (precision < 0 || iexp[i] <= precision) {
			precision = -1;
			retbuf = buildInteger(buf, (long)arg[i]);
		    } else retbuf = buildExp(buf, arg[i], upper);
		}
	    } else if (iexp[i] < -4 || iexp[i] > precision)
		retbuf = buildExp(buf, arg[i], upper);
	    else retbuf = buildFloat(buf, arg[i]);

	    if(i < 3) {
		retbuf = retbuf.append(',');
		buf = retbuf;
	    }
	}

	return(retbuf);
    }

    ///////////////////////////////////////////////////////////////////////////

    final StringBuffer
    buildFloat(StringBuffer buf, double arg) {

	double	val;
	int	sign;

	precision = (precision >= 0) ? precision : 6;
	val = arg;

	if (padding == '0' && precision >= 0)
	    padding = ' ';
	return(strpad(buf, doubleToString(val, ""), padding, width, rightpad));
    }

    ///////////////////////////////////////////////////////////////////////////

    final StringBuffer
    buildHex(StringBuffer buf, int arg, boolean upper) {

	String	str;

	scratch.setLength(0);

	str = (upper) ? Integer.toHexString(arg).toUpperCase() : Integer.toHexString(arg);

	if (precision > str.length()) {
	    if (alternate)
		scratch.append(upper ? "0X" : "0x");
	    strpad(scratch, str, '0', precision, false);
	    strpad(buf, scratch.toString(), ' ', width, rightpad);
	} else {
	    if (zeropad && !rightpad && precision < 0) {
		if (alternate) {
		    if (width > 2) {
			strpad(scratch, str, '0', width-2, rightpad);
			buf.append(upper ? "0X" : "0x");
			buf.append(scratch.toString());
		    } else {
			buf.append(upper ? "0X" : "0x");
			buf.append(str);
		    }
		} else strpad(buf, str, '0', width, rightpad);
	    } else {
		if (alternate) {
		    scratch.append(upper ? "0X" : "0x");
		    scratch.append(str);
		    str = scratch.toString();
		}
		strpad(buf, str, ' ', width, rightpad);
	    }
	}

	return(buf);
    }

    ///////////////////////////////////////////////////////////////////////////

    final StringBuffer
    buildInteger(StringBuffer buf, long arg) {

	String	str;
	String	sign;
	long	val;

	scratch.setLength(0);

	val = arg;
	sign = (val >= 0) ? plus : "-";
	str = "" + ((val < 0) ? -val : val);

	if (precision > str.length()) {
	    strpad(scratch, str, '0', precision, false);
	    scratch.insert(0, sign);
	} else {
	    scratch.append(sign);
	    scratch.append(str);
	}

	if (padding == '0' && precision >= 0)
	    padding = ' ';

	return(strpad(buf, scratch.toString(), padding, width, rightpad));
    }

    ///////////////////////////////////////////////////////////////////////////

    final StringBuffer
    buildOctal(StringBuffer buf, int arg) {

	String	str;

	scratch.setLength(0);

	if (alternate)
	    scratch.append('0');

	scratch.append(Integer.toOctalString(arg));
	if (precision > scratch.length()) {
	    str = scratch.toString();
	    scratch.setLength(0);
	    strpad(scratch, str, '0', precision, false);
	}

	if (padding == '0' && precision >= 0)
	    padding = ' ';

	return(strpad(buf, scratch.toString(), padding, width, rightpad));
    }

    ///////////////////////////////////////////////////////////////////////////

    final StringBuffer
    buildString(StringBuffer buf, String arg) {

	String	str;

	if (precision > 0) {
	    if (precision < arg.length())
		str = arg.substring(0, precision);
	    else str = arg;
	} else str = arg;

	return(strpad(buf, str, padding, width, rightpad));
    }

    ///////////////////////////////////////////////////////////////////////////
    //
    // Private methods
    //
    ///////////////////////////////////////////////////////////////////////////

    private String
    doubleToString(double val, String exp) {

	String	sign;
	double	whole;
	double	power;
	double	frac;

    //
    // Building the resulting String up by casting to an int or long
    // doesn't always work, so we use algorithm that may look harder
    // and slower than necessary.
    //

	scratch.setLength(0);

	sign = (val >= 0) ? plus : "-";
	val = (val < 0) ? -val : val;

	whole = Math.floor(val);

	if (precision != 0) {
	    power = Math.pow(10, precision);
	    frac = (val - whole)*power;
	    scratch.append((long)whole);
	    String tail = (""+((long)Math.round(frac)));
	    if(trim) {
		int len = tail.length();
		int extra = 0;
		while(extra < len && tail.charAt(len-extra-1) == '0') extra++;
		if(extra == len) {
		    if(exp.length() > 0) {
			tail = ".0";
		    } else {
			tail = "";
		    }
		    precision = 0;
		} else if(extra > 0) {
		    scratch.append('.');
		    tail = tail.substring(0,len-extra);
		    precision -= extra;
		} else {
		    scratch.append('.');
		}
	    } else {
		scratch.append('.');
	    }
	    if (precision > 0 && (power/10) > frac) {
		strpad(scratch, tail, '0', precision, false);
	    } else scratch.append(tail);
	    scratch.append(exp);
	} else {
	    scratch.append((long)whole);
	    if (alternate && exp.length() == 0)
		scratch.append('.');
	    scratch.append(exp);
	}

	if (zeropad && !rightpad) {
	    String str = scratch.toString();
	    scratch.setLength(0);
	    strpad(scratch, str, '0', width - sign.length(), false);
	}

	scratch.insert(0, sign);
	return(scratch.toString());
    }

    ///////////////////////////////////////////////////////////////////////////

    private StringBuffer
    strpad(StringBuffer buf, String str, int ch, int width, boolean right) {

	int	len;
	int	n;

	if (width > 0) {
	    if ((len = width - str.length()) > 0) {
		if (right)
		    buf.append(str);
		for (n = 0; n < len; n++)
		    buf.append((char)ch);
		if (!right)
		    buf.append(str);
	    } else buf.append(str);
	} else buf.append(str);

	return(buf);
    }

    ///////////////////////////////////////////////////////////////////////////
}

