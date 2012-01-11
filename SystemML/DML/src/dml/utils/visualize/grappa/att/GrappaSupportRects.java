
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

import java.awt.*;
import java.awt.geom.*;
import java.util.*;

/**
 * This class provides a method for parsing RECORD_SHAPE node
 * labels and deriving the RECT_ATTR information from it. It
 * is called by the GrappaNexus class.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="h
ttp://www.research.att.com">Research @ AT&T Labs</a>
 * @see Graph
 */
public class GrappaSupportRects
    implements GrappaConstants
{
    /**
     * Rough font sizing information for the roman (or serif) font.
     */
    final static double[] romanFontwidth = {                  // +------+
	0.2500,  0.3330,  0.4080,  0.5000,  0.5000,  0.8330,  // | !"#$%|
	0.7780,  0.3330,  0.3330,  0.3330,  0.5000,  0.5640,  // |&'()*+|
	0.2500,  0.3330,  0.2500,  0.2780,  0.5000,  0.5000,  // |,-./01|
	0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  0.5000,  // |234567|
	0.5000,  0.5000,  0.2780,  0.2780,  0.5640,  0.5640,  // |89:;<=|
	0.5640,  0.4440,  0.9210,  0.7220,  0.6670,  0.6670,  // |>?@ABC|
	0.7220,  0.6110,  0.5560,  0.7220,  0.5560,  0.3330,  // |DEFGHI|
	0.3890,  0.7220,  0.6110,  0.8890,  0.7220,  0.7220,  // |JKLMNO|
	0.5560,  0.7220,  0.6670,  0.5560,  0.6110,  0.7220,  // |PQRSTU|
	0.7220,  0.9440,  0.7220,  0.7220,  0.6110,  0.3330,  // |VWXYZ[|
	0.2780,  0.3330,  0.4690,  0.5000,  0.3330,  0.4440,  // |\]^_`a|
	0.5000,  0.4440,  0.5000,  0.4440,  0.3330,  0.5000,  // |bcdefg|
	0.3330,  0.2780,  0.2780,  0.5000,  0.2780,  0.7780,  // |hijklm|
	0.5000,  0.5000,  0.5000,  0.5000,  0.3330,  0.3890,  // |nopqrs|
	0.2780,  0.5000,  0.5000,  0.7220,  0.5000,  0.5000,  // |tuvwxy|
	0.4440,  0.4800,  0.2000,  0.4800,  0.5410,  0.0      // |z{|}~/
    };                                                        // +-----+

    /**
     * Rough font sizing information for the helvetica (or sansserif) font.
     */
    final static double[] helveticaFontwidth = {              // +------+
	0.2780,  0.2780,  0.3550,  0.5560,  0.5560,  0.8890,  // | !"#$%|
	0.6670,  0.2210,  0.3330,  0.3330,  0.3890,  0.5840,  // |&'()*+|
	0.2780,  0.3330,  0.2780,  0.2780,  0.5560,  0.5560,  // |,-./01|
	0.5560,  0.5560,  0.5560,  0.5560,  0.5560,  0.5560,  // |234567|
	0.5560,  0.5560,  0.2780,  0.2780,  0.5840,  0.5840,  // |89:;<=|
	0.5840,  0.5560,  01.015,  0.6670,  0.6670,  0.7220,  // |>?@ABC|
	0.7220,  0.6670,  0.6110,  0.7780,  0.6110,  0.2780,  // |DEFGHI|
	0.5000,  0.6670,  0.5560,  0.8330,  0.7220,  0.7780,  // |JKLMNO|
	0.6670,  0.7780,  0.7220,  0.6670,  0.6110,  0.7220,  // |PQRSTU|
	0.6670,  0.9440,  0.6670,  0.6670,  0.6110,  0.2780,  // |VWXYZ[|
	0.2780,  0.2780,  0.4690,  0.5560,  0.2220,  0.5560,  // |\]^_`a|
	0.5560,  0.5000,  0.5560,  0.5560,  0.2780,  0.5560,  // |bcdefg|
	0.2780,  0.2220,  0.2220,  0.5000,  0.2220,  0.8330,  // |hijklm|
	0.5560,  0.5560,  0.5560,  0.5560,  0.3330,  0.5000,  // |nopqrs|
	0.2780,  0.5560,  0.5000,  0.7220,  0.5000,  0.5000,  // |tuvwxy|
	0.5000,  0.3340,  0.2600,  0.3340,  0.5840,  0.0   ,  // |z{|}~/
    };                                                        // +-----+

    /**
     * Rough font sizing information for the courier (or constant) font.
     */
    final static double constantFontwidth = 0.6206;

    final static int HASTEXT  = 1;
    final static int HASPORT  = 2;
    final static int HASTABLE = 4;
    final static int INTEXT   = 8;
    final static int INPORT   = 16;

    final static char NBSP = '\u00a0'; // Unicode no-break space

    private static char[] parseArray = null;
    private static int arrayOffset = 0;
    private static int fields = 0;
    private static StringBuffer rbuf = null;

    // assumes shape type is RECORD_SHAPE or MRECORD_SHAPE
    protected static synchronized Object[] parseRecordInfo(Node node) {
	Object[] objs = { null, null, null };

	if(node == null) {
	    return objs;
	}

	String label = (String)node.getAttributeValue(LABEL_ATTR);

	if(label != null && label.equals("\\N")) {
	    label = node.getName();
	}

	if(label == null || label.length() == 0 || label.indexOf('|') < 0) {
	    node.setAttribute(RECTS_ATTR, null);
	    return objs;
	}

	parseArray = label.toCharArray();
	arrayOffset = 0;
	TableField tableField = doParse(node, !node.getSubgraph().isLR(), true);

	if(tableField == null) {
	    node.setAttribute(RECTS_ATTR, null);
	    return objs;
	}

	tableField.sizeFields();
	double width = ((Double)node.getAttributeValue(WIDTH_ATTR)).doubleValue() * PointsPerInch;
	double height = ((Double)node.getAttributeValue(HEIGHT_ATTR)).doubleValue() * PointsPerInch;
	Dimension sz = new Dimension((int)Math.round(width),(int)Math.round(height));
	tableField.resizeFields(sz);

	GrappaPoint pos = (GrappaPoint)node.getAttributeValue(POS_ATTR);
	tableField.positionFields(new Point((int)Math.round(pos.getX()-width/2.0),(int)Math.round(pos.getY()-height/2.0)));

	objs[0] = new String[fields];
	objs[1] = new GrappaPoint[fields];
	fields = 0;
	if(emitFields(tableField,objs)) {
	    objs[2] = rbuf.toString();
	    node.setAttribute(RECTS_ATTR, ((String)objs[2]));
	} else {
	    objs = null;
	}
	rbuf = null;


	//for(int i = 0; i < fields; i++) {
	    //pos = ((GrappaPoint[])objs[1])[i];
	    //pos.setLocation(pos.getX() - width, pos.getY() - height);
	//}

	return objs;
    }

    private static boolean emitFields(TableField tf, Object[] objs) {
	boolean retval = false;

	if(tf == null) return false;

	int fc = tf.fieldCount();

	if(fc == 0) {
	    Rectangle rect = tf.getBounds();
	    if(rbuf == null) {
		rbuf = new StringBuffer();
	    } else {
		rbuf.append(' ');
	    }
	    rbuf.append(rect.x);
	    rbuf.append(',');
	    rbuf.append(Grappa.negateStringYCoord?-rect.y:rect.y);
	    rbuf.append(',');
	    rbuf.append(rect.x+rect.width);
	    rbuf.append(',');
	    rbuf.append(Grappa.negateStringYCoord?(-rect.y-rect.height):(rect.y+rect.height));;

	    ((String[])objs[0])[fields] = tf.getText();
	    ((GrappaPoint[])objs[1])[fields] = new GrappaPoint(rect.getCenterX(), rect.getCenterY());
	    fields++;
	    return true;
	}
	for(int cnt = 0; cnt < fc; cnt++) {
	    if(emitFields(tf.fieldAt(cnt),objs))
		retval = true;
	}
	return(retval);
    }
  
    private static TableField doParse(Node node, boolean LR, boolean topLevel) {
	int maxf = 1;
	int cnt = 0;
	for(int pos = arrayOffset; pos < parseArray.length; pos++) {
	    if(parseArray[pos] == '\\') {
		pos++;
		if(pos < parseArray.length && (parseArray[pos] == '{' || parseArray[pos] == '}' || parseArray[pos] == '|')) {
		    continue;
		}
	    }
	    if(parseArray[pos] == '{') {
		cnt++;
	    } else if(parseArray[pos] == '}') {
		cnt--;
	    } else if(cnt == 0 && parseArray[pos] == '|') {
		maxf++;
	    }
	    if(cnt < 0) {
		break;
	    }
	}

	TableField rv = new TableField();
	rv.setLR(LR);
	rv.subfields(maxf);
	if(topLevel) {
	    rv.setParent(null);
	}

	StringBuffer textBuf, portBuf;
	textBuf = new StringBuffer();
	portBuf = new StringBuffer();
    
	int mode = 0;
	int fi = 0;
	boolean wflag = true;
	TableField tf = null;
	char curCh = '\000';
	while(wflag) {
	    if(arrayOffset >= parseArray.length) {
		curCh = '\000';
		wflag = false;
	    } else {
		curCh = parseArray[arrayOffset];
	    }
	    switch((int)curCh) {
	    case '<':
		if((mode & (HASTABLE|HASPORT)) != 0) {
		    return null;
		}
		mode |= (HASPORT|INPORT);
		arrayOffset++;
		break;
	    case '>':
		if((mode & INPORT) == 0) {
		    return null;
		}
		mode &= ~INPORT;
		arrayOffset++;
		break;
	    case '{':
		arrayOffset++;
		if(mode != 0 || arrayOffset >= parseArray.length) {
		    return null;
		}
		mode = HASTABLE;
		if((tf = doParse(node,!LR,false)) == null) {
		    return null;
		} else {
		    rv.addField(tf);
		    tf.setParent(rv);
		}
		break;
	    case '}':
	    case '|':
	    case '\000':
		if((arrayOffset >= parseArray.length && !topLevel) || (mode&INPORT) != 0) {
		    return null;
		}
		if((mode&HASTABLE) == 0) {
		    tf = new TableField();
		    rv.addField(tf);
		    tf.setLR(!LR);
		    tf.setParent(rv);
		    if((mode&HASPORT) != 0) {
			tf.setId(portBuf.toString().trim());
			portBuf.setLength(0);
		    }
		}
		if((mode&(HASTEXT|HASTABLE)) == 0) {
		    mode |= HASTEXT;
		    textBuf.append(' ');
		}
		if((mode&HASTEXT) != 0) {
		    tf.setTextBounds(textBuf.toString().trim(),node);
		    fields++;
		    //tf.setLR(true);
		    textBuf.setLength(0);
		}
		if(arrayOffset < parseArray.length) {
		    if(curCh == '}') {
			arrayOffset++;
			return rv;
		    }
		    mode = 0;
		    arrayOffset++;
		}
		break;
	    case '\\':
		if(arrayOffset+1 < parseArray.length) {
		    if(isSpec(parseArray[arrayOffset+1])) {
			arrayOffset++;
			curCh = parseArray[arrayOffset];
		    } else if(parseArray[arrayOffset+1] == ' ') {
			arrayOffset++;
			curCh = NBSP;
		    }
		}
		// fall through...
	    default:
		if((mode&HASTABLE) != 0 && curCh != ' ' && curCh != NBSP) {
		    return null;
		}
		if((mode&(INTEXT|INPORT)) == 0 && curCh != ' ' && curCh != NBSP) {
		    mode |= (INTEXT|HASTEXT);
		}
		if((mode&INTEXT) != 0) {
		    textBuf.append(curCh);
		} else if((mode&INPORT) != 0) {
		    portBuf.append(curCh);
		}
		arrayOffset++;
		break;
	    }
	}
	return rv;
    }

    private static boolean isSpec(char c) {
	return ((c) == '{' || (c) == '}' || (c) == '|' || (c) == '<' || (c) == '>');
    }
}

class TableField
    implements GrappaConstants
{
    private Dimension size = new Dimension();
    private Rectangle bounds = new Rectangle();
    private Rectangle textBounds = null;
    private TableField[] subFields = null;
    private int subFieldsUsed = 0;
    private boolean orientLR = false;
    private String idTag = null;
    private String text = null;
    private TableField parent = null;

    /**
     * Creates an empty <code>TableField</code> instance.
     */
    TableField() {
	//super();
    }

    void setParent(TableField prnt) {
	parent = prnt;
    }

    TableField getTopMost() {
	TableField topper = this;
	while(topper.getParent() != null)
	    topper = topper.getParent();
	return topper;
    }

    TableField getParent() {
	return parent;
    }
     
    String getText() {
	return text;
    }
    
    String getIdentifier() {
	StringBuffer buf = new StringBuffer();
	if(isLR()) {
	    buf.append("LR:");
	} else {
	    buf.append("TB:");
	}
	buf.append(fieldCount());
	buf.append('(');
	buf.append(text);
	buf.append(')');
	TableField prnt = getParent();
	while(prnt != null) {
	    buf.append(',');
	    buf.append(prnt.fieldCount());
	    buf.append('(');
	    buf.append(prnt.getText());
	    buf.append(')');
	    prnt = prnt.getParent();
	}
	return buf.toString();
    }

    /**
     * Get the bounding box of this element
     *
     * @return the bounding box of this element
     */
    Rectangle getBounds() {
	return bounds;
    }

    void setBounds(int x, int y, int width, int height) {
	bounds.setBounds(x,y,width,height);
    }

    void setBounds(Rectangle r) {
	bounds.setBounds(r);
    }

    /**
     * Get the size of this object.
     *
     * @return the size of this object.
     */
    Dimension getSize() {
	return size;
    }

    void setSize(int width, int height) {
	size.setSize(width,height);
    }

    void setSize(Dimension d) {
	size.setSize(d.width,d.height);
    }

    boolean hasFields() {
	if(subFields == null || subFields.length == 0 || subFieldsUsed == 0) {
	    return false;
	}
	return true;
    }

    synchronized int subfields(int len) {
	if(len < 1) return 0;
	subFields = new TableField[len];
	return subFields.length;
    }

    int fieldCount() {
	if(subFields == null) {
	    return 0;
	}
	return subFieldsUsed;
    }

    synchronized void addField(TableField tf) {
	// can cause exception
	subFields[subFieldsUsed++] = tf;
    }

    TableField fieldAt(int nbr) {
	if(nbr < 0 || nbr >= subFieldsUsed) return null;
	return subFields[nbr];
    }

    boolean isLR() {
	return orientLR;
    }

    void setLR(boolean lr) {
	orientLR = lr;
    }

    String getId() {
	return idTag;
    }

    void setId(String id) {
	idTag = null;
	if(id == null) return;
	char[] array = id.toCharArray();
	boolean hadNBSP = false;
	for(int i = 0; i < array.length; i++) {
	    if(array[i] == GrappaSupportRects.NBSP) {
		array[i] = ' ';
		hadNBSP = true;
	    }
	}
	if(hadNBSP) idTag = new String(array,0,array.length);
	else idTag = id;
    }

    Dimension sizeFields() {
	return sizeUpFields(this);
    }
  
    private Dimension sizeUpFields(TableField tf) {
	//System.err.println(tf.getIdentifier());
	int fc = tf.fieldCount();
	if(fc == 0) {
	    if(tf.getTextBounds() != null) {
		tf.setSize(tf.getTextBounds().getSize());
	    } else {
		tf.setSize(0,0);
	    }
	} else {
	    Dimension dtmp = null;
	    Dimension dim = new Dimension();
	    for(int cnt = 0; cnt < fc; cnt++) {
		dtmp = sizeUpFields(tf.fieldAt(cnt));
		if(tf.isLR()) {
		    dim.width += dtmp.width;
		    dim.height = (dim.height > dtmp.height) ? dim.height : dtmp.height;
		} else {
		    dim.width = (dim.width > dtmp.width) ? dim.width : dtmp.width;
		    dim.height += dtmp.height;
		}
	    }
	    tf.setSize(dim);
	}
	//System.err.println("Size:"+tf.getSize()+";"+tf.getIdentifier());
	return tf.getSize();
    }

    Dimension resizeFields(Dimension sz) {
	resizeUpFields(this,sz);
	return this.getSize();
    }
  
    void resizeUpFields(TableField tf, Dimension sz) {
	//System.err.println(tf.getIdentifier());

	Dimension delta = new Dimension(sz.width - tf.getSize().width, sz.height - tf.getSize().height);
	tf.setSize(sz);

	int fc = tf.fieldCount();
    
	if(fc == 0) {
	    //System.err.println("Size:"+tf.getSize()+";"+tf.getIdentifier());
	    return;
	}

	// adjust children, if any
	double incr = 0;
	if(tf.isLR()) {
	    incr = (double)delta.width / (double)fc;
	} else {
	    incr = (double)delta.height / (double)fc;
	}
	TableField tfield = null;
	int amt = 0;
	// reuse old space under new name for readability
	Dimension newSz = delta;
	for(int cnt = 0; cnt < fc; cnt++) {
	    tfield = tf.fieldAt(cnt);
	    amt = (int)Math.floor(((double)(cnt+1))*incr) - (int)Math.floor(((double)cnt)*incr);
	    if(tf.isLR()) {
		newSz.setSize(tfield.getSize().width+amt,sz.height);
	    } else {
		newSz.setSize(sz.width,tfield.getSize().height+amt);
	    }
	    resizeUpFields(tfield,newSz);
	}
    }

    void positionFields(Point pos) {
	posFields(this,pos);
    }

    private void posFields(TableField tf, Point pos) {
	tf.setBounds(pos.x,pos.y,tf.getSize().width,tf.getSize().height);
	// clip spillage outside outer-most bounds
	Rectangle b1 = tf.getBounds();
	Rectangle b2 = tf.getTopMost().getBounds();
	int tmpi;
	tmpi = Math.max(b1.x,b2.x);
	b1.width = Math.min(b1.x+b1.width,b2.x+b2.width) - tmpi;
	b1.x = tmpi;
	tmpi = Math.max(b1.y,b2.y);
	b1.height = Math.min(b1.y+b1.height,b2.y+b2.height) - tmpi;
	b1.y = tmpi;

	int fc = tf.fieldCount();

	if(fc == 0) {
	    return;
	}

	TableField tfield = null;
	for(int cnt = 0; cnt < fc; cnt++) {
	    tfield = tf.fieldAt(cnt);
	    posFields(tfield,new Point(pos));
	    if(tf.isLR()) {
		pos.x += tfield.getSize().width;
	    } else {
		pos.y += tfield.getSize().height;
	    }
	}
    }

    void setTextBounds(String str, Node node) {
	int lines = 1;
	boolean cwFont = false;
	double[] fontwidth = { GrappaSupportRects.constantFontwidth };
	    
	String fontname = node.getAttribute(FONTNAME_ATTR).getStringValue().toLowerCase();
	if(fontname.startsWith("courier") || fontname.startsWith("monospaced")) {
	    cwFont = true;
	} else if(fontname.startsWith("helvetica") || fontname.startsWith("sansserif")) {
	    fontwidth = GrappaSupportRects.helveticaFontwidth;
	} else {
	    fontwidth = GrappaSupportRects.romanFontwidth;
	}

	char[] array = str.toCharArray();
	double fwidth = 0;
	double xwidth = 0;
	int value = 0;
	for(int i = 0; i < array.length; i++) {
	    if(array[i] == GrappaSupportRects.NBSP) {
		array[i] = ' ';
	    }
	    if(array[i] == '\\' && (i+1) < array.length) {
		if(array[i+1] == 'n' || array[i+1] == 'l' || array[i+1] == 'r') {
		    lines++;
		    i++;
		    if(fwidth > xwidth)
			xwidth = fwidth;
		    fwidth = 0;
		    continue;
		}
	    }
	    value = array[i] - 32;
	    fwidth += (cwFont) ? fontwidth[0] : ((value >= 0 && value < fontwidth.length) ? fontwidth[value] : 0 );
	}
	if(fwidth > xwidth)
	    xwidth = fwidth;
	int height = ((Integer)node.getAttributeValue(FONTSIZE_ATTR)).intValue();
	int width = (int)Math.round((double)height * xwidth);
	textBounds = new Rectangle(0,0,width,height*lines);
	text = str;
    }

    Rectangle getTextBounds() {
	return(textBounds);
    }

    void debugID() {
	int fc = fieldCount();

	if(fc == 0) {
	    return;
	}

	TableField tfield = null;
	for(int cnt = 0; cnt < fc; cnt++) {
	    tfield = fieldAt(cnt);
	    tfield.debugID();
	}
    }

}

