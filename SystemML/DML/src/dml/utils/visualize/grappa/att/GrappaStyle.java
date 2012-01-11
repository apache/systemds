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
import java.util.Hashtable;

/**
 * This class translates and encapsulates information provided by the <I>style</I> attribute.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public class GrappaStyle
    implements
	GrappaConstants,
	Cloneable
{
    // element type associated with this style
    private int elementType;

    /**
     * A style attribute with this string value gets set to the default
     * style for the associated element type.
     */
    public final static String DEFAULT_SET_STRING = "__default__";

    /**
     * Integer value for indicating solid line info.
     */
    public final static int STYLE_SOLID         = 0;
    /**
     * Integer value for indicating dashed line info.
     */
    public final static int STYLE_DASHED        = 1;
    /**
     * Integer value for indicating dotted line info.
     */
    public final static int STYLE_DOTTED        = 2;
    /**
     * Integer value for indicating specific dashed line info.
     */
    public final static int STYLE_DASH          = 3;
    /**
     * Integer value for indicating dash phase info for a dashed line.
     */
    public final static int STYLE_DASH_PHASE    = 4;
    /**
     * Integer value for indicating line width info.
     */
    public final static int STYLE_LINE_WIDTH    = 5;
    /**
     * Integer value for indicating line color info.
     */
    public final static int STYLE_LINE_COLOR    = 6;
    /**
     * Integer value for indicating fill info.
     */
    public final static int STYLE_FILLED        = 7;
    /**
     * Integer value for indicating diagonal corner info.
     */
    public final static int STYLE_DIAGONALS     = 8;
    /**
     * Integer value for indicating rounded corner info.
     */
    public final static int STYLE_ROUNDED       = 9;
    /**
     * Integer value for indicating butt cap info.
     */
    public final static int STYLE_CAP_BUTT      = 10;
    /**
     * Integer value for indicating round cap info.
     */
    public final static int STYLE_CAP_ROUND     = 11;
    /**
     * Integer value for indicating square cap info.
     */
    public final static int STYLE_CAP_SQUARE    = 12;
    /**
     * Integer value for indicating bevel join info.
     */
    public final static int STYLE_JOIN_BEVEL    = 13;
    /**
     * Integer value for indicating miter join info.
     */
    public final static int STYLE_JOIN_MITER    = 14;
    /**
     * Integer value for indicating round miter info.
     */
    public final static int STYLE_JOIN_ROUND    = 15;
    /**
     * Integer value for indicating miter limit info.
     */
    public final static int STYLE_MITER_LIMIT   = 16;
    /**
     * Integer value for indicating fixed size info.
     */
    public final static int STYLE_FIXED_SIZE    = 17;
    /**
     * Integer value for indicating fill info.
     */
    public final static int STYLE_INVIS         = 18;

    // for compatibility with dot (Grappa uses fontstyle)
    /**
     * Integer value for indicating bold font info (should use fontstyle)
     */
    public final static int STYLE_OLD_BOLD    = 100;
    /**
     * Integer value for indicating italic font info (should use fontstyle)
     */
    public final static int STYLE_OLD_ITALIC    = 101;
    /**
     * Integer value for indicating plain font info (should use fontstyle)
     */
    public final static int STYLE_OLD_PLAIN    = 102;

    /**
     * Line color default.
     */
    public final static Color STYLE_LINE_COLOR_DEFAULT = GrappaColor.getColor("black",null);
    /**
     * Line style default.
     */
    public final static int STYLE_LINE_STYLE_DEFAULT = STYLE_SOLID;
    /**
     * Line width default.
     */
    public final static float STYLE_LINE_WIDTH_DEFAULT = 1;
    /**
     * Line cap default.
     */
    public final static int STYLE_CAP_DEFAULT = BasicStroke.CAP_BUTT;
    /**
     * Line join default.
     */
    public final static int STYLE_JOIN_DEFAULT = BasicStroke.JOIN_BEVEL;
    /**
     * Line miter default.
     */
    public final static float STYLE_MITER_LIMIT_DEFAULT = 0;
    /**
     * Line dash default.
     */
    public final static float[] STYLE_DASH_DEFAULT = null;
    /**
     * Line dash phase default.
     */
    public final static float STYLE_DASH_PHASE_DEFAULT = 0;
    /**
     * Rounded corner default.
     */
    public final static boolean STYLE_ROUNDED_DEFAULT = false;
    /**
     * Diagonal corner default.
     */
    public final static boolean STYLE_DIAGONALS_DEFAULT = false;
    /**
     * Fill default.
     */
    public final static boolean STYLE_FILLED_DEFAULT = false;
    /**
     * Invisibility default.
     */
    public final static boolean STYLE_INVIS_DEFAULT = false;
    /**
     * Fixed size default.
     */
    public final static boolean STYLE_FIXED_SIZE_DEFAULT = false;

    static BasicStroke defaultStroke = new BasicStroke(STYLE_LINE_WIDTH_DEFAULT,STYLE_CAP_DEFAULT,STYLE_JOIN_DEFAULT,STYLE_MITER_LIMIT_DEFAULT,STYLE_DASH_DEFAULT,STYLE_DASH_PHASE_DEFAULT);

    static String defaultStrokeString = generateStrokeString(STYLE_LINE_WIDTH_DEFAULT,STYLE_CAP_DEFAULT,STYLE_JOIN_DEFAULT,STYLE_MITER_LIMIT_DEFAULT,STYLE_DASH_DEFAULT,STYLE_DASH_PHASE_DEFAULT);

    private static Hashtable styleTypes = new Hashtable(20);

    private static Hashtable strokeCache = new Hashtable(4);

    static {
	styleTypes.put("solid",new Integer(STYLE_SOLID));
	styleTypes.put("dashed",new Integer(STYLE_DASHED));
	styleTypes.put("dotted",new Integer(STYLE_DOTTED));
	styleTypes.put("dash",new Integer(STYLE_DASH));
	styleTypes.put("dashphase",new Integer(STYLE_DASH_PHASE));
	styleTypes.put("dash_phase",new Integer(STYLE_DASH_PHASE));
	styleTypes.put("width",new Integer(STYLE_LINE_WIDTH));
	styleTypes.put("linewidth",new Integer(STYLE_LINE_WIDTH));
	styleTypes.put("line_width",new Integer(STYLE_LINE_WIDTH));
	styleTypes.put("setlinewidth",new Integer(STYLE_LINE_WIDTH));
	styleTypes.put("color",new Integer(STYLE_LINE_COLOR));
	styleTypes.put("linecolor",new Integer(STYLE_LINE_COLOR));
	styleTypes.put("line_color",new Integer(STYLE_LINE_COLOR));
	styleTypes.put("filled",new Integer(STYLE_FILLED));
	styleTypes.put("invis",new Integer(STYLE_INVIS));
	styleTypes.put("diagonals",new Integer(STYLE_DIAGONALS));
	styleTypes.put("rounded",new Integer(STYLE_ROUNDED));
	styleTypes.put("capbutt",new Integer(STYLE_CAP_BUTT));
	styleTypes.put("cap_butt",new Integer(STYLE_CAP_BUTT));
	styleTypes.put("capround",new Integer(STYLE_CAP_ROUND));
	styleTypes.put("cap_round",new Integer(STYLE_CAP_ROUND));
	styleTypes.put("capsquare",new Integer(STYLE_CAP_SQUARE));
	styleTypes.put("cap_square",new Integer(STYLE_CAP_SQUARE));
	styleTypes.put("joinbevel",new Integer(STYLE_JOIN_BEVEL));
	styleTypes.put("join_bevel",new Integer(STYLE_JOIN_BEVEL));
	styleTypes.put("joinmiter",new Integer(STYLE_JOIN_MITER));
	styleTypes.put("join_miter",new Integer(STYLE_JOIN_MITER));
	styleTypes.put("joinround",new Integer(STYLE_JOIN_ROUND));
	styleTypes.put("join_round",new Integer(STYLE_JOIN_ROUND));
	styleTypes.put("miterlimit",new Integer(STYLE_MITER_LIMIT));
	styleTypes.put("miter_limit",new Integer(STYLE_MITER_LIMIT));
	styleTypes.put("fixedsize",new Integer(STYLE_FIXED_SIZE));
	styleTypes.put("fixed_size",new Integer(STYLE_FIXED_SIZE));

	// for compatibility with dot (these should be fontstyle for Grappa)
	styleTypes.put("bold",new Integer(STYLE_OLD_BOLD));
	styleTypes.put("italic",new Integer(STYLE_OLD_ITALIC));
	styleTypes.put("normal",new Integer(STYLE_OLD_PLAIN));
	styleTypes.put("plain",new Integer(STYLE_OLD_PLAIN));

	strokeCache.put(defaultStrokeString, defaultStroke);
    }

    Color line_color = STYLE_LINE_COLOR_DEFAULT;
    int line_style = STYLE_LINE_STYLE_DEFAULT;
    float line_width = STYLE_LINE_WIDTH_DEFAULT;
    int cap = STYLE_CAP_DEFAULT;
    int join = STYLE_JOIN_DEFAULT;
    float miter_limit = STYLE_MITER_LIMIT_DEFAULT;
    float[] dash = STYLE_DASH_DEFAULT;
    float dash_phase = STYLE_DASH_PHASE_DEFAULT;
    boolean rounded = STYLE_ROUNDED_DEFAULT;
    boolean diagonals = STYLE_DIAGONALS_DEFAULT;
    boolean filled = STYLE_FILLED_DEFAULT;
    boolean invis = STYLE_INVIS_DEFAULT;
    boolean fixed_size = STYLE_FIXED_SIZE_DEFAULT;
    Integer font_style = null;

    BasicStroke stroke = defaultStroke;

    ////////////////////////////////////////////////////////////////////////
    //
    // Constructors
    //
    ////////////////////////////////////////////////////////////////////////

    /**
     * Constructs a new <code>GrappaStyle</code> object from a style
     * description string.
     *
     * @param type element type to associate with this style.
     * @param style the <code>String</code> that specifies the style info.
     *              format is: style1,style2(extra2),...,styleN.
     */
    public GrappaStyle(int type, String style) {
	if((type&(Grappa.NODE|Grappa.EDGE|Grappa.SUBGRAPH|Grappa.SYSTEM)) != type) {
	    throw new RuntimeException("type must specify node, edge or subgraph");
	}
	this.elementType = type;
	updateStyle(style);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // Public methods
    //
    ////////////////////////////////////////////////////////////////////////

    /**
     * Update this GrappaStyle based on the supplied style string.
     *
     * @param style a style specification
     */
    public void updateStyle(String style) {
	stroke = defaultStroke;

	line_color = STYLE_LINE_COLOR_DEFAULT;
	line_style = STYLE_LINE_STYLE_DEFAULT;
	line_width = STYLE_LINE_WIDTH_DEFAULT;
	cap = STYLE_CAP_DEFAULT;
	join = STYLE_JOIN_DEFAULT;
	miter_limit = STYLE_MITER_LIMIT_DEFAULT;
	dash = STYLE_DASH_DEFAULT;;
	dash_phase = STYLE_DASH_PHASE_DEFAULT;
	rounded = STYLE_ROUNDED_DEFAULT;
	diagonals = STYLE_DIAGONALS_DEFAULT;
	filled = STYLE_FILLED_DEFAULT;
	invis = STYLE_INVIS_DEFAULT;
	fixed_size = STYLE_FIXED_SIZE_DEFAULT;
	font_style = null;

	if(style == null) return;

	int len = style.length();
	if(len == 0 || style.equals(GrappaStyle.DEFAULT_SET_STRING)) return;

	String option_string = null;
	Object option_obj = null;
	int option = -1;
	int last_option = -1;
	boolean in_parens = false;
	boolean keyword_ok = true;
	boolean parens_ok = false;
	int i = 0, offset = 0;
	char c;
	while(i<len) {
	    c = style.charAt(i++);
	    if(Character.isWhitespace(c)) continue;
	    if(c == '(') {
		if(in_parens) {
		    throw new RuntimeException("style attribute has nested parentheses");
		}
		if(!parens_ok) {
		    throw new RuntimeException("style attribute has parentheses without keyword");
		}
		if(last_option < 0) {
		    throw new RuntimeException("style attribute (" + option_string + ") has unexpected modifier");
		}
		in_parens = true;
		keyword_ok = true;
		parens_ok = false;
	    } else if(c == ')') {
		if(!in_parens) {
		    throw new RuntimeException("style attribute has unmatched right parenthesis");
		}
		in_parens = false;
		keyword_ok = false;
		parens_ok = false;
	    } else if(c == ',') {
		if(in_parens) {
		    throw new RuntimeException("style attribute has comma within parentheses");
		}
		keyword_ok = true;
		parens_ok = false;
	    } else {
		if(!keyword_ok) {
		    throw new RuntimeException("style attribute (" + option_string + ") has bad format");
		}
		offset = i-1;
		if(in_parens) {
		    int pcnt = 0;
		    while(i < len && ((c = style.charAt(i)) != ')' || pcnt > 0)) {
			if(c == '(') pcnt++;
			else if(c == ')') pcnt--;
			i++;
		    }
		} else {
		    while(i < len && (c = style.charAt(i)) != ',' && c != ' ' && c != '(' && c != ')') i++;
		}
		option_string = style.substring(offset,i);
		if(in_parens) {
		    if(last_option == -1) {
			throw new RuntimeException("style attribute modifier (" + option_string + ") is unexpected");
		    }
		    switch(last_option) {
		    case STYLE_DASH:
			line_style = last_option;
			try {
			    dash = GrappaSupport.floatArrayForTuple(option_string);
			} catch(NumberFormatException nfe) {
			    throw new RuntimeException("dash style attribute modifier (" + option_string + ") is not a comma-delimited floating point tuple");
			}
			break;
		    case STYLE_DASH_PHASE:
			try {
			    dash_phase = Float.valueOf(option_string).floatValue();
			} catch(NumberFormatException nfe) {
			    throw new RuntimeException("dash phase style attribute modifier (" + option_string + ") is not a floating point number");
			}
			break;
		    case STYLE_LINE_WIDTH:
			try {
			    line_width = Float.valueOf(option_string).floatValue();
			} catch(NumberFormatException nfe) {
			    throw new RuntimeException("line width style attribute modifier (" + option_string + ") is not a floating point number");
			}
			break;
		    case STYLE_MITER_LIMIT:
			try {
			    miter_limit = Float.valueOf(option_string).floatValue();
			} catch(NumberFormatException nfe) {
			    throw new RuntimeException("miter limit style attribute modifier (" + option_string + ") is not a floating point number");
			}
			break;
		    case STYLE_LINE_COLOR:
			line_color = GrappaColor.getColor(option_string,null);
			break;
		    case STYLE_FILLED:
			filled = Boolean.valueOf(option_string).booleanValue();
			break;
		    case STYLE_INVIS:
			invis = Boolean.valueOf(option_string).booleanValue();
			break;
		    case STYLE_DIAGONALS:
			diagonals = Boolean.valueOf(option_string).booleanValue();
			break;
		    case STYLE_ROUNDED:
			rounded = Boolean.valueOf(option_string).booleanValue();
			break;
		    case STYLE_FIXED_SIZE:
			fixed_size = Boolean.valueOf(option_string).booleanValue();
			break;
		    case STYLE_OLD_BOLD:
		    case STYLE_OLD_ITALIC:
		    case STYLE_OLD_PLAIN:
		    case STYLE_SOLID:
		    case STYLE_DASHED:
		    case STYLE_DOTTED:
		    case STYLE_CAP_BUTT:
		    case STYLE_CAP_ROUND:
		    case STYLE_CAP_SQUARE:
		    case STYLE_JOIN_BEVEL:
		    case STYLE_JOIN_MITER:
		    case STYLE_JOIN_ROUND:
		    default:
			throw new RuntimeException("style attribute (" + option_string + ") has bad format");
		    }
		    last_option = -1;
		} else {
		    option_obj = styleTypes.get(option_string.toLowerCase());
		    if (DEFAULT_SET_STRING.equals(option_string))
			continue;
		    else if(option_obj == null || !(option_obj instanceof Integer) ) {
			throw new RuntimeException("style attribute (" + option_string + ") is unrecognized or badly implemented");
		    }
		    option = ((Integer)option_obj).intValue();

		    last_option = -1;
		    switch(option) {
		    case STYLE_SOLID:
			line_style = option;
			dash = null;
			dash_phase = 0;
			break;
		    case STYLE_DASHED:
			line_style = option;
			dash = new float[] { 12, 12 };
			dash_phase = 0;
			break;
		    case STYLE_DOTTED:
			line_style = option;
			dash = new float[] { 2, 2 };
			dash_phase = 0;
			break;
		    case STYLE_FILLED:
			last_option = option;
			filled = true;
			break;
		    case STYLE_INVIS:
			last_option = option;
			invis = true;
			break;
		    case STYLE_DIAGONALS:
			last_option = option;
			diagonals = true;
			break;
		    case STYLE_ROUNDED:
			last_option = option;
			rounded = true;
			break;
		    case STYLE_CAP_BUTT:
			cap = BasicStroke.CAP_BUTT;
			break;
		    case STYLE_CAP_ROUND:
			cap = BasicStroke.CAP_ROUND;
			break;
		    case STYLE_CAP_SQUARE:
			cap = BasicStroke.CAP_SQUARE;
			break;
		    case STYLE_JOIN_BEVEL:
			cap = BasicStroke.JOIN_BEVEL;
			break;
		    case STYLE_JOIN_MITER:
			cap = BasicStroke.JOIN_MITER;
			break;
		    case STYLE_JOIN_ROUND:
			cap = BasicStroke.JOIN_ROUND;
			break;
		    case STYLE_FIXED_SIZE:
			last_option = option;
			fixed_size = true;
			break;
		    case STYLE_OLD_BOLD:
			font_style = new Integer(Font.BOLD);
			break;
		    case STYLE_OLD_ITALIC:
			font_style = new Integer(Font.ITALIC);
			break;
		    case STYLE_OLD_PLAIN:
			font_style = new Integer(Font.PLAIN);
			break;
		    case STYLE_DASH:
		    case STYLE_DASH_PHASE:
		    case STYLE_LINE_WIDTH:
		    case STYLE_LINE_COLOR:
		    case STYLE_MITER_LIMIT:
			last_option = option;
			break;
		    default:
			throw new RuntimeException("style attribute (" + option_string + ") has bad format");
		    }
		    if(last_option != -1) parens_ok = true;
		}
	    }
	}
	if(in_parens) {
	    throw new RuntimeException("style attribute has unmatched left parenthesis");
	}

	String strokeString = generateStrokeString(line_width,cap,join,miter_limit,dash,dash_phase);
	if((stroke = (BasicStroke)strokeCache.get(strokeString)) == null) {
	    stroke = new BasicStroke(line_width,cap,join,miter_limit,dash,dash_phase);
	    strokeCache.put(strokeString,stroke);
	}
    }

    private static String generateStrokeString(
					       float lineWidth,
					       int capType,
					       int joinType,
					       float miterLimit,
					       float[] dashSpec,
					       float dashPhase
					       ) {

	StringBuffer strokeStringBuffer = new StringBuffer();

	strokeStringBuffer.append(lineWidth);
	strokeStringBuffer.append(",");
	strokeStringBuffer.append(capType);
	strokeStringBuffer.append(",");
	strokeStringBuffer.append(joinType);
	strokeStringBuffer.append(",");
	strokeStringBuffer.append(miterLimit);
	strokeStringBuffer.append(",");
	if(dashSpec == null) {
	    strokeStringBuffer.append("null");
	} else {
	    strokeStringBuffer.append("{");
	    strokeStringBuffer.append(dashSpec[0]);
	    for(int i = 1; i < dashSpec.length; i++) {
		strokeStringBuffer.append(",");
		strokeStringBuffer.append(dashSpec[i]);
	    }
	    strokeStringBuffer.append("}");
	}
	strokeStringBuffer.append(",");
	strokeStringBuffer.append(dashPhase);

	return(strokeStringBuffer.toString());
    }

    /**
     * Provides a string representation of this object consistent 
     * with Grappa attributes.
     *
     * @return attribute-suitable string representation of this GrappaStyle.
     */
    public String toAttributeString() {
	return(generateStyleString(line_color,line_style,line_width,cap,join,miter_limit,dash,dash_phase,rounded,diagonals,filled,invis,fixed_size,font_style,false,elementType));
    }

    /**
     * Provides a generic string representation of this object.
     * 
     * @return a generic string representation of this GrappaStyle. 
     */
    public String toString() {
	return(generateStyleString(line_color,line_style,line_width,cap,join,miter_limit,dash,dash_phase,rounded,diagonals,filled,invis,fixed_size,font_style,true,elementType));
    }

    private static String generateStyleString(
					      Color color,
					      int lineStyle,
					      float lineWidth,
					      int capType,
					      int joinType,
					      float miterLimit,
					      float[] dashSpec,
					      float dashPhase,
					      boolean roundedFlag,
					      boolean diagonalsFlag,
					      boolean filledFlag,
					      boolean invisFlag,
					      boolean fixedSizeFlag,
					      Integer fontStyle,
					      boolean showAll,
					      int type
					      ) {
	StringBuffer styleStringBuffer = null;
	String tmpstr = null;
	Object[] args = { "%g", null };

	if(
	   showAll
	   ||
	   (
	    color != STYLE_LINE_COLOR_DEFAULT
	    &&
	    STYLE_LINE_COLOR_DEFAULT != null
	    &&
	    !STYLE_LINE_COLOR_DEFAULT.equals(color)
	    )
	   ) {
	    if(styleStringBuffer == null) {
		styleStringBuffer = new StringBuffer();
	    } else {
		styleStringBuffer.append(',');
	    }
	    styleStringBuffer.append("lineColor(");
	    if((tmpstr = GrappaColor.getColorName(color)) == null) {
		float[] hsb = Color.RGBtoHSB(color.getRed(),color.getGreen(),color.getBlue(),null);
		//styleStringBuffer.append(hsb[0]);
		//styleStringBuffer.append(',');
		//styleStringBuffer.append(hsb[1]);
		//styleStringBuffer.append(',');
		//styleStringBuffer.append(hsb[2]);
		styleStringBuffer.append(GrappaSupportPrintf.sprintf(new Object[] { "%g,%g,%g", new Float(hsb[0]), new Float(hsb[1]), new Float(hsb[2]) }));
	    } else {
		styleStringBuffer.append(tmpstr);
	    }
	    styleStringBuffer.append(')');
	}

	if(
	   showAll
	   ||
	   lineStyle != STYLE_LINE_STYLE_DEFAULT
	   ) {
	    if(styleStringBuffer == null) {
		styleStringBuffer = new StringBuffer();
	    } else {
		styleStringBuffer.append(',');
	    }
	    switch(lineStyle) {
	    case STYLE_SOLID:
		styleStringBuffer.append("solid");
		break;
	    case STYLE_DASHED:
		styleStringBuffer.append("dashed");
		break;
	    case STYLE_DOTTED:
		styleStringBuffer.append("dotted");
		break;
	    case STYLE_DASH:
		if(dashSpec == null) {
		    styleStringBuffer.append("solid");
		} else {
		    styleStringBuffer.append("dash(");
		    //styleStringBuffer.append(dashSpec[0]);
		    args[1] = new Float(dashSpec[0]);
		    styleStringBuffer.append(GrappaSupportPrintf.sprintf(args));
		    for(int i = 1; i < dashSpec.length; i++) {
			styleStringBuffer.append(',');
			//styleStringBuffer.append(dashSpec[i]);
			args[1] = new Float(dashSpec[i]);
			styleStringBuffer.append(GrappaSupportPrintf.sprintf(args));
		    }
		    styleStringBuffer.append(')');
		}
		break;
	    default:
		throw new InternalError("unexpected lineStyle (" + lineStyle + ")");
	    }
	}

	if(
	   showAll
	   ||
	   lineWidth != STYLE_LINE_WIDTH_DEFAULT
	   ) {
	    if(styleStringBuffer == null) {
		styleStringBuffer = new StringBuffer();
	    } else {
		styleStringBuffer.append(',');
	    }
	    styleStringBuffer.append("lineWidth(");
	    //styleStringBuffer.append(lineWidth);
	    args[1] = new Float(lineWidth);
	    styleStringBuffer.append(GrappaSupportPrintf.sprintf(args));
	    styleStringBuffer.append(')');
	}

	if(
	   showAll
	   ||
	   capType != STYLE_CAP_DEFAULT
	   ) {
	    if(styleStringBuffer == null) {
		styleStringBuffer = new StringBuffer();
	    } else {
		styleStringBuffer.append(',');
	    }
	    switch(capType) {
	    case BasicStroke.CAP_BUTT:
		styleStringBuffer.append("capButt");
		break;
	    case BasicStroke.CAP_ROUND:
		styleStringBuffer.append("capRound");
		break;
	    case BasicStroke.CAP_SQUARE:
		styleStringBuffer.append("capSquare");
		break;
	    default:
		throw new InternalError("unexpected cap type (" + capType + ")");
	    }
	}

	if(
	   showAll
	   ||
	   joinType != STYLE_JOIN_DEFAULT
	   ) {
	    if(styleStringBuffer == null) {
		styleStringBuffer = new StringBuffer();
	    } else {
		styleStringBuffer.append(',');
	    }
	    switch(joinType) {
	    case BasicStroke.JOIN_BEVEL:
		styleStringBuffer.append("joinBevel");
		break;
	    case BasicStroke.JOIN_MITER:
		styleStringBuffer.append("joinMiter");
		break;
	    case BasicStroke.JOIN_ROUND:
		styleStringBuffer.append("joinRound");
		break;
	    default:
		throw new InternalError("unexpected join type (" + joinType + ")");
	    }
	}

	if(
	   showAll
	   ||
	   miterLimit != STYLE_MITER_LIMIT_DEFAULT
	   ) {
	    if(styleStringBuffer == null) {
		styleStringBuffer = new StringBuffer();
	    } else {
		styleStringBuffer.append(',');
	    }
	    styleStringBuffer.append("miterLimit(");
	    //styleStringBuffer.append(miterLimit);
	    args[1] = new Float(miterLimit);
	    styleStringBuffer.append(GrappaSupportPrintf.sprintf(args));
	    styleStringBuffer.append(')');
	}

	if(
	   showAll
	   ||
	   dashPhase != STYLE_DASH_PHASE_DEFAULT
	   ) {
	    if(styleStringBuffer == null) {
		styleStringBuffer = new StringBuffer();
	    } else {
		styleStringBuffer.append(',');
	    }
	    styleStringBuffer.append("dashPhase(");
	    //styleStringBuffer.append(dashPhase);
	    args[1] = new Float(dashPhase);
	    styleStringBuffer.append(GrappaSupportPrintf.sprintf(args));
	    styleStringBuffer.append(')');
	}

	if(
	   (
	    type > 0
	    &&
	    (type&Grappa.NODE) == Grappa.NODE
	    )
	   &&
	   (
	    showAll
	    ||
	    roundedFlag != STYLE_ROUNDED_DEFAULT
	    )
	   ) {
	    if(styleStringBuffer == null) {
		styleStringBuffer = new StringBuffer();
	    } else {
		styleStringBuffer.append(',');
	    }
	    if(roundedFlag) {
		styleStringBuffer.append("rounded");
	    } else {
		styleStringBuffer.append("rounded(false)");
	    }
	}

	if(
	   (
	    type > 0
	    &&
	    (type&Grappa.NODE) == Grappa.NODE
	    )
	   &&
	   (
	    showAll
	    ||
	    diagonalsFlag != STYLE_DIAGONALS_DEFAULT
	    )
	   ) {
	    if(styleStringBuffer == null) {
		styleStringBuffer = new StringBuffer();
	    } else {
		styleStringBuffer.append(',');
	    }
	    if(diagonalsFlag) {
		styleStringBuffer.append("diagonals");
	    } else {
		styleStringBuffer.append("diagonals(false)");
	    }
	}

	if(
	   (
	    type > 0
	    &&
	    (type&(Grappa.NODE|Grappa.SUBGRAPH)) != 0
	    )
	   &&
	   (
	    showAll
	    ||
	    filledFlag != STYLE_FILLED_DEFAULT
	    )
	   ) {
	    if(styleStringBuffer == null) {
		styleStringBuffer = new StringBuffer();
	    } else {
		styleStringBuffer.append(',');
	    }
	    if(filledFlag) {
		styleStringBuffer.append("filled");
	    } else {
		styleStringBuffer.append("filled(false)");
	    }
	}

	if(
	   (
	    type > 0
	    &&
	    (type&(Grappa.NODE|Grappa.EDGE|Grappa.SUBGRAPH)) != 0
	    )
	   &&
	   (
	    showAll
	    ||
	    invisFlag != STYLE_INVIS_DEFAULT
	    )
	   ) {
	    if(styleStringBuffer == null) {
		styleStringBuffer = new StringBuffer();
	    } else {
		styleStringBuffer.append(',');
	    }
	    if(invisFlag) {
		styleStringBuffer.append("invis");
	    } else {
		styleStringBuffer.append("invis(false)");
	    }
	}

	if(
	   showAll
	   ||
	   fixedSizeFlag != STYLE_FIXED_SIZE_DEFAULT
	   ) {
	    if(styleStringBuffer == null) {
		styleStringBuffer = new StringBuffer();
	    } else {
		styleStringBuffer.append(',');
	    }
	    if(fixedSizeFlag) {
		styleStringBuffer.append("fixedSize");
	    } else {
		styleStringBuffer.append("fixedSize(false)");
	    }
	}

	if(
	   fontStyle != null
	   ) {
	    if(styleStringBuffer == null) {
		styleStringBuffer = new StringBuffer();
	    } else {
		styleStringBuffer.append(',');
	    }
	    if(fontStyle.intValue() == Font.BOLD) {
		styleStringBuffer.append("bold");
	    } else if(fontStyle.intValue() == Font.ITALIC) {
		styleStringBuffer.append("italic");
	    } else {
		styleStringBuffer.append("plain");
	    }
	}

	if(styleStringBuffer == null) {
	    tmpstr = ""; // or set it null?
	} else {
	    tmpstr = styleStringBuffer.toString();
	}

	return(tmpstr);
    }

    /**
     * Get the line color.
     *
     * @return the line color.
     */
    public Color getLineColor() {
	return line_color;
    }

    /**
     * Get the line style.
     *
     * @return the line style.
     */
    public int getLineStyle() {
	return line_style;
    }

    /**
     * Get the line width.
     *
     * @return the line width.
     */
    public float getLineWidth() {
	return line_width;
    }

    /**
     * Get the cap style.
     *
     * @return the cap style.
     */
    public int getCapStyle() {
	return cap;
    }

    /**
     * Get the join style.
     *
     * @return the join style.
     */
    public int getJoinStyle() {
	return join;
    }

    /**
     * Get the miter limit.
     *
     * @return the miter limit.
     */
    public float getMiterLimit() {
	return miter_limit;
    }

    /**
     * Get the dash specification.
     *
     * @return the dash specification.
     */
    public float[] getDash() {
	if(dash == null) {
	    return(null);
	}
	return((float[])(dash.clone()));
    }

    /**
     * Get the dash phase.
     *
     * @return the dash phase.
     */
    public float getDashPhase() {
	return dash_phase;
    }

    /**
     * Get the rounded corner specification.
     *
     * @return the rounded corner specification color (true indicates rounded corners).
     */
    public boolean getRounded() {
	return rounded;
    }

    /**
     * Get the diagonal corner specification.
     *
     * @return the diagonal corner specification color (true indicates diagonal corners).
     */
    public boolean getDiagonals() {
	return diagonals;
    }

    /**
     * Get the fill specification.
     *
     * @return the fill specification (true indicates filling should occur).
     */
    public boolean getFilled() {
	return filled;
    }

    /**
     * Get the invisibility specification.
     *
     * @return the invisibility specification (true indicates element is invisible).
     */
    public boolean getInvis() {
	return invis;
    }

    /**
     * Get the fixed size specification.
     *
     * @return the fixed size specification (true indicates that fixed size drawing is requested).
     */
    public boolean getFixedSize() {
	return fixed_size;
    }

    /**
     * Get the font style.
     * Actually, the <I>fontstyle</I> attribute should be used for
     * font style dealings rather than the <I>style</I> attribute, but
     * the latter is permitted for backward compatibility with <I>dot</I>.
     *
     * @return the font style.
     */
    public int getFontStyle() {
	if(font_style == null) return(Font.PLAIN);
	return font_style.intValue();
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // Cloneable interface
    //
    ////////////////////////////////////////////////////////////////////////
 
    /**
     * Creates a new object of the same class as this object.
     *
     * @return     a clone of this instance.
     * @exception  OutOfMemoryError            if there is not enough memory.
     * @see        java.lang.Cloneable
     */
    public Object clone() {
	try {
	    GrappaStyle copy = (GrappaStyle) super.clone();
	    copy.dash = getDash();
	    return copy;
	} catch (CloneNotSupportedException e) {
	    // this shouldn't happen, since we are Cloneable
	    throw new InternalError();
	}
    }
}
