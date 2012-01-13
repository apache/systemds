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

import java.util.*;

/**
 * A class used for representing attributes associated with the graph elements.
 * An attribute consists of a name-value pair and an element type.
 * Once an attribute is constructed, the name cannot be changed.
 * The element type and the attribute name are used in determining how a
 * string representation of an attribute value is to be converted to an
 * Object and vice versa. The Element class method setUserAttributeType allows
 * users to take advantage of Grappa's built-in converters or to pass a
 * (negative) integer as a conversion indicator to a user-supplied
 * AttributeHandler.
 *
 * <TABLE>
 * <TR>
 * <TD COLSPAN=2>
 * Grappa String Converters
 * </TD>
 * </TR>
 * <TR><TD>BOX_TYPE</TD><TD>dml.hops.visualize.grappa.att.GrappaBox</TD></TR>
 * <TR><TD>COLOR_TYPE</TD><TD>java.awt.Color</TD></TR>
 * <TR><TD>DIR_TYPE</TD><TD>java.lang.Integer (restricted)</TD></TR>
 * <TR><TD>DOUBLE_TYPE</TD><TD>java.lang.Double</TD></TR>
 * <TR><TD>FONTSTYLE_TYPE</TD><TD>java.lang.Integer (restricted)</TD></TR>
 * <TR><TD>HASHLIST_TYPE</TD><TD>java.lang.Hashtable</TD></TR>
 * <TR><TD>INTEGER_TYPE</TD><TD>java.lang.Integer</TD></TR>
 * <TR><TD>LINE_TYPE</TD><TD>dml.hops.visualize.grappa.att.GrappaLine</TD></TR>
 * <TR><TD>POINT_TYPE</TD><TD>dml.hops.visualize.grappa.att.GrappaPoint</TD></TR>
 * <TR><TD>SHAPE_TYPE</TD><TD>java.lang.Integer (restricted)</TD></TR>
 * <TR><TD>SIZE_TYPE</TD><TD>dml.hops.visualize.grappa.att.GrappaSize</TD></TR>
 * <TR><TD>STRING_TYPE</TD><TD>java.lang.String (default)</TD></TR>
 * <TR><TD>STYLE_TYPE</TD><TD>dml.hops.visualize.grappa.att.GrappaStyle</TD></TR>
 * </TABLE>
 *
 * @see AttributeHandler
 * @see Element#setUserAttributeType
 *
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public final class Attribute extends java.util.Observable
    implements
	dml.utils.visualize.grappa.att.AttributeHandler,
	dml.utils.visualize.grappa.att.GrappaConstants
{
    // the custom attribute handler
    private static AttributeHandler specialHandler = null;

    /**
     * Set a custom attribute handler for converting a String value
     * to an Object and vice versa.
     *
     * @param newHandler the AttributeHandler to use for conversions
     *
     * @return the previously set AttributeHandler or null
     *
     * @see AttributeHandler
     */
    public static AttributeHandler setAttributeHandler(AttributeHandler newHandler) {
	AttributeHandler oldHandler = specialHandler;
	specialHandler = newHandler;
	return oldHandler;
    }

    // attribute name
    private String name;
    // attribute value as a string
    private String stringValue;
    // attribute string value converted to an object based on its type
    private Object value;

    // the element type associated with this attribute (includes system)
    private int elementType;
    // the attribute type for conversion to/from a string
    private int attributeType;

    // the hash value of the attribute name
    private int nameHash;

    /**
     * Constructs a new attribute from a name / value pair.
     *
     * @param elemType the element type with which the attribute is 
     *                 or will be associated.
     * @param attrName the name of the attribute.
     * @param attrValue the value of the attribute.
     *
     * @see GrappaConstants#NODE
     * @see GrappaConstants#EDGE
     * @see GrappaConstants#SUBGRAPH
     */
    public Attribute(int elemType, String attrName, Object attrValue) {
	super();
	if(attrName == null) {
	    throw new IllegalArgumentException("the name of an Attribute pair cannot be null");
	}
	attributeType = attributeType(elemType, attrName);
	elementType = elemType;
	name = attrName;
	nameHash = name.hashCode();
	setValue(attrValue);
    }

    /**
     * Constructs a new attribute from an existing one.
     *
     * @param attr the attribute from which this new one is to be generated
     */
    public Attribute(Attribute attr) {
	this(attr.getElementType(),attr.getName(),attr.getValue());
    }

    /**
     * Get the element type for this attribute.
     *
     * @return the element type for this attribute
     *
     * @see GrappaConstants#NODE
     * @see GrappaConstants#EDGE
     * @see GrappaConstants#SUBGRAPH
     */
    public final int getElementType() {
	return elementType;
    }

    /**
     * Get the attribute value type for this attribute.
     *
     * @return the attribute value type for this attribute.
     */
    public final int getAttributeType() {
	return attributeType;
    }

    /**
     * Get the name of this attribute.
     *
     * @return the name of this attribute.
     */
    public final String getName() {
	return name;
    }

    /**
     * Get the value of this attribute.
     *
     * @return the value of the attribute.
     */
    public final Object getValue() {
	if(value == null && stringValue != null) {
	    value = convertStringValue(elementType,name,stringValue,attributeType);
	}
	return value;
    }

    /**
     * Get the value of this attribute converted to a String.
     *
     * @return the value of the attribute as a String.
     */
    public final String getStringValue() {
	switch(attributeType) {
	    // put the types here that users might change on their own
	    // after doing a getValue() so that we always recompute
	    // the string value when it is requested
	case HASHLIST_TYPE:
	    stringValue = null;
	    break;
	}
	if(stringValue == null && value != null) {
	    stringValue = convertValue(elementType,name,value,attributeType);
	}
	return stringValue;
    }

    /**
     * Set the value of the attribute.  If the value is different than the
     * current value, the Observable changed indicator is set.
     *
     * @param attrValue the new attribute value.
     * @return the old attribute value.
     */
    public final Object setValue(Object attrValue) {
	boolean changed = false;
	boolean isString = false;
	Object oldValue = null;
	if(attrValue != null && attrValue instanceof String) {
	    isString = true;
	    oldValue = getStringValue();
	    attrValue = ((String)attrValue).trim();
	} else {
	    oldValue = getValue();
	}
	// note: since we have called either getValue() or getStringValue(),
	//       both value and stringValue are up-to-date
	if (attrValue != null) {
	    if(isString) {
		if(changed = (stringValue == null || !attrValue.equals(stringValue))) {
		    stringValue = (String)attrValue;
		    value = null;
		}
	    } else {
		if(changed = (value == null || !attrValue.equals(value))) {
		    value = copyValue(elementType, name, attrValue,attributeType);
		    stringValue = null;
		}
	    }
	} else {
	    if(changed = (value != null)) {
		value = null;
		stringValue = null;
	    }
	}
	if(changed) {
	    setChanged();
	}
	return oldValue;
    }

    /**
     * Tests for equality with the given attribute.
     *
     * @param attr the attribute with which to compare this attribute.
     * @return true if the two attributes are equal, false otherwise.
     */
    public final boolean equals(Attribute attr) {
	if(attr == null) {
	    return false;
	}
	if(this == attr) {
	    return true;
	}
	if(nameHash != attr.getNameHash() || !attr.getName().equals(name)) {
	    return false;
	}
	String attrValue = attr.getStringValue();
	if(attrValue == getStringValue()) {
	    return true;
	}
	if(attrValue == null) {
	    return false;
	}
	// note: since getStringValue() was called, stringValue is up-to-date
	return attrValue.equals(stringValue);
    }

    /**
     * Tests for equality of this attribute's value with the given attribute's
     * value. The attribute names are not compated.
     *
     * @param attr the attribute with which to compare this attribute.
     * @return true if the two attribute values are equal, false otherwise.
     */
    public final boolean equalsValue(Attribute attr) {
	if(attr == null) {
	    return false;
	}
	if(this == attr) {
	    return true;
	}
	String attrValue = attr.getStringValue();
	if(attrValue == getStringValue()) {
	    return true;
	}
	if(attrValue == null) {
	    return false;
	}
	// note: since getStringValue() was called, stringValue is up-to-date
	return attrValue.equals(stringValue);
    }

    /**
     * Get the hash value for this attributes name.
     *
     * @return the hash code for the name portion of this attribute
     */
    public final int getNameHash() {
	return nameHash;
    }

    /**
     * Use to indicate that this object has changed. 
     * This method is a convenience method that calls the corresponding
     * protected method of the Observable class.
     * 
     * @see java.util.Observable#setChanged()
     */
    public final void setChanged() {
	super.setChanged();
    }

    /**
     * Use to indicate that this object has no longer changed, or that it has
     * already notified all of its observers of its most recent change.
     * This method is a convenience method that calls the corresponding
     * protected method of the Observable class.
     * 
     * @see java.util.Observable#clearChanged()
     */
    public final void clearChanged() {
	super.clearChanged();
    }

    /**
     * Provide a generic string representation of the attribute.
     */
    public String toString() {
	return getClass().getName() + "[name=\""+name+"\",value=\""+getStringValue()+"\"]";
    }

    /**
     * Convert the supplied value to a string. How to convert the value is
     * based on the type, name and attrtype information supplied. Note: this
     * method really could be declared static except that it hides the
     * instance method declared in the AttributeHandler interface and a
     * class method cannot hide an instance method.
     *
     * @param type the element type to which the named attribute applies
     * @param name the name of the attribute
     * @param value the object value to be converted to a string
     * @param attrtype the type of the attribute
     * @return a string representation of the supplied value
     */
    public String convertValue(int type, String name, Object value, int attrtype) {
	String stringValue = null;

	switch(attrtype) {
	case BOX_TYPE:
	    if(value instanceof GrappaBox) {
		stringValue = ((GrappaBox)value).toAttributeString();
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of GrappaBox");
	    }
	    break;
	case COLOR_TYPE:
	    if(value instanceof java.awt.Color) {
		stringValue = GrappaColor.getColorName((java.awt.Color)value);
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of Color");
	    }
	    break;
	case DIR_TYPE:
	    if(value instanceof Integer) {
		stringValue = GrappaSupport.xlateDir(((Integer)value).intValue());
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of Integer");
	    }
	    break;
	case DOUBLE_TYPE:
	    if(value instanceof Double) {
	      stringValue = GrappaSupportPrintf.sprintf(new Object[] { "%g", value });
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of Double");
	    }
	    break;
	case FONTSTYLE_TYPE:
	    if(value instanceof Integer) {
		stringValue = GrappaSupport.xlateFontStyle(((Integer)value).intValue());
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of Integer");
	    }
	    break;
	case HASHLIST_TYPE:
	    if(value instanceof Hashtable) {
		StringBuffer strbuf = new StringBuffer();
		Enumeration keys = ((Hashtable)value).keys();
		synchronized(strbuf) {
		    while(keys.hasMoreElements()) {
			if(strbuf.length() > 0)
			    strbuf.append(',');
			strbuf.append((String)(keys.nextElement()));
		    }
		    stringValue = strbuf.toString();
		}
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of Hashtable");
	    }
	    break;
	case INTEGER_TYPE:
	    if(value instanceof Integer) {
		stringValue = ((Integer)value).toString();
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of Integer");
	    }
	    break;
	case LINE_TYPE:
	    if(value instanceof GrappaLine) {
		stringValue = ((GrappaLine)value).toAttributeString();
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of GrappaLine");
	    }
	    break;
	case POINT_TYPE:
	    if(value instanceof GrappaPoint) {
		stringValue = ((GrappaPoint)value).toAttributeString();
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of GrappaPoint");
	    }
	    break;
	case SHAPE_TYPE:
	    if(value instanceof Integer) {
		stringValue = (String)Grappa.shapeToKey.get(value);
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of Integer");
	    }
	    break;
	case SIZE_TYPE:
	    if(value instanceof GrappaSize) {
		stringValue = ((GrappaSize)value).toAttributeString();
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of GrappaSize");
	    }
	    break;
	case STRING_TYPE:
	    if(value instanceof String) {
		stringValue = (String)value;
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of String");
	    }
	    break;
	case STYLE_TYPE:
	    if(value instanceof GrappaStyle) {
		stringValue = ((GrappaStyle)value).toAttributeString();
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of GrappaStyle");
	    }
	    break;
	default:
	    if(Attribute.specialHandler != null) {
		stringValue = Attribute.specialHandler.convertValue(type, name, value, attrtype);
	    } else {
		throw new RuntimeException(Element.typeString(type,true) + " attribute (" + name + ") needs a special handler");
	    }
	}

	if(stringValue == null && value != null) {
	    if(value instanceof String) {
		stringValue = (String)value;
	    } else {
		throw new RuntimeException("AttributeHandler needed to perform conversion of attribute \"" + name + "\", please supply one via Attribute.setAttributeHandler()");
	    }
	}

	return stringValue;
    }

    /**
     * Convert the supplied string value to the appropriate Object.
     * How to convert the value is
     * based on the type, name and attrtype information supplied. Note: this
     * method really could be declared static except that it hides the
     * instance method declared in the AttributeHandler interface and a
     * class method cannot hide an instance method.
     *
     * @param type the element type to which the named attribute applies
     * @param name the name of the attribute
     * @param value the string value to be converted to an object
     * @param attrtype the type of the attribute
     * @return an object representation of the supplied value
     */
    public Object convertStringValue(int type, String name, String stringValue, int attrtype) {
	Object value = null;

	if(stringValue == null || (stringValue != null && attrtype != STRING_TYPE && stringValue.trim().length() == 0)) {
	  value = null;
	  stringValue = null;
	} else {

	    switch(attrtype) {
	    case BOX_TYPE:
		// stringValue is x1, y1, x2, y2
		value = new GrappaBox(stringValue, false);
		break;
	    case COLOR_TYPE:
		value = GrappaColor.getColor(stringValue,null);
		break;
	    case DIR_TYPE:
		value = new Integer(GrappaSupport.xlateDirString(stringValue));
		break;
	    case DOUBLE_TYPE:
		try {
		    value = Double.valueOf(stringValue);
		}
		catch(NumberFormatException nfe) {
		    throw new IllegalArgumentException("bad number format (" + stringValue + ") for attribute \"" + name + "\"");
		}
		break;
	    case FONTSTYLE_TYPE:
		value = new Integer(GrappaSupport.xlateFontStyleString(stringValue));
		break;
	    case HASHLIST_TYPE:
		String[] listvals = GrappaSupport.strsplit(stringValue);
		if(this.value != null && this.value instanceof Hashtable) {
		    // is this more efficient than creating a new one??
		    // does this introduce the danger of users being
		    // tempted to hold on to the value??
		    value = this.value;
		    ((Hashtable)value).clear();
		} else {
		    value = new Hashtable();
		}
		for(int i=0; i<listvals.length; i++) {
		    ((Hashtable)value).put(listvals[i],listvals[i]);
		}
		break;
	    case INTEGER_TYPE:
		try {
		    value = Integer.valueOf(stringValue);
		}
		catch(NumberFormatException nfe) {
		    throw new IllegalArgumentException("bad integer format (" + stringValue + ") for attribute \"" + name + "\"");
		}
		break;
	    case LINE_TYPE:
		value = new GrappaLine(stringValue);
		break;
	    case POINT_TYPE:
		value = new GrappaPoint(stringValue);
		break;
	    case SHAPE_TYPE:
		if((value = (Integer)Grappa.keyToShape.get(stringValue)) == null) {
		    Attribute attr;
		    if((attr = Graph.getGlobalAttribute(Grappa.NODE, "shape")) == null || (value = (Integer)Grappa.keyToShape.get(attr.getValue())) == null) {
			throw new InternalError("could not provide default when unkown shape (" + stringValue + ") supplied for attribute \"" + name + "\"");
		    }
		}
		break;
	    case SIZE_TYPE:
		value = new GrappaSize(stringValue);
		break;
	    case STRING_TYPE:
		value = stringValue;
		break;
	    case STYLE_TYPE:
		value = new GrappaStyle(elementType, stringValue);
		break;
	    default:
		if(Attribute.specialHandler != null) {
		    value = Attribute.specialHandler.convertStringValue(type, name, stringValue, attrtype);
		} else {
		    throw new RuntimeException(Element.typeString(type,true) + " attribute (" + name + ") needs a special handler");
		}
	    }

	    if(value == null && stringValue != null) {
		value = stringValue;
	    }
	}

	return value;
    }

    /**
     * Make a copy of the supplied value. How to copy the value is
     * based on the type, name and attrtype information supplied. Note: this
     * method really could be declared static except that it hides the
     * instance method declared in the AttributeHandler interface and a
     * class method cannot hide an instance method.
     *
     * @param type the element type to which the named attribute applies
     * @param name the name of the attribute
     * @param value the attribute value to be copied
     * @param attrtype the type of the attribute
     * @return a copy of the supplied value
     */
    public Object copyValue(int type, String name, Object value, int attrtype) {
	Object copy_value = null;

	switch(attrtype) {
	case BOX_TYPE:
	    if(value instanceof GrappaBox) {
		copy_value = ((GrappaBox)value).clone();
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of GrappaBox");
	    }
	    break;
	case COLOR_TYPE:
	    if(value instanceof java.awt.Color) {
		copy_value = value;
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of Color");
	    }
	    break;
	case DIR_TYPE:
	    if(value instanceof Integer) {
		copy_value = value;
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of Integer");
	    }
	    break;
	case DOUBLE_TYPE:
	    if(value instanceof Double) {
		copy_value = value;
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of Double");
	    }
	    break;
	case FONTSTYLE_TYPE:
	    if(value instanceof Integer) {
		copy_value = value;
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of Integer");
	    }
	    break;
	case HASHLIST_TYPE:
	    if(value instanceof Hashtable) {
		copy_value = ((Hashtable)value).clone();
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of Integer");
	    }
	    break;
	case INTEGER_TYPE:
	    if(value instanceof Integer) {
		copy_value = value;
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of Integer");
	    }
	    break;
	case LINE_TYPE:
	    if(value instanceof GrappaLine) {
		copy_value = ((GrappaLine)value).clone();
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of GrappaLine");
	    }
	    break;
	case POINT_TYPE:
	    if(value instanceof GrappaPoint) {
		copy_value = ((GrappaPoint)value).clone();
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of GrappaPoint");
	    }
	    break;
	case SHAPE_TYPE:
	    if(value instanceof Integer) {
		copy_value = value;
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of Integer");
	    }
	    break;
	case SIZE_TYPE:
	    if(value instanceof GrappaSize) {
		copy_value = ((GrappaSize)value).clone();
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of GrappaSize");
	    }
	    break;
	case STRING_TYPE:
	    if(value instanceof String) {
		copy_value = value;
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of String");
	    }
	    break;
	case STYLE_TYPE:
	    if(value instanceof GrappaStyle) {
		copy_value = ((GrappaStyle)value).clone();
	    } else {
		throw new IllegalArgumentException("value of attribute \"" + name + "\" is not an instance of GrappaPoint");
	    }
	    break;
	default:
	    if(Attribute.specialHandler != null) {
		copy_value = Attribute.specialHandler.copyValue(type, name, value, attrtype);
	    } else {
		throw new RuntimeException(Element.typeString(type,true) + " attribute (" + name + ") needs a special handler");
	    }
	}

	if(copy_value == null && value != null) {
	    if(value instanceof String) {
		copy_value = value;
	    } else {
		throw new RuntimeException("AttributeHandler needed to perform copy of attribute \"" + name + "\", please supply one via Attribute.setAttributeHandler()");
	    }
	}

	return copy_value;
    }

    /**
     * Returns the attribute conversion type for the supplied attribute name and element type..
     *
     * @param elemType the element type
     * @param attrName the attribute name
     * @return the currently associated attribute type
     *
     * @see Element#attributeType
     * @see Node#attributeType
     * @see Edge#attributeType
     * @see Subgraph#attributeType
     * @see Graph#attributeType
     */
  public static int attributeType(int elemType, String attrName) {

    int attrType = _NO_TYPE;

	switch(elemType) {
	case NODE:
	    attrType = Node.attributeType(attrName);
	    break;
	case EDGE:
	    attrType = Edge.attributeType(attrName);
	    break;
	case SUBGRAPH:
	    attrType = Subgraph.attributeType(attrName);
	    break;
	case SYSTEM:
	    attrType = Graph.attributeType(attrName);
	    break;
	default:
	  // mention SYSTEM? it is for internal use, afterall...
	    throw new IllegalArgumentException("type of attribute \"" + attrName + "\" must be one of Grappa.NODE, Grappa.EDGE or Grappa.SUBGRAPH");
	}
	return(attrType);
  }
}
