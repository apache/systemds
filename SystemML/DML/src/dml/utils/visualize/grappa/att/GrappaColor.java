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
import java.util.*;

/**
 * This abstract class sets up and provides name-to-color and color-to-name
 * mappings and some associated class methods.
 *
 * @version 1.2, 04 Mar 2008; Copyright 1996 - 2008 by AT&T Corp.
 * @author  <a href="mailto:john@research.att.com">John Mocenigo</a>, <a href="http://www.research.att.com">Research @ AT&T Labs</a>
 */
public abstract class GrappaColor
{
  // given name, get color
  private static Hashtable colorTable  = new Hashtable(660,10);
  // given color, get name
  private static Hashtable colorLookUp = new Hashtable(660,10);

  // initialize colorTable
  static {
    doAddColor("aliceblue",new Color(240,248,255));
    doAddColor("antiquewhite",new Color(250,235,215));
    doAddColor("antiquewhite1",new Color(255,239,219),false);
    doAddColor("antiquewhite2",new Color(238,223,204),false);
    doAddColor("antiquewhite3",new Color(205,192,176),false);
    doAddColor("antiquewhite4",new Color(139,131,120),false);
    doAddColor("aquamarine",new Color(127,255,212));
    doAddColor("aquamarine1",new Color(127,255,212),false);
    doAddColor("aquamarine2",new Color(118,238,198),false);
    doAddColor("aquamarine3",new Color(102,205,170),false);
    doAddColor("aquamarine4",new Color(69,139,116),false);
    doAddColor("azure",new Color(240,255,255));
    doAddColor("azure1",new Color(240,255,255),false);
    doAddColor("azure2",new Color(224,238,238),false);
    doAddColor("azure3",new Color(193,205,205),false);
    doAddColor("azure4",new Color(131,139,139),false);
    doAddColor("beige",new Color(245,245,220));
    doAddColor("bisque",new Color(255,228,196));
    doAddColor("bisque1",new Color(255,228,196),false);
    doAddColor("bisque2",new Color(238,213,183),false);
    doAddColor("bisque3",new Color(205,183,158),false);
    doAddColor("bisque4",new Color(139,125,107),false);
    doAddColor("black",new Color(0,0,0));
    doAddColor("blanchedalmond",new Color(255,235,205));
    doAddColor("blue",new Color(0,0,255));
    doAddColor("blue1",new Color(0,0,255),false);
    doAddColor("blue2",new Color(0,0,238),false);
    doAddColor("blue3",new Color(0,0,205),false);
    doAddColor("blue4",new Color(0,0,139),false);
    doAddColor("blueviolet",new Color(138,43,226));
    doAddColor("brown",new Color(165,42,42));
    doAddColor("brown1",new Color(255,64,64),false);
    doAddColor("brown2",new Color(238,59,59),false);
    doAddColor("brown3",new Color(205,51,51),false);
    doAddColor("brown4",new Color(139,35,35),false);
    doAddColor("burlywood",new Color(222,184,135));
    doAddColor("burlywood1",new Color(255,211,155),false);
    doAddColor("burlywood2",new Color(238,197,145),false);
    doAddColor("burlywood3",new Color(205,170,125),false);
    doAddColor("burlywood4",new Color(139,115,85),false);
    doAddColor("cadetblue",new Color(95,158,160));
    doAddColor("cadetblue1",new Color(152,245,255),false);
    doAddColor("cadetblue2",new Color(142,229,238),false);
    doAddColor("cadetblue3",new Color(122,197,205),false);
    doAddColor("cadetblue4",new Color(83,134,139),false);
    doAddColor("chartreuse",new Color(127,255,0));
    doAddColor("chartreuse1",new Color(127,255,0),false);
    doAddColor("chartreuse2",new Color(118,238,0),false);
    doAddColor("chartreuse3",new Color(102,205,0),false);
    doAddColor("chartreuse4",new Color(69,139,0),false);
    doAddColor("chocolate",new Color(210,105,30));
    doAddColor("chocolate1",new Color(255,127,36),false);
    doAddColor("chocolate2",new Color(238,118,33),false);
    doAddColor("chocolate3",new Color(205,102,29),false);
    doAddColor("chocolate4",new Color(139,69,19),false);
    doAddColor("coral",new Color(255,127,80));
    doAddColor("coral1",new Color(255,114,86),false);
    doAddColor("coral2",new Color(238,106,80),false);
    doAddColor("coral3",new Color(205,91,69),false);
    doAddColor("coral4",new Color(139,62,47),false);
    doAddColor("cornflowerblue",new Color(100,149,237));
    doAddColor("cornsilk",new Color(255,248,220));
    doAddColor("cornsilk1",new Color(255,248,220),false);
    doAddColor("cornsilk2",new Color(238,232,205),false);
    doAddColor("cornsilk3",new Color(205,200,177),false);
    doAddColor("cornsilk4",new Color(139,136,120),false);
    doAddColor("crimson",new Color(220,20,60));
    doAddColor("cyan",new Color(0,255,255));
    doAddColor("cyan1",new Color(0,255,255),false);
    doAddColor("cyan2",new Color(0,238,238),false);
    doAddColor("cyan3",new Color(0,205,205),false);
    doAddColor("cyan4",new Color(0,139,139),false);
    doAddColor("darkblue",new Color(0,0,139));
    doAddColor("darkcyan",new Color(0,139,139));
    doAddColor("darkgoldenrod",new Color(184,134,11));
    doAddColor("darkgoldenrod1",new Color(255,185,15),false);
    doAddColor("darkgoldenrod2",new Color(238,173,14),false);
    doAddColor("darkgoldenrod3",new Color(205,149,12),false);
    doAddColor("darkgoldenrod4",new Color(139,101,8),false);
    doAddColor("darkgray",new Color(169,169,169));
    doAddColor("darkgreen",new Color(0,100,0));
    doAddColor("darkgrey",new Color(169,169,169),false);
    doAddColor("darkkhaki",new Color(189,183,107));
    doAddColor("darkmagenta",new Color(139,0,139));
    doAddColor("darkolivegreen",new Color(85,107,47));
    doAddColor("darkolivegreen1",new Color(202,255,112),false);
    doAddColor("darkolivegreen2",new Color(188,238,104),false);
    doAddColor("darkolivegreen3",new Color(162,205,90),false);
    doAddColor("darkolivegreen4",new Color(110,139,61),false);
    doAddColor("darkorange",new Color(255,140,0));
    doAddColor("darkorange1",new Color(255,127,0),false);
    doAddColor("darkorange2",new Color(238,118,0),false);
    doAddColor("darkorange3",new Color(205,102,0),false);
    doAddColor("darkorange4",new Color(139,69,0),false);
    doAddColor("darkorchid",new Color(153,50,204));
    doAddColor("darkorchid1",new Color(191,62,255),false);
    doAddColor("darkorchid2",new Color(178,58,238),false);
    doAddColor("darkorchid3",new Color(154,50,205),false);
    doAddColor("darkorchid4",new Color(104,34,139),false);
    doAddColor("darkred",new Color(139,0,0));
    doAddColor("darksalmon",new Color(233,150,122));
    doAddColor("darkseagreen",new Color(143,188,143));
    doAddColor("darkseagreen1",new Color(193,255,193),false);
    doAddColor("darkseagreen2",new Color(180,238,180),false);
    doAddColor("darkseagreen3",new Color(155,205,155),false);
    doAddColor("darkseagreen4",new Color(105,139,105),false);
    doAddColor("darkslateblue",new Color(72,61,139));
    doAddColor("darkslategray",new Color(47,79,79));
    doAddColor("darkslategray1",new Color(151,255,255),false);
    doAddColor("darkslategray2",new Color(141,238,238),false);
    doAddColor("darkslategray3",new Color(121,205,205),false);
    doAddColor("darkslategray4",new Color(82,139,139),false);
    doAddColor("darkslategrey",new Color(47,79,79),false);
    doAddColor("darkturquoise",new Color(0,206,209));
    doAddColor("darkviolet",new Color(148,0,211));
    doAddColor("deeppink",new Color(255,20,147));
    doAddColor("deeppink1",new Color(255,20,147),false);
    doAddColor("deeppink2",new Color(238,18,137),false);
    doAddColor("deeppink3",new Color(205,16,118),false);
    doAddColor("deeppink4",new Color(139,10,80),false);
    doAddColor("deepskyblue",new Color(0,191,255));
    doAddColor("deepskyblue1",new Color(0,191,255),false);
    doAddColor("deepskyblue2",new Color(0,178,238),false);
    doAddColor("deepskyblue3",new Color(0,154,205),false);
    doAddColor("deepskyblue4",new Color(0,104,139),false);
    doAddColor("dimgray",new Color(105,105,105));
    doAddColor("dimgrey",new Color(105,105,105),false);
    doAddColor("dodgerblue",new Color(30,144,255));
    doAddColor("dodgerblue1",new Color(30,144,255),false);
    doAddColor("dodgerblue2",new Color(28,134,238),false);
    doAddColor("dodgerblue3",new Color(24,116,205),false);
    doAddColor("dodgerblue4",new Color(16,78,139),false);
    doAddColor("firebrick",new Color(178,34,34));
    doAddColor("firebrick1",new Color(255,48,48),false);
    doAddColor("firebrick2",new Color(238,44,44),false);
    doAddColor("firebrick3",new Color(205,38,38),false);
    doAddColor("firebrick4",new Color(139,26,26),false);
    doAddColor("floralwhite",new Color(255,250,240));
    doAddColor("forestgreen",new Color(34,139,34));
    doAddColor("gainsboro",new Color(220,220,220));
    doAddColor("ghostwhite",new Color(248,248,255));
    doAddColor("gold",new Color(255,215,0));
    doAddColor("gold1",new Color(255,215,0),false);
    doAddColor("gold2",new Color(238,201,0),false);
    doAddColor("gold3",new Color(205,173,0),false);
    doAddColor("gold4",new Color(139,117,0),false);
    doAddColor("goldenrod",new Color(218,165,32));
    doAddColor("goldenrod1",new Color(255,193,37),false);
    doAddColor("goldenrod2",new Color(238,180,34),false);
    doAddColor("goldenrod3",new Color(205,155,29),false);
    doAddColor("goldenrod4",new Color(139,105,20),false);
    doAddColor("green",new Color(0,255,0));
    doAddColor("green1",new Color(0,255,0),false);
    doAddColor("green2",new Color(0,238,0),false);
    doAddColor("green3",new Color(0,205,0),false);
    doAddColor("green4",new Color(0,139,0),false);
    doAddColor("greenyellow",new Color(173,255,47));
    doAddColor("gray",new Color(190,190,190));
    doAddColor("grey",new Color(190,190,190),false);
    doAddColor("gray0",new Color(0,0,0),false);
    doAddColor("grey0",new Color(0,0,0),false);
    doAddColor("gray1",new Color(3,3,3),false);
    doAddColor("grey1",new Color(3,3,3),false);
    doAddColor("gray2",new Color(5,5,5),false);
    doAddColor("grey2",new Color(5,5,5),false);
    doAddColor("gray3",new Color(8,8,8),false);
    doAddColor("grey3",new Color(8,8,8),false);
    doAddColor("gray4",new Color(10,10,10),false);
    doAddColor("grey4",new Color(10,10,10),false);
    doAddColor("gray5",new Color(13,13,13),false);
    doAddColor("grey5",new Color(13,13,13),false);
    doAddColor("gray6",new Color(15,15,15),false);
    doAddColor("grey6",new Color(15,15,15),false);
    doAddColor("gray7",new Color(18,18,18),false);
    doAddColor("grey7",new Color(18,18,18),false);
    doAddColor("gray8",new Color(20,20,20),false);
    doAddColor("grey8",new Color(20,20,20),false);
    doAddColor("gray9",new Color(23,23,23),false);
    doAddColor("grey9",new Color(23,23,23),false);
    doAddColor("gray10",new Color(26,26,26),false);
    doAddColor("grey10",new Color(26,26,26),false);
    doAddColor("gray11",new Color(28,28,28),false);
    doAddColor("grey11",new Color(28,28,28),false);
    doAddColor("gray12",new Color(31,31,31),false);
    doAddColor("grey12",new Color(31,31,31),false);
    doAddColor("gray13",new Color(33,33,33),false);
    doAddColor("grey13",new Color(33,33,33),false);
    doAddColor("gray14",new Color(36,36,36),false);
    doAddColor("grey14",new Color(36,36,36),false);
    doAddColor("gray15",new Color(38,38,38),false);
    doAddColor("grey15",new Color(38,38,38),false);
    doAddColor("gray16",new Color(41,41,41),false);
    doAddColor("grey16",new Color(41,41,41),false);
    doAddColor("gray17",new Color(43,43,43),false);
    doAddColor("grey17",new Color(43,43,43),false);
    doAddColor("gray18",new Color(46,46,46),false);
    doAddColor("grey18",new Color(46,46,46),false);
    doAddColor("gray19",new Color(48,48,48),false);
    doAddColor("grey19",new Color(48,48,48),false);
    doAddColor("gray20",new Color(51,51,51),false);
    doAddColor("grey20",new Color(51,51,51),false);
    doAddColor("gray21",new Color(54,54,54),false);
    doAddColor("grey21",new Color(54,54,54),false);
    doAddColor("gray22",new Color(56,56,56),false);
    doAddColor("grey22",new Color(56,56,56),false);
    doAddColor("gray23",new Color(59,59,59),false);
    doAddColor("grey23",new Color(59,59,59),false);
    doAddColor("gray24",new Color(61,61,61),false);
    doAddColor("grey24",new Color(61,61,61),false);
    doAddColor("gray25",new Color(64,64,64),false);
    doAddColor("grey25",new Color(64,64,64),false);
    doAddColor("gray26",new Color(66,66,66),false);
    doAddColor("grey26",new Color(66,66,66),false);
    doAddColor("gray27",new Color(69,69,69),false);
    doAddColor("grey27",new Color(69,69,69),false);
    doAddColor("gray28",new Color(71,71,71),false);
    doAddColor("grey28",new Color(71,71,71),false);
    doAddColor("gray29",new Color(74,74,74),false);
    doAddColor("grey29",new Color(74,74,74),false);
    doAddColor("gray30",new Color(77,77,77),false);
    doAddColor("grey30",new Color(77,77,77),false);
    doAddColor("gray31",new Color(79,79,79),false);
    doAddColor("grey31",new Color(79,79,79),false);
    doAddColor("gray32",new Color(82,82,82),false);
    doAddColor("grey32",new Color(82,82,82),false);
    doAddColor("gray33",new Color(84,84,84),false);
    doAddColor("grey33",new Color(84,84,84),false);
    doAddColor("gray34",new Color(87,87,87),false);
    doAddColor("grey34",new Color(87,87,87),false);
    doAddColor("gray35",new Color(89,89,89),false);
    doAddColor("grey35",new Color(89,89,89),false);
    doAddColor("gray36",new Color(92,92,92),false);
    doAddColor("grey36",new Color(92,92,92),false);
    doAddColor("gray37",new Color(94,94,94),false);
    doAddColor("grey37",new Color(94,94,94),false);
    doAddColor("gray38",new Color(97,97,97),false);
    doAddColor("grey38",new Color(97,97,97),false);
    doAddColor("gray39",new Color(99,99,99),false);
    doAddColor("grey39",new Color(99,99,99),false);
    doAddColor("gray40",new Color(102,102,102),false);
    doAddColor("grey40",new Color(102,102,102),false);
    doAddColor("gray41",new Color(105,105,105),false);
    doAddColor("grey41",new Color(105,105,105),false);
    doAddColor("gray42",new Color(107,107,107),false);
    doAddColor("grey42",new Color(107,107,107),false);
    doAddColor("gray43",new Color(110,110,110),false);
    doAddColor("grey43",new Color(110,110,110),false);
    doAddColor("gray44",new Color(112,112,112),false);
    doAddColor("grey44",new Color(112,112,112),false);
    doAddColor("gray45",new Color(115,115,115),false);
    doAddColor("grey45",new Color(115,115,115),false);
    doAddColor("gray46",new Color(117,117,117),false);
    doAddColor("grey46",new Color(117,117,117),false);
    doAddColor("gray47",new Color(120,120,120),false);
    doAddColor("grey47",new Color(120,120,120),false);
    doAddColor("gray48",new Color(122,122,122),false);
    doAddColor("grey48",new Color(122,122,122),false);
    doAddColor("gray49",new Color(125,125,125),false);
    doAddColor("grey49",new Color(125,125,125),false);
    doAddColor("gray50",new Color(127,127,127),false);
    doAddColor("grey50",new Color(127,127,127),false);
    doAddColor("gray51",new Color(130,130,130),false);
    doAddColor("grey51",new Color(130,130,130),false);
    doAddColor("gray52",new Color(133,133,133),false);
    doAddColor("grey52",new Color(133,133,133),false);
    doAddColor("gray53",new Color(135,135,135),false);
    doAddColor("grey53",new Color(135,135,135),false);
    doAddColor("gray54",new Color(138,138,138),false);
    doAddColor("grey54",new Color(138,138,138),false);
    doAddColor("gray55",new Color(140,140,140),false);
    doAddColor("grey55",new Color(140,140,140),false);
    doAddColor("gray56",new Color(143,143,143),false);
    doAddColor("grey56",new Color(143,143,143),false);
    doAddColor("gray57",new Color(145,145,145),false);
    doAddColor("grey57",new Color(145,145,145),false);
    doAddColor("gray58",new Color(148,148,148),false);
    doAddColor("grey58",new Color(148,148,148),false);
    doAddColor("gray59",new Color(150,150,150),false);
    doAddColor("grey59",new Color(150,150,150),false);
    doAddColor("gray60",new Color(153,153,153),false);
    doAddColor("grey60",new Color(153,153,153),false);
    doAddColor("gray61",new Color(156,156,156),false);
    doAddColor("grey61",new Color(156,156,156),false);
    doAddColor("gray62",new Color(158,158,158),false);
    doAddColor("grey62",new Color(158,158,158),false);
    doAddColor("gray63",new Color(161,161,161),false);
    doAddColor("grey63",new Color(161,161,161),false);
    doAddColor("gray64",new Color(163,163,163),false);
    doAddColor("grey64",new Color(163,163,163),false);
    doAddColor("gray65",new Color(166,166,166),false);
    doAddColor("grey65",new Color(166,166,166),false);
    doAddColor("gray66",new Color(168,168,168),false);
    doAddColor("grey66",new Color(168,168,168),false);
    doAddColor("gray67",new Color(171,171,171),false);
    doAddColor("grey67",new Color(171,171,171),false);
    doAddColor("gray68",new Color(173,173,173),false);
    doAddColor("grey68",new Color(173,173,173),false);
    doAddColor("gray69",new Color(176,176,176),false);
    doAddColor("grey69",new Color(176,176,176),false);
    doAddColor("gray70",new Color(179,179,179),false);
    doAddColor("grey70",new Color(179,179,179),false);
    doAddColor("gray71",new Color(181,181,181),false);
    doAddColor("grey71",new Color(181,181,181),false);
    doAddColor("gray72",new Color(184,184,184),false);
    doAddColor("grey72",new Color(184,184,184),false);
    doAddColor("gray73",new Color(186,186,186),false);
    doAddColor("grey73",new Color(186,186,186),false);
    doAddColor("gray74",new Color(189,189,189),false);
    doAddColor("grey74",new Color(189,189,189),false);
    doAddColor("gray75",new Color(191,191,191),false);
    doAddColor("grey75",new Color(191,191,191),false);
    doAddColor("gray76",new Color(194,194,194),false);
    doAddColor("grey76",new Color(194,194,194),false);
    doAddColor("gray77",new Color(196,196,196),false);
    doAddColor("grey77",new Color(196,196,196),false);
    doAddColor("gray78",new Color(199,199,199),false);
    doAddColor("grey78",new Color(199,199,199),false);
    doAddColor("gray79",new Color(201,201,201),false);
    doAddColor("grey79",new Color(201,201,201),false);
    doAddColor("gray80",new Color(204,204,204),false);
    doAddColor("grey80",new Color(204,204,204),false);
    doAddColor("gray81",new Color(207,207,207),false);
    doAddColor("grey81",new Color(207,207,207),false);
    doAddColor("gray82",new Color(209,209,209),false);
    doAddColor("grey82",new Color(209,209,209),false);
    doAddColor("gray83",new Color(212,212,212),false);
    doAddColor("grey83",new Color(212,212,212),false);
    doAddColor("gray84",new Color(214,214,214),false);
    doAddColor("grey84",new Color(214,214,214),false);
    doAddColor("gray85",new Color(217,217,217),false);
    doAddColor("grey85",new Color(217,217,217),false);
    doAddColor("gray86",new Color(219,219,219),false);
    doAddColor("grey86",new Color(219,219,219),false);
    doAddColor("gray87",new Color(222,222,222),false);
    doAddColor("grey87",new Color(222,222,222),false);
    doAddColor("gray88",new Color(224,224,224),false);
    doAddColor("grey88",new Color(224,224,224),false);
    doAddColor("gray89",new Color(227,227,227),false);
    doAddColor("grey89",new Color(227,227,227),false);
    doAddColor("gray90",new Color(229,229,229),false);
    doAddColor("grey90",new Color(229,229,229),false);
    doAddColor("gray91",new Color(232,232,232),false);
    doAddColor("grey91",new Color(232,232,232),false);
    doAddColor("gray92",new Color(235,235,235),false);
    doAddColor("grey92",new Color(235,235,235),false);
    doAddColor("gray93",new Color(237,237,237),false);
    doAddColor("grey93",new Color(237,237,237),false);
    doAddColor("gray94",new Color(240,240,240),false);
    doAddColor("grey94",new Color(240,240,240),false);
    doAddColor("gray95",new Color(242,242,242),false);
    doAddColor("grey95",new Color(242,242,242),false);
    doAddColor("gray96",new Color(245,245,245),false);
    doAddColor("grey96",new Color(245,245,245),false);
    doAddColor("gray97",new Color(247,247,247),false);
    doAddColor("grey97",new Color(247,247,247),false);
    doAddColor("gray98",new Color(250,250,250),false);
    doAddColor("grey98",new Color(250,250,250),false);
    doAddColor("gray99",new Color(252,252,252),false);
    doAddColor("grey99",new Color(252,252,252),false);
    doAddColor("gray100",new Color(255,255,255),false);
    doAddColor("grey100",new Color(255,255,255),false);
    doAddColor("honeydew",new Color(240,255,240));
    doAddColor("honeydew1",new Color(240,255,240),false);
    doAddColor("honeydew2",new Color(224,238,224),false);
    doAddColor("honeydew3",new Color(193,205,193),false);
    doAddColor("honeydew4",new Color(131,139,131),false);
    doAddColor("hotpink",new Color(255,105,180));
    doAddColor("hotpink1",new Color(255,110,180),false);
    doAddColor("hotpink2",new Color(238,106,167),false);
    doAddColor("hotpink3",new Color(205,96,144),false);
    doAddColor("hotpink4",new Color(139,58,98),false);
    doAddColor("indianred",new Color(205,92,92));
    doAddColor("indianred1",new Color(255,106,106),false);
    doAddColor("indianred2",new Color(238,99,99),false);
    doAddColor("indianred3",new Color(205,85,85),false);
    doAddColor("indianred4",new Color(139,58,58),false);
    doAddColor("indigo",new Color(75,0,130));
    doAddColor("ivory",new Color(255,255,240));
    doAddColor("ivory1",new Color(255,255,240),false);
    doAddColor("ivory2",new Color(238,238,224),false);
    doAddColor("ivory3",new Color(205,205,193),false);
    doAddColor("ivory4",new Color(139,139,131),false);
    doAddColor("khaki",new Color(240,230,140));
    doAddColor("khaki1",new Color(255,246,143),false);
    doAddColor("khaki2",new Color(238,230,133),false);
    doAddColor("khaki3",new Color(205,198,115),false);
    doAddColor("khaki4",new Color(139,134,78),false);
    doAddColor("lavender",new Color(230,230,250));
    doAddColor("lavenderblush",new Color(255,240,245));
    doAddColor("lavenderblush1",new Color(255,240,245),false);
    doAddColor("lavenderblush2",new Color(238,224,229),false);
    doAddColor("lavenderblush3",new Color(205,193,197),false);
    doAddColor("lavenderblush4",new Color(139,131,134),false);
    doAddColor("lawngreen",new Color(124,252,0));
    doAddColor("lemonchiffon",new Color(255,250,205));
    doAddColor("lemonchiffon1",new Color(255,250,205),false);
    doAddColor("lemonchiffon2",new Color(238,233,191),false);
    doAddColor("lemonchiffon3",new Color(205,201,165),false);
    doAddColor("lemonchiffon4",new Color(139,137,112),false);
    doAddColor("lightblue",new Color(173,216,230));
    doAddColor("lightblue1",new Color(191,239,255),false);
    doAddColor("lightblue2",new Color(178,223,238),false);
    doAddColor("lightblue3",new Color(154,192,205),false);
    doAddColor("lightblue4",new Color(104,131,139),false);
    doAddColor("lightcoral",new Color(240,128,128));
    doAddColor("lightcyan",new Color(224,255,255));
    doAddColor("lightcyan1",new Color(224,255,255),false);
    doAddColor("lightcyan2",new Color(209,238,238),false);
    doAddColor("lightcyan3",new Color(180,205,205),false);
    doAddColor("lightcyan4",new Color(122,139,139),false);
    doAddColor("lightgoldenrod",new Color(238,221,130));
    doAddColor("lightgoldenrod1",new Color(255,236,139),false);
    doAddColor("lightgoldenrod2",new Color(238,220,130),false);
    doAddColor("lightgoldenrod3",new Color(205,190,112),false);
    doAddColor("lightgoldenrod4",new Color(139,129,76),false);
    doAddColor("lightgoldenrodyellow",new Color(250,250,210));
    doAddColor("lightgray",new Color(211,211,211));
    doAddColor("lightgreen",new Color(144,238,144));
    doAddColor("lightgrey",new Color(211,211,211),false);
    doAddColor("lightpink",new Color(255,182,193));
    doAddColor("lightpink1",new Color(255,174,185),false);
    doAddColor("lightpink2",new Color(238,162,173),false);
    doAddColor("lightpink3",new Color(205,140,149),false);
    doAddColor("lightpink4",new Color(139,95,101),false);
    doAddColor("lightsalmon",new Color(255,160,122));
    doAddColor("lightsalmon1",new Color(255,160,122),false);
    doAddColor("lightsalmon2",new Color(238,149,114),false);
    doAddColor("lightsalmon3",new Color(205,129,98),false);
    doAddColor("lightsalmon4",new Color(139,87,66),false);
    doAddColor("lightseagreen",new Color(32,178,170));
    doAddColor("lightskyblue",new Color(135,206,250));
    doAddColor("lightskyblue1",new Color(176,226,255),false);
    doAddColor("lightskyblue2",new Color(164,211,238),false);
    doAddColor("lightskyblue3",new Color(141,182,205),false);
    doAddColor("lightskyblue4",new Color(96,123,139),false);
    doAddColor("lightslateblue",new Color(132,112,255));
    doAddColor("lightslategray",new Color(119,136,153));
    doAddColor("lightslategrey",new Color(119,136,153),false);
    doAddColor("lightsteelblue",new Color(176,196,222));
    doAddColor("lightsteelblue1",new Color(202,225,255),false);
    doAddColor("lightsteelblue2",new Color(188,210,238),false);
    doAddColor("lightsteelblue3",new Color(162,181,205),false);
    doAddColor("lightsteelblue4",new Color(110,123,139),false);
    doAddColor("lightyellow",new Color(255,255,224));
    doAddColor("lightyellow1",new Color(255,255,224),false);
    doAddColor("lightyellow2",new Color(238,238,209),false);
    doAddColor("lightyellow3",new Color(205,205,180),false);
    doAddColor("lightyellow4",new Color(139,139,122),false);
    doAddColor("limegreen",new Color(50,205,50));
    doAddColor("linen",new Color(250,240,230));
    doAddColor("magenta",new Color(255,0,255));
    doAddColor("magenta1",new Color(255,0,255),false);
    doAddColor("magenta2",new Color(238,0,238),false);
    doAddColor("magenta3",new Color(205,0,205),false);
    doAddColor("magenta4",new Color(139,0,139),false);
    doAddColor("maroon",new Color(176,48,96));
    doAddColor("maroon1",new Color(255,52,179),false);
    doAddColor("maroon2",new Color(238,48,167),false);
    doAddColor("maroon3",new Color(205,41,144),false);
    doAddColor("maroon4",new Color(139,28,98),false);
    doAddColor("mediumaquamarine",new Color(102,205,170));
    doAddColor("mediumblue",new Color(0,0,205));
    doAddColor("mediumorchid",new Color(186,85,211));
    doAddColor("mediumorchid1",new Color(224,102,255),false);
    doAddColor("mediumorchid2",new Color(209,95,238),false);
    doAddColor("mediumorchid3",new Color(180,82,205),false);
    doAddColor("mediumorchid4",new Color(122,55,139),false);
    doAddColor("mediumpurple",new Color(147,112,219));
    doAddColor("mediumpurple1",new Color(171,130,255),false);
    doAddColor("mediumpurple2",new Color(159,121,238),false);
    doAddColor("mediumpurple3",new Color(137,104,205),false);
    doAddColor("mediumpurple4",new Color(93,71,139),false);
    doAddColor("mediumseagreen",new Color(60,179,113));
    doAddColor("mediumslateblue",new Color(123,104,238));
    doAddColor("mediumspringgreen",new Color(0,250,154));
    doAddColor("mediumturquoise",new Color(72,209,204));
    doAddColor("mediumvioletred",new Color(199,21,133));
    doAddColor("midnightblue",new Color(25,25,112));
    doAddColor("mintcream",new Color(245,255,250));
    doAddColor("mistyrose",new Color(255,228,225));
    doAddColor("mistyrose1",new Color(255,228,225),false);
    doAddColor("mistyrose2",new Color(238,213,210),false);
    doAddColor("mistyrose3",new Color(205,183,181),false);
    doAddColor("mistyrose4",new Color(139,125,123),false);
    doAddColor("moccasin",new Color(255,228,181));
    doAddColor("navajowhite",new Color(255,222,173));
    doAddColor("navajowhite1",new Color(255,222,173),false);
    doAddColor("navajowhite2",new Color(238,207,161),false);
    doAddColor("navajowhite3",new Color(205,179,139),false);
    doAddColor("navajowhite4",new Color(139,121,94),false);
    doAddColor("navy",new Color(0,0,128));
    doAddColor("navyblue",new Color(0,0,128),false);
    doAddColor("oldlace",new Color(253,245,230));
    doAddColor("olivedrab",new Color(107,142,35));
    doAddColor("olivedrab1",new Color(192,255,62),false);
    doAddColor("olivedrab2",new Color(179,238,58),false);
    doAddColor("olivedrab3",new Color(154,205,50),false);
    doAddColor("olivedrab4",new Color(105,139,34),false);
    doAddColor("orange",new Color(255,165,0));
    doAddColor("orange1",new Color(255,165,0),false);
    doAddColor("orange2",new Color(238,154,0),false);
    doAddColor("orange3",new Color(205,133,0),false);
    doAddColor("orange4",new Color(139,90,0),false);
    doAddColor("orangered",new Color(255,69,0));
    doAddColor("orangered1",new Color(255,69,0),false);
    doAddColor("orangered2",new Color(238,64,0),false);
    doAddColor("orangered3",new Color(205,55,0),false);
    doAddColor("orangered4",new Color(139,37,0),false);
    doAddColor("orchid",new Color(218,112,214));
    doAddColor("orchid1",new Color(255,131,250),false);
    doAddColor("orchid2",new Color(238,122,233),false);
    doAddColor("orchid3",new Color(205,105,201),false);
    doAddColor("orchid4",new Color(139,71,137),false);
    doAddColor("palegoldenrod",new Color(238,232,170));
    doAddColor("palegreen",new Color(152,251,152));
    doAddColor("palegreen1",new Color(154,255,154),false);
    doAddColor("palegreen2",new Color(144,238,144),false);
    doAddColor("palegreen3",new Color(124,205,124),false);
    doAddColor("palegreen4",new Color(84,139,84),false);
    doAddColor("paleturquoise",new Color(175,238,238));
    doAddColor("paleturquoise1",new Color(187,255,255),false);
    doAddColor("paleturquoise2",new Color(174,238,238),false);
    doAddColor("paleturquoise3",new Color(150,205,205),false);
    doAddColor("paleturquoise4",new Color(102,139,139),false);
    doAddColor("palevioletred",new Color(219,112,147));
    doAddColor("palevioletred1",new Color(255,130,171),false);
    doAddColor("palevioletred2",new Color(238,121,159),false);
    doAddColor("palevioletred3",new Color(205,104,137),false);
    doAddColor("palevioletred4",new Color(139,71,93),false);
    doAddColor("papayawhip",new Color(255,239,213));
    doAddColor("peachpuff",new Color(255,218,185));
    doAddColor("peachpuff1",new Color(255,218,185),false);
    doAddColor("peachpuff2",new Color(238,203,173),false);
    doAddColor("peachpuff3",new Color(205,175,149),false);
    doAddColor("peachpuff4",new Color(139,119,101),false);
    doAddColor("peru",new Color(205,133,63));
    doAddColor("pink",new Color(255,192,203));
    doAddColor("pink1",new Color(255,181,197),false);
    doAddColor("pink2",new Color(238,169,184),false);
    doAddColor("pink3",new Color(205,145,158),false);
    doAddColor("pink4",new Color(139,99,108),false);
    doAddColor("plum",new Color(221,160,221));
    doAddColor("plum1",new Color(255,187,255),false);
    doAddColor("plum2",new Color(238,174,238),false);
    doAddColor("plum3",new Color(205,150,205),false);
    doAddColor("plum4",new Color(139,102,139),false);
    doAddColor("powderblue",new Color(176,224,230));
    doAddColor("purple",new Color(160,32,240));
    doAddColor("purple1",new Color(155,48,255),false);
    doAddColor("purple2",new Color(145,44,238),false);
    doAddColor("purple3",new Color(125,38,205),false);
    doAddColor("purple4",new Color(85,26,139),false);
    doAddColor("red",new Color(255,0,0));
    doAddColor("red1",new Color(255,0,0),false);
    doAddColor("red2",new Color(238,0,0),false);
    doAddColor("red3",new Color(205,0,0),false);
    doAddColor("red4",new Color(139,0,0),false);
    doAddColor("rosybrown",new Color(188,143,143));
    doAddColor("rosybrown1",new Color(255,193,193),false);
    doAddColor("rosybrown2",new Color(238,180,180),false);
    doAddColor("rosybrown3",new Color(205,155,155),false);
    doAddColor("rosybrown4",new Color(139,105,105),false);
    doAddColor("royalblue",new Color(65,105,225));
    doAddColor("royalblue1",new Color(72,118,255),false);
    doAddColor("royalblue2",new Color(67,110,238),false);
    doAddColor("royalblue3",new Color(58,95,205),false);
    doAddColor("royalblue4",new Color(39,64,139),false);
    doAddColor("saddlebrown",new Color(139,69,19));
    doAddColor("salmon",new Color(250,128,114));
    doAddColor("salmon1",new Color(255,140,105),false);
    doAddColor("salmon2",new Color(238,130,98),false);
    doAddColor("salmon3",new Color(205,112,84),false);
    doAddColor("salmon4",new Color(139,76,57),false);
    doAddColor("sandybrown",new Color(244,164,96));
    doAddColor("seagreen",new Color(46,139,87));
    doAddColor("seagreen1",new Color(84,255,159),false);
    doAddColor("seagreen2",new Color(78,238,148),false);
    doAddColor("seagreen3",new Color(67,205,128),false);
    doAddColor("seagreen4",new Color(46,139,87),false);
    doAddColor("seashell",new Color(255,245,238));
    doAddColor("seashell1",new Color(255,245,238),false);
    doAddColor("seashell2",new Color(238,229,222),false);
    doAddColor("seashell3",new Color(205,197,191),false);
    doAddColor("seashell4",new Color(139,134,130),false);
    doAddColor("sgiindigo2",new Color(33,136,104),false);
    doAddColor("sienna",new Color(160,82,45));
    doAddColor("sienna1",new Color(255,130,71),false);
    doAddColor("sienna2",new Color(238,121,66),false);
    doAddColor("sienna3",new Color(205,104,57),false);
    doAddColor("sienna4",new Color(139,71,38),false);
    doAddColor("skyblue",new Color(135,206,235));
    doAddColor("skyblue1",new Color(135,206,255),false);
    doAddColor("skyblue2",new Color(126,192,238),false);
    doAddColor("skyblue3",new Color(108,166,205),false);
    doAddColor("skyblue4",new Color(74,112,139),false);
    doAddColor("slateblue",new Color(106,90,205));
    doAddColor("slateblue1",new Color(131,111,255),false);
    doAddColor("slateblue2",new Color(122,103,238),false);
    doAddColor("slateblue3",new Color(105,89,205),false);
    doAddColor("slateblue4",new Color(71,60,139),false);
    doAddColor("slategray",new Color(112,128,144));
    doAddColor("slategray1",new Color(198,226,255),false);
    doAddColor("slategray2",new Color(185,211,238),false);
    doAddColor("slategray3",new Color(159,182,205),false);
    doAddColor("slategray4",new Color(108,123,139),false);
    doAddColor("slategrey",new Color(112,128,144),false);
    doAddColor("snow",new Color(255,250,250));
    doAddColor("snow1",new Color(255,250,250),false);
    doAddColor("snow2",new Color(238,233,233),false);
    doAddColor("snow3",new Color(205,201,201),false);
    doAddColor("snow4",new Color(139,137,137),false);
    doAddColor("springgreen",new Color(0,255,127));
    doAddColor("springgreen1",new Color(0,255,127),false);
    doAddColor("springgreen2",new Color(0,238,118),false);
    doAddColor("springgreen3",new Color(0,205,102),false);
    doAddColor("springgreen4",new Color(0,139,69),false);
    doAddColor("steelblue",new Color(70,130,180));
    doAddColor("steelblue1",new Color(99,184,255),false);
    doAddColor("steelblue2",new Color(92,172,238),false);
    doAddColor("steelblue3",new Color(79,148,205),false);
    doAddColor("steelblue4",new Color(54,100,139),false);
    doAddColor("tan",new Color(210,180,140));
    doAddColor("tan1",new Color(255,165,79),false);
    doAddColor("tan2",new Color(238,154,73),false);
    doAddColor("tan3",new Color(205,133,63),false);
    doAddColor("tan4",new Color(139,90,43),false);
    doAddColor("thistle",new Color(216,191,216));
    doAddColor("thistle1",new Color(255,225,255),false);
    doAddColor("thistle2",new Color(238,210,238),false);
    doAddColor("thistle3",new Color(205,181,205),false);
    doAddColor("thistle4",new Color(139,123,139),false);
    doAddColor("tomato",new Color(255,99,71));
    doAddColor("tomato1",new Color(255,99,71),false);
    doAddColor("tomato2",new Color(238,92,66),false);
    doAddColor("tomato3",new Color(205,79,57),false);
    doAddColor("tomato4",new Color(139,54,38),false);
    doAddColor("turquoise",new Color(64,224,208));
    doAddColor("turquoise1",new Color(0,245,255),false);
    doAddColor("turquoise2",new Color(0,229,238),false);
    doAddColor("turquoise3",new Color(0,197,205),false);
    doAddColor("turquoise4",new Color(0,134,139),false);
    doAddColor("violet",new Color(238,130,238));
    doAddColor("violetred",new Color(208,32,144));
    doAddColor("violetred1",new Color(255,62,150),false);
    doAddColor("violetred2",new Color(238,58,140),false);
    doAddColor("violetred3",new Color(205,50,120),false);
    doAddColor("violetred4",new Color(139,34,82),false);
    doAddColor("wheat",new Color(245,222,179));
    doAddColor("wheat1",new Color(255,231,186),false);
    doAddColor("wheat2",new Color(238,216,174),false);
    doAddColor("wheat3",new Color(205,186,150),false);
    doAddColor("wheat4",new Color(139,126,102),false);
    doAddColor("white",new Color(255,255,255));
    doAddColor("whitesmoke",new Color(245,245,245));
    doAddColor("yellow",new Color(255,255,0));
    doAddColor("yellow1",new Color(255,255,0),false);
    doAddColor("yellow2",new Color(238,238,0),false);
    doAddColor("yellow3",new Color(205,205,0),false);
    doAddColor("yellow4",new Color(139,139,0),false);
    doAddColor("yellowgreen",new Color(154,205,50));
  }

  // be sure to specify colors that exist initially in the colorTable
  /**
   * The default foreground color (black).
   */
  public static final Color  defaultForeground = getColor("black",null);
  /**
   * The default background color (white).
   */
  public static final Color  defaultBackground = getColor("white",null);
  /**
   * The default XOR color (light gray).
   */
  public static final Color  defaultXOR        = getColor("light gray",null);
  /**
   * The default font color (black).
   */
  public static final Color  defaultFontcolor  = getColor("black",null);
  /**
   * The default color of last resort in all cases (black).
   */
  public static final Color  defaultColor      = getColor("black",null);

  /**
   * Adds a color to the application color table. For search purposes, names
   * are canonicalized by converting to lower case and stripping
   * non-alphanumerics.  A name must contains at least one alphabetic.
   * Once in the table, colors can be set by name, and names can be
   * retrieved by color (although a single color referred to by multiple names
   * only causes the retrieval of the last name mapped to that color).
   *
   * @param name the name to be used to reference the color.
   * @param color the Color value.
   */
  public static void addColor(String name, Color color) throws IllegalArgumentException {
    if(name == null || color == null) {
      throw new IllegalArgumentException("supplied name or color is null");
    }
    String canonName = canonColor(name, null);
    if(canonName == null) {
      throw new IllegalArgumentException("supplied name does not contain alphabetics (" + name + ")");
    }
    doAddColor(canonName,color);
  }

  // performs actual color table puts
  private static void doAddColor(String name, Color color, boolean override) {
    colorTable.put(name,color);
    if(override || colorLookUp.get(color) == null) colorLookUp.put(color,name);
  }

  // convenience version
  private static void doAddColor(String name, Color color) {
      doAddColor(name,color,true);
  }

  // canonicalizes color string (removes non-alphanumeric and lowers case)
  private static String canonColor(String name, float[] hsb) {
    if(hsb != null) {
      hsb[0] = hsb[1] = hsb[2] = -1;
    }
    if(name == null) return null;
    char[] array = name.toCharArray();
    int len = 0;
    int commas = 0;
    int[] commaSpots = new int[3];
    int dots = 0;
    boolean allDigits = true;
    for(int i = 0; i < array.length; i++) {
      if(Character.isUpperCase(array[i])) {
	array[len++] = Character.toLowerCase(array[i]);
	allDigits = false;
      } else if(Character.isLowerCase(array[i])) {
	array[len++] = array[i];
	allDigits = false;
      } else if(Character.isDigit(array[i])) {
	array[len++] = array[i];
      } else if(array[i] == ',') {
	if(commas < 2) {
	  commaSpots[commas] = i;
	}
	commas++;
	array[len++] = array[i];
      } else if(array[i] == '.') {
	dots++;
	array[len++] = array[i];
      }
    }

    if(hsb != null && allDigits && commas == 2 && dots <= 3) {
      commaSpots[2] = array.length;
      int prev = 0;
      try {
	for(int i = 0;  i < 3; i++) {
	  hsb[i] = Float.valueOf(name.substring(prev,commaSpots[i])).floatValue();
	  prev = commaSpots[i] + 1;
	}
      } catch(NumberFormatException nfe) {
	return null;
      }
      return new String(array,0,len);
    }
    if(len == 0 || allDigits) return null;
    if(commas > 0 || dots > 0) {
      int l = len;
      len = 0;
      for(int i = 0; i < l; i++) {
	if(array[i] != '.' && array[i] != ',') {
	  array[len++] = array[i];
	}
      }
    }
    return new String(array,0,len);
  }

  /**
   * Return the color in the color table with the given name.
   * If the color is not found, the supplied default is returned.
   * If the supplied default is null, the class default is returned.
   * If the name consists of three comma or space separated floating
   * point numbers in the range 0 to 1 inclusive, then it is assumed
   * to represent an HSB color specification and generated directly.
   * The name search is case insensitive and looks at alphanumerics only.
   *
   * @param name the name of the color to be retrieved.
   * @param color the color value to return if requested color
   *              is not found.
   *
   * @return the color matching the name or the default.
   */
  public static Color getColor(String name, Color color) {
    if(color == null) color = defaultColor;
    
    if(name == null) return color;

    float[] hsb = new float[3];

    String canonName = canonColor(name, hsb);

    Color retColor = (Color)colorTable.get(canonName);

    if(retColor == null) {
      if(hsb[0] < 0) {
	retColor = color;
      } else {
	retColor = Color.getHSBColor(hsb[0],hsb[1],hsb[2]);
	if(retColor == null) {
	  retColor = color;
	} else {
	  doAddColor(canonName,retColor);
	}
      }
    }

    return retColor;
  }

  /**
   * Return the name of the supplied color.
   *
   * @param color the color whose name is to be retrieved.
   *
   * @return the color's (most recently entered) name, if it is in the
   *         color table, or its HSB value string otherwise.
   */
  public static String getColorName(Color color) {
      if(color == null) return null;
      String name = (String)(colorLookUp.get(color));
      if(name == null) {
	  float[] hsb = Color.RGBtoHSB(color.getRed(),color.getGreen(),color.getBlue(),null);
	  name = hsb[0]+","+hsb[1]+","+hsb[2];
      }
      return(name);
  }

}
