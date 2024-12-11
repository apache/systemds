package org.apache.sysds.runtime.io.cog;

public enum IFDTagDictionary {
    // Right now we will only support baseline TIFF
    // not the extended version
    NewSubfileType(254),
    ImageWidth(256),
    ImageLength(257),
    BitsPerSample(258),
    Compression(259),
    PhotometricInterpretation(262),
    Threshholding(263),
    CellWidth(264),
    CellLength(265),
    FillOrder(266),
    ImageDescription(270),
    Make(271),
    Model(272),
    StripOffsets(273),
    Orientation(274),
    SamplesPerPixel(277),
    RowsPerStrip(278),
    StripByteCounts(279),
    MinSampleValue(280),
    MaxSampleValue(281),
    XResolution(282),
    YResolution(283),
    PlanarConfiguration(284),
    FreeOffsets(288),
    FreeByteCounts(289),
    GrayResponseUnit(290),
    GrayResponseCurve(291),
    ResolutionUnit(296),
    Software(305),
    DateTime(306),
    Artist(315),
    HostComputer(316),
    ColorMap(320),
    ExtraSamples(338);



    private final int value;

    IFDTagDictionary(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public static IFDTagDictionary valueOf(int value) {
        for (IFDTagDictionary tag : IFDTagDictionary.values()) {
            if (tag.getValue() == value) {
                return tag;
            }
        }
        return null;
    }
}
