package org.apache.sysds.runtime.io.cog;

public enum IFDTagDictionary {
    // Right now we will only support baseline TIFF
    // not the extended version
    NewSubfileType(254),
    ImageWidth(256),
    ImageLength(257),
    // TODO: Implement different bits per sample
    BitsPerSample(258),
    // TODO: LZW Compression
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
    // TODO: Find out if this is necessary
    // TODO: Distinguish between R G B R G B and R R R G G G B B B
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
    ExtraSamples(338),
    /**
     * 1 = unsigned integer data
     * 2 = two's complement signed integer data
     * 3 = IEEE floating point data [IEEE]
     * 4 = undefined data format
     * Has as many values as SamplesPerPixel
     */
    SampleFormat(339),

    // Extended tags we need (COG specifically)
    TileWidth(322),
    TileLength(323),
    TileOffsets(324),
    TileByteCounts(325);



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
