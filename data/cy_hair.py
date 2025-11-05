"""
Utility class for loading and formatting Cem-Yuksel Hair Dataset.
"""

import struct
import math

# Constants
_CY_HAIR_FILE_SEGMENTS_BIT = 1
_CY_HAIR_FILE_POINTS_BIT = 2
_CY_HAIR_FILE_THICKNESS_BIT = 4
_CY_HAIR_FILE_TRANSPARENCY_BIT = 8
_CY_HAIR_FILE_COLORS_BIT = 16

_CY_HAIR_FILE_INFO_SIZE = 88

# File read errors
CY_HAIR_FILE_ERROR_CANT_OPEN_FILE = -1
CY_HAIR_FILE_ERROR_CANT_READ_HEADER = -2
CY_HAIR_FILE_ERROR_WRONG_SIGNATURE = -3
CY_HAIR_FILE_ERROR_READING_SEGMENTS = -4
CY_HAIR_FILE_ERROR_READING_POINTS = -5
CY_HAIR_FILE_ERROR_READING_THICKNESS = -6
CY_HAIR_FILE_ERROR_READING_TRANSPARENCY = -7
CY_HAIR_FILE_ERROR_READING_COLORS = -8


class CYHairFile:
    class Header:
        def __init__(self):
            self.signature = ["H", "A", "I", "R"]
            self.hair_count = 0
            self.point_count = 0
            self.arrays = 0  # bit array of data in the file

            self.d_segments = 0
            self.d_thickness = 1.0
            self.d_transparency = 0.0
            self.d_color = [1.0, 1.0, 1.0]

            self.info = "\0" * _CY_HAIR_FILE_INFO_SIZE  # information about the file

    def __init__(self):
        self.segments = None
        self.points = None
        self.thickness = None
        self.transparency = None
        self.colors = None
        self.header = self.Header()

    def Initialize(self):
        self.segments = None
        self.points = None
        self.thickness = None
        self.transparency = None
        self.colors = None
        self.header = self.Header()

    # Constant Data Access Methods
    def GetHeader(self):
        return self.header

    def GetSegmentsArray(self):
        return self.segments

    def GetPointsArray(self):
        return self.points

    def GetThicknessArray(self):
        return self.thickness

    def GetTransparencyArray(self):
        return self.transparency

    def GetColorsArray(self):
        return self.colors

    # Data Access Methods
    # (In Python, getters return mutable references, so no need to distinguish)

    # Methods for Setting Array Sizes
    def SetHairCount(self, count):
        self.header.hair_count = count
        if self.segments is not None:
            self.segments = [0] * self.header.hair_count

    def SetPointCount(self, count):
        self.header.point_count = count
        if self.points is not None:
            self.points = [0.0] * (self.header.point_count * 3)
        if self.thickness is not None:
            self.thickness = [0.0] * self.header.point_count
        if self.transparency is not None:
            self.transparency = [0.0] * self.header.point_count
        if self.colors is not None:
            self.colors = [0.0] * (self.header.point_count * 3)

    def SetArrays(self, array_types):
        self.header.arrays = array_types
        if (self.header.arrays & _CY_HAIR_FILE_SEGMENTS_BIT) and self.segments is None:
            self.segments = [0] * self.header.hair_count
        if (
            not (self.header.arrays & _CY_HAIR_FILE_SEGMENTS_BIT)
            and self.segments is not None
        ):
            self.segments = None
        if (self.header.arrays & _CY_HAIR_FILE_POINTS_BIT) and self.points is None:
            self.points = [0.0] * (self.header.point_count * 3)
        if (
            not (self.header.arrays & _CY_HAIR_FILE_POINTS_BIT)
            and self.points is not None
        ):
            self.points = None
        if (
            self.header.arrays & _CY_HAIR_FILE_THICKNESS_BIT
        ) and self.thickness is None:
            self.thickness = [0.0] * self.header.point_count
        if (
            not (self.header.arrays & _CY_HAIR_FILE_THICKNESS_BIT)
            and self.thickness is not None
        ):
            self.thickness = None
        if (
            self.header.arrays & _CY_HAIR_FILE_TRANSPARENCY_BIT
        ) and self.transparency is None:
            self.transparency = [0.0] * self.header.point_count
        if (
            not (self.header.arrays & _CY_HAIR_FILE_TRANSPARENCY_BIT)
            and self.transparency is not None
        ):
            self.transparency = None
        if (self.header.arrays & _CY_HAIR_FILE_COLORS_BIT) and self.colors is None:
            self.colors = [0.0] * (self.header.point_count * 3)
        if (
            not (self.header.arrays & _CY_HAIR_FILE_COLORS_BIT)
            and self.colors is not None
        ):
            self.colors = None

    def SetDefaultSegmentCount(self, s):
        self.header.d_segments = s

    def SetDefaultThickness(self, t):
        self.header.d_thickness = t

    def SetDefaultTransparency(self, t):
        self.header.d_transparency = t

    def SetDefaultColor(self, r, g, b):
        self.header.d_color = [r, g, b]

    # Load and Save Methods
    def LoadFromFile(self, filename):
        self.Initialize()
        try:
            with open(filename, "rb") as f:
                header_data = f.read(128)  # Header is 128 bytes
                if len(header_data) < 128:
                    return CY_HAIR_FILE_ERROR_CANT_READ_HEADER

                unpacked_data = struct.unpack("<4sIIIIff3f88s", header_data)
                self.header.signature = list(unpacked_data[0].decode("ascii"))
                self.header.hair_count = unpacked_data[1]
                self.header.point_count = unpacked_data[2]
                self.header.arrays = unpacked_data[3]
                self.header.d_segments = unpacked_data[4]
                self.header.d_thickness = unpacked_data[5]
                self.header.d_transparency = unpacked_data[6]
                self.header.d_color = list(unpacked_data[7:10])
                self.header.info = unpacked_data[10].decode("ascii").rstrip("\0")

                # Check signature
                if "".join(self.header.signature) != "HAIR":
                    return CY_HAIR_FILE_ERROR_WRONG_SIGNATURE

                # Read segments array
                if self.header.arrays & _CY_HAIR_FILE_SEGMENTS_BIT:
                    segments_data = f.read(self.header.hair_count * 2)
                    if len(segments_data) < self.header.hair_count * 2:
                        return CY_HAIR_FILE_ERROR_READING_SEGMENTS
                    self.segments = list(
                        struct.unpack("<" + "H" * self.header.hair_count, segments_data)
                    )

                # Read points array
                if self.header.arrays & _CY_HAIR_FILE_POINTS_BIT:
                    points_data = f.read(self.header.point_count * 3 * 4)
                    if len(points_data) < self.header.point_count * 3 * 4:
                        return CY_HAIR_FILE_ERROR_READING_POINTS
                    self.points = list(
                        struct.unpack(
                            "<" + "f" * self.header.point_count * 3, points_data
                        )
                    )

                # Read thickness array
                if self.header.arrays & _CY_HAIR_FILE_THICKNESS_BIT:
                    thickness_data = f.read(self.header.point_count * 4)
                    if len(thickness_data) < self.header.point_count * 4:
                        return CY_HAIR_FILE_ERROR_READING_THICKNESS
                    self.thickness = list(
                        struct.unpack(
                            "<" + "f" * self.header.point_count, thickness_data
                        )
                    )

                # Read transparency array
                if self.header.arrays & _CY_HAIR_FILE_TRANSPARENCY_BIT:
                    transparency_data = f.read(self.header.point_count * 4)
                    if len(transparency_data) < self.header.point_count * 4:
                        return CY_HAIR_FILE_ERROR_READING_TRANSPARENCY
                    self.transparency = list(
                        struct.unpack(
                            "<" + "f" * self.header.point_count, transparency_data
                        )
                    )

                # Read colors array
                if self.header.arrays & _CY_HAIR_FILE_COLORS_BIT:
                    colors_data = f.read(self.header.point_count * 3 * 4)
                    if len(colors_data) < self.header.point_count * 3 * 4:
                        return CY_HAIR_FILE_ERROR_READING_COLORS
                    self.colors = list(
                        struct.unpack(
                            "<" + "f" * self.header.point_count * 3, colors_data
                        )
                    )

            return self.header.hair_count
        except IOError:
            return CY_HAIR_FILE_ERROR_CANT_OPEN_FILE

    def SaveToFile(self, filename):
        try:
            with open(filename, "wb") as f:
                # Prepare the header data
                header_packed = struct.pack(
                    "<4sIIIIff3f88s",
                    bytes("".join(self.header.signature), "ascii"),
                    self.header.hair_count,
                    self.header.point_count,
                    self.header.arrays,
                    self.header.d_segments,
                    self.header.d_thickness,
                    self.header.d_transparency,
                    self.header.d_color[0],
                    self.header.d_color[1],
                    self.header.d_color[2],
                    bytes(
                        self.header.info.ljust(_CY_HAIR_FILE_INFO_SIZE, "\0"), "ascii"
                    ),
                )
                f.write(header_packed)

                # Write arrays
                if (
                    self.header.arrays & _CY_HAIR_FILE_SEGMENTS_BIT
                    and self.segments is not None
                ):
                    segments_packed = struct.pack(
                        "<" + "H" * self.header.hair_count, *self.segments
                    )
                    f.write(segments_packed)

                if (
                    self.header.arrays & _CY_HAIR_FILE_POINTS_BIT
                    and self.points is not None
                ):
                    points_packed = struct.pack(
                        "<" + "f" * self.header.point_count * 3, *self.points
                    )
                    f.write(points_packed)

                if (
                    self.header.arrays & _CY_HAIR_FILE_THICKNESS_BIT
                    and self.thickness is not None
                ):
                    thickness_packed = struct.pack(
                        "<" + "f" * self.header.point_count, *self.thickness
                    )
                    f.write(thickness_packed)

                if (
                    self.header.arrays & _CY_HAIR_FILE_TRANSPARENCY_BIT
                    and self.transparency is not None
                ):
                    transparency_packed = struct.pack(
                        "<" + "f" * self.header.point_count, *self.transparency
                    )
                    f.write(transparency_packed)

                if (
                    self.header.arrays & _CY_HAIR_FILE_COLORS_BIT
                    and self.colors is not None
                ):
                    colors_packed = struct.pack(
                        "<" + "f" * self.header.point_count * 3, *self.colors
                    )
                    f.write(colors_packed)

            return self.header.hair_count
        except IOError:
            return -1

    # Other Methods
    def FillDirectionArray(self, dir_array):
        if dir_array is None or self.header.point_count <= 0 or self.points is None:
            return 0

        p = 0  # point index
        for i in range(self.header.hair_count):
            s = self.segments[i] if self.segments else self.header.d_segments
            if s > 1:
                len0, len1 = self.ComputeDirection(
                    dir_array[(p + 1) * 3 : (p + 1) * 3 + 3],
                    self.points[p * 3 : p * 3 + 3],
                    self.points[(p + 1) * 3 : (p + 1) * 3 + 3],
                    self.points[(p + 2) * 3 : (p + 2) * 3 + 3],
                )

                # direction at point0
                d0 = [
                    self.points[(p + 1) * 3]
                    - dir_array[(p + 1) * 3] * len0 * 0.3333
                    - self.points[p * 3],
                    self.points[(p + 1) * 3 + 1]
                    - dir_array[(p + 1) * 3 + 1] * len0 * 0.3333
                    - self.points[p * 3 + 1],
                    self.points[(p + 1) * 3 + 2]
                    - dir_array[(p + 1) * 3 + 2] * len0 * 0.3333
                    - self.points[p * 3 + 2],
                ]
                d0len_sq = sum([x**2 for x in d0])
                d0len = math.sqrt(d0len_sq) if d0len_sq > 0 else 1.0
                dir_array[p * 3 : p * 3 + 3] = [x / d0len for x in d0]

                p += 2  # We computed the first 2 points

                # Compute the direction for the rest
                for t in range(2, s):
                    len0, len1 = self.ComputeDirection(
                        dir_array[p * 3 : p * 3 + 3],
                        self.points[(p - 1) * 3 : (p - 1) * 3 + 3],
                        self.points[p * 3 : p * 3 + 3],
                        self.points[(p + 1) * 3 : (p + 1) * 3 + 3],
                    )
                    p += 1

                # direction at the last point
                d0 = [
                    -self.points[(p - 1) * 3]
                    + dir_array[(p - 1) * 3] * len1 * 0.3333
                    + self.points[p * 3],
                    -self.points[(p - 1) * 3 + 1]
                    + dir_array[(p - 1) * 3 + 1] * len1 * 0.3333
                    + self.points[p * 3 + 1],
                    -self.points[(p - 1) * 3 + 2]
                    + dir_array[(p - 1) * 3 + 2] * len1 * 0.3333
                    + self.points[p * 3 + 2],
                ]
                d0len_sq = sum([x**2 for x in d0])
                d0len = math.sqrt(d0len_sq) if d0len_sq > 0 else 1.0
                dir_array[p * 3 : p * 3 + 3] = [x / d0len for x in d0]
                p += 1
            elif s > 0:
                # if it has a single segment
                d0 = [
                    self.points[(p + 1) * 3] - self.points[p * 3],
                    self.points[(p + 1) * 3 + 1] - self.points[p * 3 + 1],
                    self.points[(p + 1) * 3 + 2] - self.points[p * 3 + 2],
                ]
                d0len_sq = sum([x**2 for x in d0])
                d0len = math.sqrt(d0len_sq) if d0len_sq > 0 else 1.0
                dir_array[p * 3 : p * 3 + 3] = [x / d0len for x in d0]
                dir_array[(p + 1) * 3 : (p + 1) * 3 + 3] = dir_array[p * 3 : p * 3 + 3]
                p += 2

        return p

    def ComputeDirection(self, d, p0, p1, p2):
        # line from p0 to p1
        d0 = [p1[i] - p0[i] for i in range(3)]
        d0len_sq = sum(x * x for x in d0)
        d0len = math.sqrt(d0len_sq) if d0len_sq > 0 else 1.0

        # line from p1 to p2
        d1 = [p2[i] - p1[i] for i in range(3)]
        d1len_sq = sum(x * x for x in d1)
        d1len = math.sqrt(d1len_sq) if d1len_sq > 0 else 1.0

        # make sure that d0 and d1 has the same length
        scale = d1len / d0len
        d0 = [x * scale for x in d0]

        # direction at p1
        d_vec = [d0[i] + d1[i] for i in range(3)]
        dlen_sq = sum(x * x for x in d_vec)
        dlen = math.sqrt(dlen_sq) if dlen_sq > 0 else 1.0
        for i in range(3):
            d[i] = d_vec[i] / dlen

        return d0len, d1len
