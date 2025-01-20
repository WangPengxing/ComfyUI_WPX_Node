__version__ = "0.0.1"

from .main_image import AnimalContour, RectangleContour, CircleContour, PenetrateStyle, AnimalContourSilhouette, SplitStickers
from .main_image import DetermineRowsAndCols, SplitMaskElements
# 插件的节点类映射
NODE_CLASS_MAPPINGS = {
    "AnimalContour": AnimalContour,
    "AnimalContourSilhouette": AnimalContourSilhouette,
    "CircleContour": CircleContour,
    "PenetrateStyle": PenetrateStyle,
    "RectangleContour": RectangleContour,
    "SplitStickers": SplitStickers,
    "DetermineRowsAndCols": DetermineRowsAndCols,
    "SplitMaskElements": SplitMaskElements,
}

# 节点的显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimalContour": "AnimalContour",
    "AnimalContourSilhouette": "AnimalContourSilhouette",
    "CircleContour": "CircleContour",
    "PenetrateStyle": "PenetrateStyle",
    "RectangleContour": "RectangleContour",
    "SplitStickers": "SplitStickers",
    "DetermineRowsAndCols": "DetermineRowsAndCols",
    "SplitMaskElements": "SplitMaskElements",
}

print(f"[WPX_Node] __version__ : {__version__}")    
