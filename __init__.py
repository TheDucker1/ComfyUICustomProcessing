from .nodes import *


#  Map all your custom nodes classes with the names that will be displayed in the UI.
NODE_CLASS_MAPPINGS = {
    "L0 Smoothing": L0Smoother,
    "L0 Smoothing (EAP)": L0SmootherEAP,
    "L1 Smoothing": L1Smoother,
    "RTV Smoothing": RTVSmoother,
    "Guided Filter": GuidedFilterer,
    "MangaLine Model Loader": MangaLineExtractionModelLoader,
    "MangaLine Extraction": MangaLineExtraction,
    "DanbooRegion Model Loader": DanbooRegionModelLoader,
    "DanbooRegion Segmentator": DanbooRegionSegmentator,
    "MEMatte Model Loader": MEMatteModelLoader,
    "MEMatte Fix Alpha": MEMatteFixMatt,
    "Extract Bounding Box From Mask": BoundingBoxFromMask,
    "Image Reflection Pad": ImagePadReflect,
    "Get Image Size": ImageSizeGet,
#    "(Down)Load My Model": MyModelLoader,
#    "Calculate Plus": CalculatePlus,
#    "Calculate Minus": CalculateMinus,
#    "Example Output Node": ExampleOutputNode,
}


__all__ = ['NODE_CLASS_MAPPINGS']
