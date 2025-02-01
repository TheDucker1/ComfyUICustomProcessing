from .nodes import *


#  Map all your custom nodes classes with the names that will be displayed in the UI.
NODE_CLASS_MAPPINGS = {
    "L0 Smoothing": L0Smoother,
    "L0 Smoothing (EAP)": L0SmootherEAP,
    "L1 Smoothing": L1Smoother,
    "RTV Smoothing": RTVSmoother,
#    "(Down)Load My Model": MyModelLoader,
#    "Calculate Plus": CalculatePlus,
#    "Calculate Minus": CalculateMinus,
#    "Example Output Node": ExampleOutputNode,
}


__all__ = ['NODE_CLASS_MAPPINGS']
