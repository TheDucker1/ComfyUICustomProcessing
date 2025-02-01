#  Package Modules
import os
from typing import Union, BinaryIO, Dict, List, Tuple, Optional
import time
import torch
import numpy as np

#  ComfyUI Modules
import folder_paths
from comfy.utils import ProgressBar

#  Your Modules
from .modules.calculator import CalculatorModel
from .modules.l0smoothing import L0Smooth
from .modules.l1smoothing import L1Smooth
from .modules.rtv import tsmooth
from .modules.eap import EAP

#  Basic practice to get paths from ComfyUI
custom_nodes_script_dir = os.path.dirname(os.path.abspath(__file__))
#custom_nodes_model_dir = os.path.join(folder_paths.models_dir, "my-custom-nodes")
#custom_nodes_output_dir = os.path.join(folder_paths.get_output_directory(), "my-custom-nodes")


#  These are example nodes that only contains basic functionalities with some comments.
#  If you need detailed explanation, please refer to : https://docs.comfy.org/essentials/custom_node_walkthrough
#  First Node:

class L0Smoother:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "is_grayscale": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "display": "slider",
                }),
                "_lambda": ("FLOAT", {
                    "default": 2e-2,
                    "min": 1e-3,
                    "max": 1,
                    "step": 1e-3,
                }),
                "_kappa": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.1,
                    "max": 5.0,
                    "step": 0.1,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    CATEGORY = "examples"
    FUNCTION = "l0smooth"

    def l0smooth(self, images, is_grayscale, _lambda, _kappa):
        if _lambda is None:
            _lambda = 2e-2
        if _kappa is None:
            _kappa = 2.0
        L = []
        images = images.cpu()
        for i in range(images.shape[0]):
            img = images[i].numpy().copy()
            if is_grayscale:
                img = img[:,:,0]
            img = L0Smooth(img, _lambda, _kappa)
            if is_grayscale:
                if len(img.shape) < 3:
                    img = img[:,:,np.newaxis]
                img = np.repeat(img, 3, axis=2)
            L.append(torch.from_numpy(img))
        return (torch.stack(L, dim=0),)

class L0SmootherEAP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "iteration": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "number",
                }),
            },
            "optional": {
                "is_grayscale": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "display": "slider",
                }),
                "_lambda": ("FLOAT", {
                    "default": 2e-2,
                    "min": 1e-3,
                    "max": 1,
                    "step": 1e-3,
                }),
                "_kappa": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.1,
                    "max": 5.0,
                    "step": 0.1,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    CATEGORY = "examples"
    FUNCTION = "l0smooth"

    def l0smooth(self, images, iteration, is_grayscale, _lambda, _kappa):
        if _lambda is None:
            _lambda = 2e-2
        if _kappa is None:
            _kappa = 2.0
        L = []
        images = images.cpu()
        SmoothFunc = lambda x, msk: L0Smooth(x, _lambda, _kappa, msk)
        for i in range(images.shape[0]):
            img = images[i].numpy().copy()
            if is_grayscale:
                img = img[:,:,0]
            img = EAP(img, SmoothFunc, 0.1, iteration)
            if is_grayscale:
                if len(img.shape) < 3:
                    img = img[:,:,np.newaxis]
                img = np.repeat(img, 3, axis=2)
            L.append(torch.from_numpy(img))
        return (torch.stack(L, dim=0),)

class L1Smoother:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "is_grayscale": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "display": "slider",
                }),
                "edge_preserving": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "display": "slider",
                }),
                "alpha": ("FLOAT", {
                    "default": 50.0,
                    "min": 0,
                    "max": 100,
                    "step": 0.01,
                }),
                "beta": ("FLOAT", {
                    "default": 5.0,
                    "min": 0,
                    "max": 100,
                    "step": 0.01,
                }),
                "gamma": ("FLOAT", {
                    "default": 2.5,
                    "min": 0,
                    "max": 100,
                    "step": 0.01,
                }),
                "_lambda": ("FLOAT", {
                    "default": 5.0,
                    "min": 0,
                    "max": 100,
                    "step": 0.01,
                }),
                "maxIteration": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                }),
                "kappa": ("FLOAT", {
                    "default": 3.0,
                    "min": 0,
                    "max": 100,
                    "step": 0.01,
                }),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0,
                    "max": 100,
                    "step": 0.1,
                }),
                "half_window": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 7,
                    "step": 1,
                }),
                "eta": ("FLOAT", {
                    "default": 15.0,
                    "min": 0,
                    "max": 100,
                    "step": 0.01,
                }),
                "global_size": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0,
                    "max": 10,
                    "step": 0.001,
                }),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    CATEGORY = "examples"
    FUNCTION = "l1smooth"

    def l1smooth(self, images, is_grayscale, edge_preserving, alpha, beta, gamma, _lambda, maxIteration, kappa, sigma, half_window, eta, global_size, threshold):
        images = images.cpu()
        L = []
        for i in range(images.shape[0]):
            img = images[i].numpy().copy()
            if is_grayscale:
                img = img[:,:,0]
            img = L1Smooth(img, alpha, beta, gamma, _lambda, maxIteration, kappa, sigma, half_window, eta, edge_preserving, global_size, threshold)
            if is_grayscale:
                if len(img.shape) < 3:
                    img = img[:,:,np.newaxis]
                img = np.repeat(img, 3, axis=2)
            L.append(torch.from_numpy(img))
        return (torch.stack(L, dim=0),)

class RTVSmoother:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "is_grayscale": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1,
                    "step": 1,
                    "display": "slider",
                }),
                "_lambda": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.001,
                }),
                "sigma": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01,
                }),
                "sharpness": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.001,
                }),
                "maxIteration": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 100,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    CATEGORY = "examples"
    FUNCTION = "rtvsmooth"

    def rtvsmooth(self, images, is_grayscale, _lambda, sigma, sharpness, maxIteration):
        if _lambda is None:
            _lambda = 0.01
        if sigma is None:
            sigma = 3.0
        if sharpness is None:
            sharpness = 0.02
        if maxIteration is None:
            maxIteration = 4
        L = []
        images = images.cpu()
        for i in range(images.shape[0]):
            img = images[i].numpy().copy()
            if is_grayscale:
                img = img[:,:,0]
            img = tsmooth(img, _lambda, sigma, sharpness, maxIteration)
            if is_grayscale:
                if len(img.shape) < 3:
                    img = img[:,:,np.newaxis]
                img = np.repeat(img, 3, axis=2)
            L.append(torch.from_numpy(img))
        return (torch.stack(L, dim=0),)

class MyModelLoader:
    #  Define the input parameters of the node here.
    @classmethod
    def INPUT_TYPES(s):
        my_models = ["Model A", "Model B", "Model C"]

        return {
            #  If the key is "required", the value must be filled.
            "required": {
                #  `my_models` is the list, so it will be shown as a dropdown menu in the node. ( So that user can select one of them. )
                #  You must provide the value in the tuple format. e.g. ("value",) or (3,) or ([1, 2],) etc.
                "model": (my_models,),
                "device": (['cuda', 'cpu', 'auto'],),
            },
            #  If the key is "optional", the value is optional.
            "optional": {
                "compute_type": (['float32', 'float16'],),
            }
        }

    #  Define these constants inside the node.
    #  `RETURN_TYPES` is important, as it limits the parameter types that can be passed to the next node, in `INPUT_TYPES()` above.
    RETURN_TYPES = ("MY_MODEL",)
    RETURN_NAMES = ("my_model",)
    #  `FUNCTION` is the function name that will be called in the node.
    FUNCTION = "load_model"
    #  `CATEGORY` is the category name that will be used when user searches the node.
    CATEGORY = "CustomNodesTemplate"

    #  In the function, use same parameter names as you specified in `INPUT_TYPES()`
    def load_model(self,
                   model: str,
                   device: str,
                   compute_type: Optional[str] = None,
                   ) -> Tuple[CalculatorModel]:
        calculator_model = CalculatorModel()
        calculator_model.load_model(model, device, compute_type)

        #  You can use `comfy.utils.ProgressBar` to show the progress of the process.
        #  First, initialize the total amount of the process.
        total_steps = 5
        comfy_pbar = ProgressBar(total_steps)
        #  Then, update the progress.
        for i in range(1, total_steps):
            time.sleep(1)
            comfy_pbar.update(i)  #  Alternatively, you can use `comfy_pbar.update_absolute(value)` to update the progress with absolute value.

        #  Return the model as a tuple.
        return (calculator_model, )

#  Second Node
class CalculatePlus:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MY_MODEL", ),
            },
            #  Specify the parameters with type and default value.
            "optional": {
                "a": ("INT", {"default": 5}),
                "b": ("INT", {"default": 10}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("plus_value",)
    FUNCTION = "plus"
    CATEGORY = "CustomNodesTemplate"

    def plus(self,
             model: CalculatorModel,
             a: Optional[int],
             b: Optional[int],
             ) -> Tuple[int]:
        result = model.plus(a, b)
        return (result, )



#  Third Node
class CalculateMinus:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MY_MODEL", ),
                "a": ("INT", ),
            },
            "optional": {
                "b": ("INT", {"default": 10}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("minus_value",)
    FUNCTION = "minus"
    CATEGORY = "CustomNodesTemplate"

    def minus(self,
             model: CalculatorModel,
             a: Optional[int],
             b: Optional[int],
             ) -> Tuple[int]:
        result = model.minus(a, b)
        return (result, )



#  Output Node
class ExampleOutputNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("INT", ),
            },
        }

    #  If the node is output node, set this to True.
    OUTPUT_NODE = True
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "result"
    CATEGORY = "CustomNodesTemplate"

    def result(self,
               value: int,) -> Tuple[int]:
        return (value, )
