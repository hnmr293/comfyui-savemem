import json
from io import BytesIO
from multiprocessing import shared_memory as sm
import struct

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class SaveImagesMemory:
    """
    Put images into shared memory with given tag.

    Memory format:
    - 4 bytes: number of images
    - 4n bytes: length of each image
    - rest: image data

    If given shared memory does not have enough space,
    only the writable header is written and the exception is raised.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {"tooltip": "name of the shared memory"}),
                "images": ("IMAGE",),
            },
            "optional": {"dummy_input": ("INT", {"default": 0, "tooltip": "dummy input for rerunning"})},
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "hnmr/image"

    def save_images(self, name: str, images: torch.Tensor, dummy_input: int = None, prompt=None, extra_pnginfo=None):
        lens = []
        io = BytesIO()

        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            img.save(io, format="png", pnginfo=metadata, compress_level=4)
            lens.append(io.tell())
            if 2 <= len(lens):
                lens[-1] -= lens[-2]

        data = io.getvalue()
        io.close()

        ll = struct.pack(f"=I{len(lens)}I", len(lens), *lens)
        total_len = len(ll) + len(data)

        shim = sm.SharedMemory(name=name)
        # may throw FileNotFoundError if the buffer is not found

        if shim.size < total_len:
            if len(ll) <= shim.size:
                shim.buf[: len(ll)] = ll
            elif 4 <= shim.size:
                shim.buf[:4] = struct.pack("=I", len(lens))
            shim.close()
            raise RuntimeError(f"Buffer size {shim.size} is smaller than required {total_len}")

        shim.buf[: len(ll)] = ll
        shim.buf[len(ll) : len(ll) + len(data)] = data
        shim.close()

        return {}


class SaveLatentsMemory:
    """
    Put latents into shared memory with given tag.

    To load latents, use `torch.load` with the same name.

    Memory format:
    - 8 bytes: bytes size
    - rest: data

    If given shared memory does not have enough space,
    only the writable header is written and the exception is raised.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {"tooltip": "name of the shared memory"}),
                "latents": ("LATENT",),
            },
            "optional": {"dummy_input": ("INT", {"default": 0, "tooltip": "dummy input for rerunning"})},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_latents"

    OUTPUT_NODE = True

    CATEGORY = "hnmr/latent"

    def save_latents(self, name: str, latents: dict, dummy_input: int = None):
        io = BytesIO()
        x: torch.Tensor = latents["samples"]
        torch.save(x.cpu().data, io)

        data = io.getvalue()
        io.close()

        ll = struct.pack(f"=Q", len(data))
        total_len = len(ll) + len(data)

        shim = sm.SharedMemory(name=name)
        # may throw FileNotFoundError if the buffer is not found

        if shim.size < total_len:
            if len(ll) <= shim.size:
                shim.buf[: len(ll)] = ll
            shim.close()
            raise RuntimeError(f"Buffer size {shim.size} is smaller than required {total_len}")

        shim.buf[: len(ll)] = ll
        shim.buf[len(ll) : len(ll) + len(data)] = data
        shim.close()

        return {}
