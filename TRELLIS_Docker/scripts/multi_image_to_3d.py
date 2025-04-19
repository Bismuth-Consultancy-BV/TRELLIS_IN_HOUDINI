import argparse
import os
import sys

sys.path.append("/app")

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils


def main(filename):
    # Load a pipeline from a model folder or a Hugging Face model hub.
    pipeline = TrellisImageTo3DPipeline.from_pretrained("/models/TRELLIS-image-large")
    pipeline.cuda()

    image_paths = os.listdir("/mnt/data/images/")
    images = [Image.open(os.path.join("/mnt/data/images/", image_path)) for image_path in image_paths]

    # Run the pipeline
    outputs = pipeline.run_multi_image(
        images,
        seed=1,
        # Optional parameters
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 3,
        },
    )

    glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            # Optional parameters
            simplify=0.95,  # Ratio of triangles to remove in the simplification process
            texture_size=1024,  # Size of the texture used for the GLB
        )
    glb.export("/mnt/data/" + filename + ".glb")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("filename", type=str)
    args = arg_parser.parse_args()
    main(args.filename)
