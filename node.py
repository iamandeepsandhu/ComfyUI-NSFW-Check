# Use a pipeline as a high-level helper
from PIL import Image
from transformers import pipeline
import torchvision.transforms as T
import torch
import numpy


def tensor2pil(image):
    return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))


def pil2tensor(image):
    return torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0).unsqueeze(0)


class NSFWScore:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),            
            },
        }

    RETURN_TYPES = ("FLOAT",)

    FUNCTION = "run"

    CATEGORY = "NSFWScore"

    def run(self, image):
        transform = T.ToPILImage()
        cuda = torch.cuda.is_available()
        classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device="cuda" if cuda else "cpu")
        
        result = classifier(transform(image[0].permute(2, 0, 1)))
        nsfw_score = 0.0
        
        for r in result:
            if r["label"] == "nsfw":
                nsfw_score = r["score"]
                
        return (nsfw_score,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "NSFWScore": NSFWScore
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "NSFWScore": "NSFW Score"
}
