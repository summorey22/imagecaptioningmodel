from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, AutoFeatureExtractor, AutoTokenizer, TrainingArguments, Trainer
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, Resize
import cv2

import warnings
warnings.filterwarnings('ignore')

gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')
feature_extractor = AutoFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
finetuned_model = VisionEncoderDecoderModel.from_pretrained('./torch_model')

inference_compose = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(
    mean = feature_extractor.image_mean,
    std = feature_extractor.image_std
)
])


def caption_image(m, path):
  img = Image.open(path)
  image_matrix = inference_compose(img).unsqueeze(0)

  generated = m.generate(
      image_matrix,
      num_beams = 3,
      max_length = 15,
      early_stopping = True,
      do_sample = True,
      top_k = 10,
      num_return_sequences = 5,
  )
  caption_options = [gpt2_tokenizer.decode(g, skip_special_tokens = True).strip() for g in generated]
  print(caption_options)
  return caption_options, generated, image_matrix

def getImage():
    cam = cv2.VideoCapture(0)
    result, image = cam.read()
    if result:
        cv2.imwrite("./temp.jpg", image)
        return './temp.jpg'
    else:
       print('Please check your camera configuration')

c, g, i = caption_image(
    finetuned_model,
    getImage()
)




