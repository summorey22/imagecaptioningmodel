from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, AutoFeatureExtractor, AutoTokenizer, TrainingArguments, Trainer
from PIL import Image
from datasets import Dataset
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, Resize
import pandas as pd
import json
import argparse
import os
from datetime import datetime
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
                    prog='train.py',
                    description='Trains a model to caption images',
)

parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train for')
parser.add_argument('--p', type=str, default='./data/train2017', help='Path to the training data')
parser.add_argument('--a', type=str, default='./data/annotations_trainval2017/annotations/captions_train2017.json', help='Path to the training annotations file')
parser.add_argument('--encode', type=str, default='google/vit-base-patch16-224-in21k', help='Encoder model to use')
parser.add_argument('--decode', type=str, default='distilgpt2', help='Decoder model to use')
parser.add_argument('--samples', type=int, default=1000, help='Number of images to train on')
parser.add_argument('--size', type=int, default=64, help='Batch size for training')
parser.add_argument('--o', type=str, help='Path to save the model')
args = parser.parse_args()

# Define the constants
IMAGES_PATH = args.p
ANN_PATH = args.a
MIN_CAPTION, MAX_CAPTION = 10, 50
ENCODER = args.encode
DECODER = args.decode
EPOCHS = args.epochs
SIZE = args.size
N_SAMPLES = args.samples
SAVE_PATH = args.o or f'./ckpts_{datetime.now().strftime("%d%m%Y%H%M%S")}'

print('Downloading Pretrained Models')
# Define the model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    ENCODER,
    DECODER
)

# Define the tokenizer
print('Downloading Tokenizers') 
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(DECODER)

# Define the feature extractor
print('Downloading Feature Extractor')
feature_extractor = AutoFeatureExtractor.from_pretrained(ENCODER)

# Load the annotations file
with open(os.path.join(ANN_PATH), 'r') as f:
    annotations = json.load(f)

print('Creating DataFrame')
# Iterate through the annotations and group the captions by image ID
captions_by_image_id = {}
for annotation in annotations['annotations']:
    image_id = annotation['image_id']
    caption = annotation['caption']
    if image_id in captions_by_image_id:
        captions_by_image_id[image_id].append(caption)
    else:
        captions_by_image_id[image_id] = [caption]

# Iterate through the images and add their paths and captions to the dictionary
rows = []
count = 0
for image in annotations['images']:
    image_id = image['id']
    filename = os.path.join(IMAGES_PATH, image['file_name'])
    print(SIZE)
    if image_id in captions_by_image_id:
        captions = captions_by_image_id[image_id]
        for k in captions:
          if len(k) < MIN_CAPTION or len(caption) > MAX_CAPTION:
            continue
          data = {}
          data['path'] = filename
          data['caption'] = k
          rows.append(data)
          break
    count += 1
    if count == N_SAMPLES:
      break

# Create the DataFrame
image_df = pd.DataFrame(rows)


# Define the transforms
normalize = Normalize(
    mean = feature_extractor.image_mean,
    std = feature_extractor.image_std
)

_transforms = Compose([
    RandomResizedCrop(224),
    ToTensor(),
    normalize
])

# Create the dataset
image_dataset = Dataset.from_pandas(image_df)

# Set the special tokens
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# Define the preprocessing function
def image_preprocess(examples):
  examples['pixel_values'] = [_transforms(Image.open(path).convert('RGB')) for path in examples['path']]
  tokenized = gpt2_tokenizer(
      examples['caption'],
      padding = 'max_length',
      max_length = 10,
      truncation = True
  )['input_ids']
  examples['labels'] = [[l if l != gpt2_tokenizer.pad_token_id else -100 for l in t] for t in tokenized]
  del examples['path']
  del examples['caption']
  return examples

print('Preprocessing Data')
# Preprocess the dataset
image_dataset = image_dataset.map(image_preprocess, batched=True)

print('Splitting Data')
# Split the dataset into training and testing sets
image_dataset = image_dataset.train_test_split(test_size = 0.1)

# Set the special tokens
model.config.pad_token = gpt2_tokenizer.pad_token
model.config.pad_token_id = gpt2_tokenizer.pad_token_id

model.config.decoder_start_token = gpt2_tokenizer.bos_token
model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id

# Freeze the encoder layers which perform general feature extraction
for name, param in model.encoder.named_parameters():
  if 'encoder.layer.10' in name:
    break
  param.requires_grad = False

# Create the TrainingArguments
training_args = TrainingArguments(
    output_dir = SAVE_PATH,
    overwrite_output_dir = True,
    num_train_epochs = EPOCHS,
    per_device_train_batch_size = 64,
    per_device_eval_batch_size = 64,
    load_best_model_at_end = True,
    log_level = 'info',
    logging_steps = 50,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
)

# Create the Trainer instance
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = image_dataset['train'],
    eval_dataset = image_dataset['test']
)

# Training the model
print('Evaluating the model')
trainer.evaluate()

print('Training the model')
trainer.train()

print('Saving the model')
trainer.save_model()
print(f'Model saved at {SAVE_PATH}')