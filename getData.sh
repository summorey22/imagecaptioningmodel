# Create a folder called data if it doesn't exist
mkdir -p data

# Download train images
wget http://images.cocodataset.org/zips/train2017.zip -O data/train2017.zip

# Extract train images
unzip -q data/train2017.zip -d data

# Delete that zip
rm data/train2017.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/annotations_trainval2017.zip

# Extract annotations
unzip -q data/annotations_trainval2017.zip -d data

# Delete that zip
rm data/annotations_trainval2017.zip