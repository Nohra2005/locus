import kagglehub

print("Downloading DeepFashion dataset... this might take a few minutes...")

# Download latest version
path = kagglehub.dataset_download("hserdaraltan/deepfashion-inshop-clothes-retrieval")

print("✅ Download and extraction complete!")
print("Path to dataset files:", path)