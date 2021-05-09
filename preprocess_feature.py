from data_loader import VideoImageData
import os
from torch.utils.data import Dataset, DataLoader
import h5py
import torch
from tqdm import tqdm
from feature_extraction import ResNetFeature
import sys
def main(root = "", save_path = "./out.h5"):
    hf = h5py.File(save_path, 'a')
    images_group = hf.create_group("pool5")
    model = ResNetFeature()
    for image_dir in os.listdir(root):
        image_group = images_group.create_group(image_dir)
        image_dir_path = os.path.join(root, image_dir)
        batch_size = 32
        loader = DataLoader(VideoImageData(image_dir_path), batch_size=batch_size)
        for i, batch in enumerate(tqdm(loader, desc="process: ")):

            for j in range(batch_size):
                features = model(batch)
                image_t = batch[j, : , :, :]
                print(image_t.size())
                image_t = torch.squeeze(image_t)
                print(image_t.size())
                image_data = image_t.numpy()
                count = i * batch_size + j
                break
                dset = image_group.create_dataset(str(count), data=image_data)


if __name__ == '__main__':
    main(sys.argv[1])