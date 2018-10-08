import torchvision.transforms as transforms
from scipy.misc import imread, imresize
from torch.utils.data import Dataset

from utils import *


class ZslDataset(Dataset):
    def __init__(self, superclass, split):
        self.superclass = superclass
        self.split = split
        assert self.superclass in {'Animals', 'Fruits', 'Vehicles', 'Electronics', 'Hairstyles'}
        assert self.split in {'train', 'valid'}

        self.image_folder, annotations_labels, annotations_attributes_per_class, _ = get_annotations_by_superclass(
            superclass)
        # print('self.image_folder: ' + str(self.image_folder))
        self.label_name2idx = get_label_name2idx_by_superclass(superclass)

        annotations_labels = pd.read_csv(annotations_labels, header=None, usecols=[1, 6])
        annotations_labels.columns = ['label_name', 'img_path']
        annotations_labels['label_name'] = annotations_labels['label_name'].str.strip()
        annotations_labels['img_path'] = annotations_labels['img_path'].str.strip()
        attributes_per_class = pd.read_csv(annotations_attributes_per_class, header=None)
        attributes_per_class.columns = ['label_name', 'attributes']
        attributes_per_class['attributes'] = attributes_per_class['attributes'].str.strip()

        samples = pd.merge(annotations_labels, attributes_per_class, on='label_name')
        # Shuffle DataFrame rows
        samples = samples.sample(frac=1).reset_index(drop=True)
        train_count = int(len(samples) * train_split)

        if split == 'train':
            self.samples = samples[:train_count]
            self.start_index = 0
        else:
            self.samples = samples[train_count:]
            self.start_index = train_count

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([normalize])

    def __getitem__(self, i):
        img_path = self.samples['img_path'][self.start_index + i]
        if self.superclass in ['Vehicles', 'Electronics', 'Hairstyles']:
            img_path = img_path[8:]
        path = os.path.join(self.image_folder, img_path)
        # Read images
        img = imread(path, flatten=False, mode='RGB')
        img = imresize(img, (224, 224))
        # print(img.shape)
        img = img.transpose(2, 0, 1)
        assert img.shape == (3, 224, 224)
        assert np.max(img) <= 255
        img = torch.FloatTensor(img / 255.)
        if self.transform is not None:
            img = self.transform(img)

        label_name = self.samples['label_name'][self.start_index + i]
        # print('label_name: ' + str(label_name))
        label_id = self.label_name2idx[label_name]
        # print('label_id: ' + str(label_id))
        label_id = torch.IntTensor([label_id])
        # print('label_id: ' + str(label_id))

        attribute = parse_attributes(self.samples['attributes'][self.start_index + i])
        attribute = torch.FloatTensor(attribute)

        return img, label_id, attribute

    def __len__(self):
        return self.samples.shape[0]
