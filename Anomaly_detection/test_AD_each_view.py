import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import glob
from PIL import Image
import pytorch_lightning as pl
import pickle
from sampling_methods.kcenter_greedy import kCenterGreedy
from sklearn.random_projection import SparseRandomProjection
from scipy.ndimage import gaussian_filter
from torchvision import transforms, models


def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors

    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)

    return dist


class NN():

    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]


class KNN(NN):

    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)

        knn = dist.topk(self.k, largest=False)

        return knn


def prep_dirs(root):
    # make embeddings dir
    # embeddings_path = os.path.join(root, 'embeddings')
    embeddings_path = os.path.join('./', 'embeddings', args.category)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    return embeddings_path, sample_path, source_code_save_path


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list


# imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]


class MyDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase,position):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        self.mode = phase
        self.position = position
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        img_paths = glob.glob(data_originpath + '*'+self.position + '.png')


        img_tot_paths.extend(img_paths)
        gt_tot_paths.extend([0] * len(img_paths))
        tot_labels.extend([0] * len(img_paths))
        tot_types.extend(['good'] * len(img_paths))


        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type




def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)



class STPM(pl.LightningModule):
    def __init__(self, hparams):
        super(STPM, self).__init__()

        self.save_hyperparameters(hparams)

        self.init_features()

        def hook_t(module, input, output):
            self.features.append(output)

        self.model = models.wide_resnet50_2(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.init_results_list()

        self.data_transforms = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size), Image.ANTIALIAS),
            transforms.ToTensor(),
            # transforms.CenterCrop(args.input_size),
            transforms.Normalize(mean=mean_train,
                                 std=std_train)])
        self.gt_transforms = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor()])

        self.inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                                  std=[1 / 0.229, 1 / 0.224, 1 / 0.255])

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []
        self.testfilename = []

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features


    def test_dataloader(self):
        test_datasets = MyDataset(root=os.path.join(args.dataset_path, args.category), transform=self.data_transforms,
                                  gt_transform=self.gt_transforms, phase='test', position=positions)
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False,
                                 num_workers=0)  # , pin_memory=True) # only work on batch_size=1, now.
        return test_loader

    def configure_optimizers(self):
        return None

    def on_test_start(self):
        self.init_results_list()
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)


    def test_step(self, batch, batch_idx):  # Nearest Neighbour Search
        self.embedding_coreset = pickle.load(
            open(os.path.join('./embeddings', 'embedding_' + positions + '_sample.pickle'), 'rb'))
        x, gt, label, file_name, x_type = batch
        # extract embedding
        features = self(x)
        self.testfilename.append(file_name[0])
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_)))

        knn = KNN(torch.from_numpy(self.embedding_coreset).cuda(), k=9)
        score_patches = knn(torch.from_numpy(embedding_test).cuda())[0].cpu().detach().numpy()

        anomaly_map = score_patches[:, 0].reshape((28, 28))
        N_b = score_patches[np.argmax(score_patches[:, 0])]


        w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
        # w = 1
        score = w * max(score_patches[:, 0])  # Image-level score

        gt_np = gt.cpu().numpy()[0, 0].astype(int)
        anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size, args.input_size))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        np.save(savedir + file_name[0].split('/')[-1].replace('png', 'npy'), anomaly_map_resized_blur)
        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        self.img_path_list.extend(file_name)


    def test_epoch_end(self, outputs):

        print('test_epoch_end')



def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train', 'test'], default='test')
    parser.add_argument('--dataset_path',
                        default=r'./')  # 'D:\Dataset\mvtec_anomaly_detection')#
    parser.add_argument('--category', default='')
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=256)  # 256
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--project_root_path',
                        default=r'../result/')
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()

    for position in ['l', 'r']:
        for proj_axis in ['c','s','a']:
            positions = position+'_' + proj_axis
            savedir = '../result/cammap/'
            os.makedirs(savedir, exist_ok=True)
            data_originpath = '../projection_data/'
            trainer = pl.Trainer.from_argparse_args(args,
                                                    default_root_dir=os.path.join(args.project_root_path,
                                                    args.category),
                                                    max_epochs=args.num_epochs,
                                                    gpus=1)
            model = STPM(hparams=args)
            trainer.test(model)

