import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader


class ChannelsData(torch.utils.data.Dataset):
    def __init__(
        self, channels_res: np.ndarray, y_res: np.ndarray, type_data: str = "train"
    ):
        self.len_dataset = channels_res.shape[0]

        self.transform_channel = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.transform_y_signal = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        ratio = 4 / 5
        if type_data == "train":
            self.channels_res = channels_res[: int(self.len_dataset * ratio)]
            self.y_res = y_res[: int(self.len_dataset * ratio)]
        elif type_data == "eval":
            self.channels_res = channels_res[int(self.len_dataset * ratio) :]
            self.y_res = y_res[int(self.len_dataset * ratio) :]
            self.channels_res = self.channels_res[: int(len(self.channels_res) / 2)]
            self.y_res = self.y_res[: int(len(self.y_res) / 2)]
        else:
            self.channels_res = channels_res[int(self.len_dataset * ratio) :]
            self.y_res = y_res[int(self.len_dataset * ratio) :]
            self.channels_res = self.channels_res[int(len(self.channels_res) / 2) :]
            self.y_res = self.y_res[int(len(self.y_res) / 2) :]

    def preprocess_response(self, y_res):
        normalized_y_res = np.zeros_like(y_res)
        for i in range(len(y_res)):
            y_res_i = y_res[i]
            y_res_i = (y_res_i - y_res_i.min()) / (y_res_i.max() - y_res_i.min())
            y_res_i = 2 * y_res_i - 1
            normalized_y_res[i] = y_res_i

        normalized_y_res = normalized_y_res.reshape(32, 32, 2).transpose(1, 0, 2)
        return normalized_y_res

    def preprocess_chanel(self, channel_res):
        normalized_channel_res = np.zeros_like(channel_res)
        for i in range(len(channel_res)):
            channel_res_i = channel_res[i]
            channel_res_i = (channel_res_i - channel_res_i.min()) / (
                channel_res_i.max() - channel_res_i.min()
            )
            channel_res_i = 2 * channel_res_i - 1
            normalized_channel_res[i] = channel_res_i

        normalized_channel_res = normalized_channel_res.reshape(32, 32, 2).transpose(
            1, 0, 2
        )
        return normalized_channel_res

    def __getitem__(self, item):
        channel_res = self.channels_res[item]
        y_res = self.y_res[item]

        channel_res = self.preprocess_chanel(channel_res)
        y_res = self.preprocess_response(y_res)

        item_channel_res = torch.FloatTensor(channel_res).permute(2, 0, 1)
        item_y_res = torch.FloatTensor(y_res).permute(2, 0, 1)

        return item_channel_res, item_y_res

    def __len__(self):
        return len(self.channels_res)


class ChannelDataloader:
    def __init__(
        self,
        channels_res: np.ndarray,
        y_res: np.ndarray,
        batch_size: int = 64,
        pin_memory: bool = False,
        num_worker: int = 1,
    ):
        self.channels_res = channels_res
        self.y_res = y_res
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_worker = num_worker

    def train_dataloader(self):
        return DataLoader(
            ChannelsData(
                channels_res=self.channels_res,
                y_res=self.y_res,
                type_data="train",
            ),
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=True,
            num_workers=self.num_worker,
        )

    def test_dataloader(self):
        return DataLoader(
            ChannelsData(
                channels_res=self.channels_res,
                y_res=self.y_res,
                type_data="test",
            ),
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_worker,
            shuffle=False,
        )

    def eval_dataloader(self):
        return DataLoader(
            ChannelsData(
                channels_res=self.channels_res,
                y_res=self.y_res,
                type_data="eval",
            ),
            batch_size=1,
            pin_memory=self.pin_memory,
            num_workers=self.num_worker,
            shuffle=True,
        )


class ChannelsDataGAN(torch.utils.data.Dataset):
    def __init__(
        self, channels_res: np.ndarray, y_res: np.ndarray, type_data: str = "train"
    ):
        self.len_dataset = channels_res.shape[0]

        self.transform_channel = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.transform_y_signal = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        ratio = 4 / 5
        if type_data == "train":
            self.channels_res = channels_res[: int(self.len_dataset * ratio)]
            self.y_res = y_res[: int(self.len_dataset * ratio)]
        elif type_data == "eval":
            self.channels_res = channels_res[int(self.len_dataset * ratio) :]
            self.y_res = y_res[int(self.len_dataset * ratio) :]
            self.channels_res = self.channels_res[: int(len(self.channels_res) / 2)]
            self.y_res = self.y_res[: int(len(self.y_res) / 2)]
        else:
            self.channels_res = channels_res[int(self.len_dataset * ratio) :]
            self.y_res = y_res[int(self.len_dataset * ratio) :]
            self.channels_res = self.channels_res[int(len(self.channels_res) / 2) :]
            self.y_res = self.y_res[int(len(self.y_res) / 2) :]

    def preprocess_response(self, y_res):

        normalized_y_res = np.zeros_like(y_res)
        for i in range(len(y_res)):
            y_res_i = y_res[i]
            y_res_i = (y_res_i - y_res_i.min()) / (y_res_i.max() - y_res_i.min())
            y_res_i = 2 * y_res_i - 1
            normalized_y_res[i] = y_res_i

        normalized_y_res = normalized_y_res.reshape(32, 32, 2).transpose(1, 0, 2)
        normalized_y_res = normalized_y_res.T.flatten()
        return normalized_y_res

    def preprocess_chanel(self, channel_res):

        normalized_channel_res = np.zeros_like(channel_res)
        for i in range(len(channel_res)):
            channel_res_i = channel_res[i]
            channel_res_i = (channel_res_i - channel_res_i.min()) / (
                channel_res_i.max() - channel_res_i.min()
            )
            channel_res_i = 2 * channel_res_i - 1
            normalized_channel_res[i] = channel_res_i

        normalized_channel_res = normalized_channel_res.reshape(32, 32, 2).transpose(
            1, 0, 2
        )
        return normalized_channel_res

    def __getitem__(self, item):
        channel_res = self.channels_res[item]
        y_res = self.y_res[item]

        channel_res = self.preprocess_chanel(channel_res)
        y_res = self.preprocess_response(y_res)

        item_channel_res = torch.FloatTensor(channel_res).permute(2, 0, 1)
        item_y_res = torch.FloatTensor(y_res)

        return item_channel_res, item_y_res

    def __len__(self):
        return len(self.channels_res)


class ChannelDataloaderGAN:
    def __init__(
        self,
        channels_res: np.ndarray,
        y_res: np.ndarray,
        batch_size: int = 64,
        pin_memory: bool = False,
        num_worker: int = 1,
    ):
        self.channels_res = channels_res
        self.y_res = y_res
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_worker = num_worker

    def train_dataloader(self):
        return DataLoader(
            ChannelsDataGAN(
                channels_res=self.channels_res,
                y_res=self.y_res,
                type_data="train",
            ),
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=True,
            num_workers=self.num_worker,
        )

    def test_dataloader(self):
        return DataLoader(
            ChannelsDataGAN(
                channels_res=self.channels_res,
                y_res=self.y_res,
                type_data="test",
            ),
            batch_size=1,
            pin_memory=self.pin_memory,
            num_workers=self.num_worker,
            shuffle=False,
        )

    def eval_dataloader(self):
        return DataLoader(
            ChannelsDataGAN(
                channels_res=self.channels_res,
                y_res=self.y_res,
                type_data="eval",
            ),
            batch_size=1,
            pin_memory=self.pin_memory,
            num_workers=self.num_worker,
            shuffle=True,
        )
