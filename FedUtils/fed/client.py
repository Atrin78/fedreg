from FedUtils.models.utils import CusDataset
from torch.utils.data import DataLoader, ConcatDataset
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from PIL import Image
import numpy as np


class Client(object):
    def __init__(self, id, group, train_data, eval_data, model, batchsize, train_transform=None, test_transform=None, traincusdataset=None, evalcusdataset=None):
        super(Client, self).__init__()
        self.model = model
        self.id = id
        self.group = group
        self.train_samplenum = len(train_data["x"])
        self.num_train_samples = len(train_data["x"])
        self.num_test_samples = [len(ed["x"]) for ed in eval_data]
        drop_last = False
        self.gen_data = None
        self.drop_last = drop_last
        self.batchsize = batchsize
        if traincusdataset:  # load data use costomer's dataset
            self.train_dataset = traincusdataset(train_data, transform=train_transform)
            self.train_data = DataLoader(traincusdataset(train_data, transform=train_transform), batch_size=batchsize, shuffle=True, drop_last=drop_last)
            self.train_data_fortest = DataLoader(evalcusdataset(train_data, transform=test_transform), batch_size=batchsize, shuffle=False,)
            num_workers = 0
            self.eval_data = [DataLoader(evalcusdataset(ed, transform=test_transform), batch_size=100, shuffle=False, num_workers=num_workers) for ed in eval_data]
        else:
            self.train_dataset = CusDataset(train_data, transform=train_transform)
            self.train_data = DataLoader(CusDataset(train_data, transform=train_transform), batch_size=batchsize, shuffle=True, drop_last=drop_last)
            self.train_data_fortest = DataLoader(CusDataset(train_data, transform=test_transform), batch_size=batchsize, shuffle=False)
            self.eval_data = [DataLoader(CusDataset(ed, transform=test_transform), batch_size=100, shuffle=False) for ed in eval_data]
        self.train_iter = iter(self.train_data)

    def set_param(self, state_dict):
        self.model.set_param(state_dict)
        return True

    def get_param(self):
        return self.model.get_param()

    def solve_grad(self):
        bytes_w = self.model.size
        grads, comp = self.model.get_gradients(self.train_data)
        bytes_r = self.model.size
        return ((self.num_train_samples, grads), (bytes_w, comp, bytes_r))

    def solve_inner(self, num_epochs=1, step_func=None):
        bytes_w = self.model.size
   #     if self.gen_data is None:
   #         training = self.train_dataset
   #     else: 
   #         training = ConcatDataset([self.train_dataset, self.gen_data])
   #         training = self.gen_data
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batchsize, shuffle=True, drop_last=self.drop_last)
        if self.gen_data is None:
            data_loaders = [train_dataloader]
        else:
            gen_dataloader = DataLoader(self.gen_data, batch_size=self.batchsize*5, shuffle=True, drop_last=self.drop_last)
            data_loaders = [train_dataloader, gen_dataloader]
        for d in iter(train_dataloader):
            x, y = d
         #   out = self.model.AE(x)[0].cpu().detach().numpy()*255
            out = x
            out = out.reshape((32, 32))
            print(np.max(out))
            im = Image.fromarray(out.astype('uint8'))
            im.save(str(self.id)+'.jpeg')
            break
        soln, comp, weight = self.model.solve_inner(data_loaders, num_epochs=num_epochs, step_func=step_func)
        bytes_r = self.model.size
        return (self.num_train_samples*weight, soln), (bytes_w, comp, bytes_r)

    def test(self):
        TC = []
        LS = []
        for ed in self.eval_data:
            total_correct, loss = self.model.test(ed)
            TC.append(total_correct)
            LS.append(loss)
        return TC,  self.num_test_samples

    def train_error_and_loss(self):
        tot_correct, loss = self.model.test(self.train_data_fortest)
        return tot_correct, loss, self.train_samplenum

    def testAE(self):
        TC = []
        LS = []
        for ed in self.eval_data:
            total_correct, loss = self.model.testAE(ed)
            TC.append(total_correct)
            LS.append(loss)
        return TC,  self.num_test_samples

    def train_error_and_lossAE(self):
        tot_correct, loss = self.model.testAE(self.train_data_fortest)
        return tot_correct, loss, self.train_samplenum

