from FedUtils.models.utils import CusDataset, feature_space_linear_cka, cka, gram_linear
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from loguru import logger
from torch_cka import CKA


class Client(object):
    def __init__(self, id, train_data, eval_data,model, batchsize):
        super(Client, self).__init__()
        self.rotate=False
        self.model = model
        self.id = id
        self.group = None
        self.train_samplenum = train_data["datasize"] 
        self.num_train_samples = train_data["datasize"] 
        self.num_test_samples = [eval_data["test_datasize"]]
        drop_last = False
        self.drop_last = drop_last
        self.gen_data = None
        self.batchsize = batchsize
        self.train_data = train_data["train"]
        self.train_data_fortest = train_data["train_for_test"]
        self.eval_data = [eval_data["test"]]
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

        if self.rotate:
            x_rot = None
            y_rot = None
            for d in train_dataloader:
                x, y = d
                x = torch.reshape(torch.rot90(torch.reshape(x, (-1, 28, 28)), np.random.choice([1, 3, 4]), [1, 2]), (-1, 784))
                if x_rot is None:
                    x_rot = x
                    y_rot = y
                else:
                    x_rot = torch.cat((x_rot, x), 0)
                    y_rot = torch.cat((y_rot, y), 0)
            train_dataloader = DataLoader(TensorDataset(x_rot, y_rot), batch_size=self.batchsize, shuffle=True, drop_last=self.drop_last)
                    
        if self.gen_data is None:
            data_loaders = [self.train_data]
        else:
            gen_dataloader = DataLoader(self.gen_data, batch_size=self.batchsize, shuffle=True, drop_last=self.drop_last)
            data_loaders = [self.train_data, gen_dataloader]
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
    
    
    def get_cka(self, global_model):

        cka_model= CKA(self.model, global_model,
        model1_name="local_model",   # good idea to provide names to avoid confusion
        model2_name="global_model",   
        model1_layers=['bottleneck.1'], # List of layers to extract features from
        model2_layers= ['bottleneck.1'], # extracts all layer features by default
        device='cuda')
        cka_model.compare(self.train_data_fortest) # secondary dataloader is optional

        results = cka_model.export()  # returns a dict that contains model names, layer names
                        # and the CKA matrix
        
        logger.info(f"CKA: {results}")

        return cka_value

    def train_error_and_loss(self):
        tot_correct, loss = self.model.test(self.train_data_fortest)
        return tot_correct, loss, self.train_samplenum