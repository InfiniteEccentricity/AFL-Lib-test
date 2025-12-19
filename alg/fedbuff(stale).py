import torch

from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, Status
from utils.time_utils import time_record


def add_args(parser):
    parser.add_argument('--etag', type=float, default=5)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    return parser.parse_args()


class Client(AsyncBaseClient):
    @time_record
    def run(self):
        w_last = self.model2tensor()
        self.train()
        self.dW = self.model2tensor() -  w_last


class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.buffer = []
        self.weight_buffer = []

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()
        self.update_status()

    def aggregate(self):
        s = self.staleness[self.cur_client.id]
        
        # 2. Calculate the scaling weight (Polynomial Scaling is most common)
        # weight = (1 + s)^-alpha
        weight = (1 + s) ** (-self.args.alpha)
        self.buffer.append(self.cur_client.dW)
        self.weight_buffer.append(weight)
        if len(self.buffer) == self.args.k:
            t_g = self.model2tensor()
            
            # 4. Use a weighted average instead of a simple mean
            # Instead of torch.mean(buffer), we do sum(weighted_updates) / sum(weights)
            weighted_update_sum = torch.stack(self.buffer).sum(dim=0)
            total_weight = sum(self.weight_buffer)
            
            # Update global model: W = W + etag * (weighted_average)
            t_g_new = t_g + self.args.etag * (weighted_update_sum / total_weight)
            self.tensor2model(t_g_new)

            # 5. Clear both buffers
            self.buffer = []
            self.weight_buffer = []
