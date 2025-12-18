import torch
import numpy as np # Ensure numpy is imported for math if needed

def add_args(parser):
    parser.add_argument('--etag', type=float, default=5)
    parser.add_argument('--k', type=int, default=10)
    # Add alpha for the scaling intensity
    parser.add_argument('--alpha', type=float, default=0.5) 
    return parser.parse_args()

class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.buffer = []
        self.stale_weights = [] # Track weights for each item in buffer

    def aggregate(self):
        # 1. Calculate staleness weight for the incoming client
        # s = current_server_staleness
        s = self.staleness[self.cur_client.id]
        
        # Polynomial scaling: weight = (s + 1)^-alpha
        weight = (s + 1) ** (-self.args.alpha)
        
        # 2. Add both the update and its weight to the buffer
        self.buffer.append(self.cur_client.dW * weight)
        self.stale_weights.append(weight)

        # 3. Only update when buffer is full
        if len(self.buffer) == self.args.k:
            t_g = self.model2tensor()
            
            # Weighted average calculation
            # Sum of (dW * weight) / Sum of weights
            sum_updates = torch.stack(self.buffer).sum(dim=0)
            total_weight = sum(self.stale_weights)
            
            t_g_new = t_g + self.args.etag * (sum_updates / total_weight)
            self.tensor2model(t_g_new)

            # Clear buffers
            self.buffer = []
            self.stale_weights = []
