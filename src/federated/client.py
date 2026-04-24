import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

class Client:
    def __init__(self, dataset, idxs, config):
        self.config = config
        self.device = config["device"]
        self.train_loader = DataLoader(Subset(dataset, list(idxs)), batch_size=config["batch_size"], shuffle=True)

    def train(self, model):
        model.to(self.device)
        model.train()
        momentum = float(self.config.get("optimizer_momentum", 0.0))
        weight_decay = float(self.config.get("weight_decay", 0.0))
        nesterov = self.config.get("client_nesterov", False)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config["lr"],
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov if momentum > 0 else False
        )
        epoch_loss = []
        num_samples = len(self.train_loader.dataset)
        if num_samples == 0:
            return model.state_dict(), 0.0, 0
            
        for epoch in range(self.config["local_epochs"]):
            batch_loss = []
            for data, target in self.train_loader:
                if data.size(0) == 0:
                    continue
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected for client. Skipping batch.")
                    continue
                    
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            else:
                epoch_loss.append(0.0)
                
        avg_loss = sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0
        return model.state_dict(), avg_loss, num_samples
