import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import dgl
from dgl.data import citation_graph as citegrh
from dgl.nn.pytorch import GATConv

class GCN(pl.LightningModule):
    def __init__(self, g, in_feats, hidden_feats, num_classes, num_heads):
        super(GCN, self).__init__()
        self.g = g
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads=num_heads)
        self.conv2 = GATConv(hidden_feats * num_heads, num_classes, num_heads=1)

    def forward(self, inputs):
        h = inputs
        h = F.elu(self.conv1(self.g, h).flatten(1))
        h = self.conv2(self.g, h).mean(1)
        return h

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

# Load the Cora dataset
data = citegrh.load_cora()

# Create a DGL graph from the dataset
g = dgl.DGLGraph(data.graph)
g = g.remove_self_loop().add_self_loop()

# Create a PyTorch Lightning dataloader
train_loader = torch.utils.data.DataLoader(list(zip(g.edges()[0], g.edges()[1])), batch_size=64, shuffle=True)

# Initialize the model
model = GCN(g, data.features.shape[1], 16, data.num_labels, 4)

# Train the model using PyTorch Lightning Trainer
trainer = pl.Trainer(gpus=1, max_epochs=10)
trainer.fit(model, train_loader)