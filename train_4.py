from pathlib import Path

import torch
import wandb
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanMetric, AUROC
from tqdm import tqdm

from dataset import LoanCollator, load_loan
from model import LoanModel

wandb.login()


def _map_to_device(batch: dict, dev: torch.device) -> None:
    batch['target'] = batch['target'].to(dev)

    for i in batch['cat_features']:
        batch['cat_features'][i] = batch['cat_features'][i].to(dev)

    for i in batch['numeric_features']:
        batch['numeric_features'][i] = batch['numeric_features'][i].to(dev)


class Config:
    def __init__(self):
        self.version = 4
        self.dev = torch.device('cuda:0')
        self.n_epochs = 10
        self.lr = 0.01
        self.base_hidden_size = 128
        self.batch_size = 64
        self.seed = 42
        self.weight_decay = 0
        self.dropout_p = 0


def train(train_dataset: Dataset, eval_dataset: Dataset, config: Config):
    version = config.version
    dev = config.dev
    n_epochs = config.n_epochs
    lr = config.lr
    base_hidden_size = config.base_hidden_size
    batch_size = config.batch_size
    seed = config.seed
    weight_decay = config.weight_decay
    dropout_p = config.dropout_p

    torch.random.manual_seed(seed)

    loss_bce = BCEWithLogitsLoss()

    collator = LoanCollator()
    model = LoanModel(hidden_size=base_hidden_size, version=version, dropout_p=dropout_p).to(dev)
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    run = wandb.init(
        project='hw_dl_1',
        name=f'exp_{version}_p={dropout_p}',
        config={
            'version': version,
            'n_epochs': n_epochs,
            "learning_rate": lr,
            "base_hidden_size": base_hidden_size,
            "batch_size": batch_size,
            'weight_decay': weight_decay,
            'dropout_p': dropout_p
        },
    )

    train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, collate_fn=collator, pin_memory=True)
    eval_dl = DataLoader(eval_dataset, batch_size=batch_size, num_workers=8, collate_fn=collator, pin_memory=True)

    for i_epoch in tqdm(range(n_epochs)):
        train_loss = MeanMetric().to(dev)
        train_rocauc = AUROC(task='binary').to(dev)
        for i, batch in enumerate(train_dl):
            if dev == torch.device('cuda:0'):
                _map_to_device(batch, dev)

            result = model(cat_features=batch['cat_features'], numeric_features=batch['numeric_features'])
            loss_value = loss_bce(result, batch['target'])
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.update(loss_value)
            train_rocauc.update(torch.sigmoid(result), batch['target'])

        train_loss = train_loss.compute().item()
        train_rocauc = train_rocauc.compute().item()

        run.log({'train_loss': train_loss, 'train_rocauc': train_rocauc}, step=i_epoch)

        eval_loss = MeanMetric().to(dev)
        eval_rocauc = AUROC(task='binary').to(dev)

        model.eval()
        with torch.no_grad():
            for i_eval, batch_eval in enumerate(eval_dl):
                if dev == torch.device('cuda:0'):
                    _map_to_device(batch_eval, dev)

                result_eval = model(cat_features=batch_eval['cat_features'], numeric_features=batch_eval['numeric_features'])
                eval_loss_value = loss_bce(result_eval, batch_eval['target'])

                eval_loss.update(eval_loss_value)
                eval_rocauc.update(torch.sigmoid(result_eval), batch_eval['target'])
        model.train()

        eval_loss = eval_loss.compute().item()
        eval_rocauc = eval_rocauc.compute().item()
        print(f'\nEpoch {i_epoch}, train/loss: {train_loss}, train/aucroc: {train_rocauc}, '
              f'val/loss: {eval_loss}, val/aucroc: {eval_rocauc}\n')
        run.log({'eval_loss': eval_loss, 'eval_rocauc': eval_rocauc}, step=i_epoch)

    run.finish()


if __name__ == '__main__':
    train_ds, test_ds = load_loan(Path('loan_train.csv'), Path('loan_test.csv'))
    #dropout_values = [0.1, 0.2, 0.5]
    dropout_values = [0.1, 0.9]

    for cur_dropout_p in dropout_values:
        config_v4 = Config()
        config_v4.dropout_p = cur_dropout_p
        train(train_ds, test_ds, config_v4)
