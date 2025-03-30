import torch
from torch import nn, Tensor


class BaseBlock(nn.Module):
    def __init__(self, hidden_size: int, version=1):
        super().__init__()
        self.version = version
        if self.version == 1:
            self.linear_1 = nn.Linear(hidden_size, hidden_size * 4)
            self.act = nn.ReLU()
            self.linear_2 = nn.Linear(hidden_size * 4, hidden_size)

        # self.bn = nn.BatchNorm1d(hidden_size)
        # self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.bn(x)
        if self.version == 1:
            x = self.linear_1(x)
            x = self.act(x)
            x = self.linear_2(x)
        return x


class LoanModel(nn.Module):
    def __init__(self, hidden_size: int, version=1):
        super().__init__()
        self.version = version
        self.emb_owner = nn.Embedding(4,embedding_dim=hidden_size)
        self.emb_intent = nn.Embedding(6, embedding_dim=hidden_size)
        self.emb_grade = nn.Embedding(7, embedding_dim=hidden_size)
        self.emb_cb_default = nn.Embedding(2, embedding_dim=hidden_size)

        self.numeric_linear = nn.Linear(7, hidden_size)

        self.block_1 = BaseBlock(hidden_size, version)
        if version > 1:
            self.block_1 = BaseBlock(hidden_size, version)
            self.block_2 = BaseBlock(hidden_size, version)
            self.block_3 = BaseBlock(hidden_size, version)

        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, cat_features: dict[str, Tensor], numeric_features: dict[str, Tensor]) -> Tensor:
        x_owner = self.emb_owner(cat_features['person_home_ownership'])
        x_intent = self.emb_intent(cat_features['loan_intent'])
        x_grade = self.emb_grade(cat_features['loan_grade'])
        x_cb_default = self.emb_cb_default(cat_features['cb_person_default_on_file'])

        stacked_numeric = torch.stack([
            numeric_features['person_age'],
            numeric_features['person_income'],
            numeric_features['person_emp_length'],
            numeric_features['loan_amnt'],
            numeric_features['loan_int_rate'],
            numeric_features['loan_percent_income'],
            numeric_features['cb_person_cred_hist_length']
        ],
          dim=-1)
        x_numeric = self.numeric_linear(stacked_numeric)

        x_total = x_owner + x_intent + x_grade + x_cb_default + x_numeric

        if self.version == 1:
            #print('arch = 1 block')
            x_total = self.block_1(x_total)
        elif self.version == 2:
            #print('arch = 3 blocks')
            x_total = self.block_1(x_total)
            x_total = self.block_2(x_total)
            x_total = self.block_3(x_total)

        # x_total = self.block_1(x_total) + x_total
        # x_total = self.block_2(x_total) + x_total
        # x_total = self.block_3(x_total) + x_total
        # x_total = self.block_4(x_total) + x_total

        result = self.linear_out(x_total)

        result = result.squeeze(-1)

        return result
