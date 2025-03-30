from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.utils.data import Dataset
from model import LoanModel


_OWNERSHIP_MAP = {
    'OTHER': 0,
    'OWN': 1,
    'MORTGAGE': 2,
    'RENT': 3
}

_INTENT_MAP = {
    'HOMEIMPROVEMENT': 0,
    'DEBTCONSOLIDATION': 1,
    'PERSONAL': 2,
    'VENTURE': 3,
    'MEDICAL': 4,
    'EDUCATION': 5
}

_GRADE_MAP = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6
}


_CB_DEFAULT_MAP = {
    'N': 0,
    'Y': 1,
}


def minmax_scale_column_fast(column, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    return pd.Series(
        scaler.fit_transform(column.values.reshape(-1, 1)).flatten(),
        index=column.index,
        name=column.name
    )


class LoanDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        # числовые
        data['person_age'] = minmax_scale_column_fast(data['person_age'])
        data['person_income'] = minmax_scale_column_fast(data['person_income'])
        data['person_emp_length'] = minmax_scale_column_fast(data['person_emp_length'])
        data['loan_amnt'] = minmax_scale_column_fast(data['loan_amnt'])
        data['loan_int_rate'] = minmax_scale_column_fast(data['loan_int_rate'])
        data['loan_percent_income'] = minmax_scale_column_fast(data['loan_percent_income'])
        data['cb_person_cred_hist_length'] = minmax_scale_column_fast(data['cb_person_cred_hist_length'])
        # категориальные
        data['person_home_ownership'] = data['person_home_ownership'].map(_OWNERSHIP_MAP)
        data['loan_intent'] = data['loan_intent'].map(_INTENT_MAP)
        data['loan_grade'] = data['loan_grade'].map(_GRADE_MAP)
        data['cb_person_default_on_file'] = data['cb_person_default_on_file'].map(_CB_DEFAULT_MAP)
        #print(data.info())
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item: int) -> dict[str, dict[str | Tensor] | Tensor]:
        item = self._data.iloc[item]
        return {
            'target': torch.scalar_tensor(item['loan_status'], dtype=torch.float32),
            'cat_features': {
                'person_home_ownership': torch.scalar_tensor(item['person_home_ownership'], dtype=torch.long),
                'loan_intent': torch.scalar_tensor(item['loan_intent'], dtype=torch.long),
                'loan_grade': torch.scalar_tensor(item['loan_grade'], dtype=torch.long),
                'cb_person_default_on_file': torch.scalar_tensor(item['cb_person_default_on_file'], dtype=torch.long),
            },
            'numeric_features': {
                'person_age': torch.scalar_tensor(item['person_age'], dtype=torch.float32),
                'person_income': torch.scalar_tensor(item['person_income'], dtype=torch.float32),
                'person_emp_length': torch.scalar_tensor(item['person_emp_length'], dtype=torch.float32),
                'loan_amnt': torch.scalar_tensor(item['loan_amnt'], dtype=torch.float32),
                'loan_int_rate': torch.scalar_tensor(item['loan_int_rate'], dtype=torch.float32),
                'loan_percent_income': torch.scalar_tensor(item['loan_percent_income'], dtype=torch.float32),
                'cb_person_cred_hist_length': torch.scalar_tensor(item['cb_person_cred_hist_length'], dtype=torch.float32)
            }
        }


class LoanCollator:
    def __call__(self, items: list[dict[str, dict[str | Tensor] | Tensor]]) -> dict[str, dict[str | Tensor] | Tensor]:
        return {
            'target': torch.stack([x['target'] for x in items]),
            'cat_features': {
                'person_home_ownership': torch.stack([x['cat_features']['person_home_ownership'] for x in items]),
                'loan_intent': torch.stack([x['cat_features']['loan_intent'] for x in items]),
                'loan_grade': torch.stack([x['cat_features']['loan_grade'] for x in items]),
                'cb_person_default_on_file': torch.stack([x['cat_features']['cb_person_default_on_file'] for x in items])
            },
            'numeric_features': {
                'person_age': torch.stack([x['numeric_features']['person_age'] for x in items]),
                'person_income': torch.stack([x['numeric_features']['person_income'] for x in items]),
                'person_emp_length': torch.stack([x['numeric_features']['person_emp_length'] for x in items]),
                'loan_amnt': torch.stack([x['numeric_features']['loan_amnt'] for x in items]),
                'loan_int_rate': torch.stack([x['numeric_features']['loan_int_rate'] for x in items]),
                'loan_percent_income': torch.stack([x['numeric_features']['loan_percent_income'] for x in items]),
                'cb_person_cred_hist_length': torch.stack([x['numeric_features']['cb_person_cred_hist_length'] for x in items])
            }
        }


def load_loan(file: Path) -> LoanDataset:
    df = pd.read_csv(file)

    return LoanDataset(df)


