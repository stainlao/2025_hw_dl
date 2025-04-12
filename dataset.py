from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numerical_columns = [
        'person_age', 'person_income', 'person_emp_length',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length'
    ]

    # Инициализация и обучение скейлеров на тренировочных данных
    scalers = {}
    for col in numerical_columns:
        scaler = StandardScaler()
        scaler.fit(train_df[col].values.reshape(-1, 1))
        scalers[col] = scaler

    # Применение скейлеров
    for col in numerical_columns:
        train_df[col] = pd.Series(
            scalers[col].transform(train_df[col].values.reshape(-1, 1)).flatten(),
            index=train_df.index,
            name=col
        )
        test_df[col] = pd.Series(
            scalers[col].transform(test_df[col].values.reshape(-1, 1)).flatten(),
            index=test_df.index,
            name=col
        )

    # Обработка категориальных признаков
    categorical_mappings = {
        'person_home_ownership': _OWNERSHIP_MAP,
        'loan_intent': _INTENT_MAP,
        'loan_grade': _GRADE_MAP,
        'cb_person_default_on_file': _CB_DEFAULT_MAP
    }

    for col, mapping in categorical_mappings.items():
        train_df[col] = train_df[col].map(mapping)
        test_df[col] = test_df[col].map(mapping)

    return train_df, test_df


class LoanDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor] | Tensor]:
        row = self._data.iloc[idx]
        return {
            'target': torch.tensor(row['loan_status'], dtype=torch.float32),
            'cat_features': {
                'person_home_ownership': torch.tensor(row['person_home_ownership'], dtype=torch.long),
                'loan_intent': torch.tensor(row['loan_intent'], dtype=torch.long),
                'loan_grade': torch.tensor(row['loan_grade'], dtype=torch.long),
                'cb_person_default_on_file': torch.tensor(row['cb_person_default_on_file'], dtype=torch.long),
            },
            'numeric_features': {
                'person_age': torch.tensor(row['person_age'], dtype=torch.float32),
                'person_income': torch.tensor(row['person_income'], dtype=torch.float32),
                'person_emp_length': torch.tensor(row['person_emp_length'], dtype=torch.float32),
                'loan_amnt': torch.tensor(row['loan_amnt'], dtype=torch.float32),
                'loan_int_rate': torch.tensor(row['loan_int_rate'], dtype=torch.float32),
                'loan_percent_income': torch.tensor(row['loan_percent_income'], dtype=torch.float32),
                'cb_person_cred_hist_length': torch.tensor(row['cb_person_cred_hist_length'], dtype=torch.float32)
            }
        }


class LoanCollator:
    def __call__(self, batch: list) -> dict:
        return {
            'target': torch.stack([x['target'] for x in batch]),
            'cat_features': {
                k: torch.stack([x['cat_features'][k] for x in batch])
                for k in batch[0]['cat_features'].keys()
            },
            'numeric_features': {
                k: torch.stack([x['numeric_features'][k] for x in batch])
                for k in batch[0]['numeric_features'].keys()
            }
        }


def load_loan(train_path: Path, test_path: Path) -> tuple[LoanDataset, LoanDataset]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df, test_df = preprocess_data(train_df, test_df)
    return LoanDataset(train_df), LoanDataset(test_df)


