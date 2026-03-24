#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data():
    train_df = pd.read_csv('../datasets/house_prices/train.csv')
    test_df = pd.read_csv('../datasets/house_prices/test.csv')
    return train_df, test_df


def remove_outliers(df):
    """Remove outliers from training data"""
    df = df.copy()
    # Remover casas con GrLivArea > 4000 y precio bajo (outliers conocidos)
    df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000))]
    return df


def feature_engineering(df, is_train=True):
    df = df.copy()
    
    # Llenar NaN en TotalBsmtSF antes de usarlo
    df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
    df['GarageArea'] = df['GarageArea'].fillna(0)
    df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)
    df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(0)
    df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0)
    df['2ndFlrSF'] = df['2ndFlrSF'].fillna(0)
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # === Features de área total ===
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    df['TotalArea'] = df['GrLivArea'] + df['TotalBsmtSF'] + df['GarageArea']
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'].fillna(0) + 0.5 * df['BsmtHalfBath'].fillna(0)
    
    # === Features de edad ===
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['GarageAge'] = df['YrSold'] - df['GarageYrBlt'].fillna(df['YearBuilt'])
    df['IsNew'] = (df['YearBuilt'] == df['YrSold']).astype(int)
    df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    
    # === Features de calidad ===
    qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, np.nan: 0}
    
    for col in qual_cols:
        if col in df.columns:
            df[col + '_Num'] = df[col].map(qual_map).fillna(0)
    
    df['OverallScore'] = df['OverallQual'] * df['OverallCond']
    df['ExterScore'] = df.get('ExterQual_Num', 0) * df.get('ExterCond_Num', 0)
    df['GarageScore'] = df.get('GarageQual_Num', 0) * df.get('GarageCond_Num', 0)
    df['TotalQual'] = df['OverallQual'] + df.get('ExterQual_Num', 0) + df.get('KitchenQual_Num', 0)
    
    # === Ratios y proporciones ===
    df['LotAreaPerRoom'] = df['LotArea'] / (df['TotRmsAbvGrd'] + 1)
    df['GrLivAreaPerRoom'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)
    df['BsmtFinRatio'] = (df['BsmtFinSF1'] + df['BsmtFinSF2']) / (df['TotalBsmtSF'] + 1)
    df['GarageAreaPerCar'] = df['GarageArea'] / (df['GarageCars'].fillna(0) + 1)
    
    # === Features binarias ===
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['HasMasVnr'] = (df['MasVnrArea'] > 0).astype(int)
    
    # === Polinomiales de features importantes ===
    df['OverallQual_2'] = df['OverallQual'] ** 2
    df['GrLivArea_2'] = df['GrLivArea'] ** 2
    df['TotalSF_2'] = df['TotalSF'] ** 2
    
    # === Log transforms para features con distribución sesgada ===
    skewed_features = ['LotArea', 'GrLivArea', 'TotalSF', 'TotalArea', '1stFlrSF']
    for feat in skewed_features:
        if feat in df.columns:
            df[feat + '_Log'] = np.log1p(df[feat])
    
    # === Interacciones ===
    df['Qual_x_Area'] = df['OverallQual'] * df['GrLivArea']
    df['Qual_x_TotalSF'] = df['OverallQual'] * df['TotalSF']
    df['Age_x_Qual'] = df['HouseAge'] * df['OverallQual']
    
    return df


def handle_missing_values(train_df, test_df):
    """Handle NaN values: median for numeric, mode for categorical"""
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    
    numeric_cols = [c for c in numeric_cols if c not in ['SalePrice', 'Id']]
    
    impute_values = {}
    
    for col in numeric_cols:
        impute_values[col] = train_df[col].median()
    
    for col in categorical_cols:
        mode_val = train_df[col].mode()
        impute_values[col] = mode_val[0] if len(mode_val) > 0 else 'Missing'
    
    for col in numeric_cols:
        train_df[col] = train_df[col].fillna(impute_values[col])
        test_df[col] = test_df[col].fillna(impute_values.get(col, 0))
    
    for col in categorical_cols:
        train_df[col] = train_df[col].fillna(impute_values[col])
        test_df[col] = test_df[col].fillna(impute_values.get(col, 'Missing'))
    
    return train_df, test_df


def preprocess_data(train_df, test_df):
    """Full preprocessing pipeline"""
    test_ids = test_df['Id'].values
    y_train = train_df['SalePrice'].values
    
    # Remover outliers del training
    train_df = remove_outliers(train_df)
    y_train = train_df['SalePrice'].values
    
    train_df = train_df.drop(['Id', 'SalePrice'], axis=1)
    test_df = test_df.drop(['Id'], axis=1)
    
    # Feature engineering
    train_df = feature_engineering(train_df, is_train=True)
    test_df = feature_engineering(test_df, is_train=False)
    
    # Asegurar mismas columnas
    common_cols = list(set(train_df.columns) & set(test_df.columns))
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]
    
    # Handle missing values
    train_df, test_df = handle_missing_values(train_df, test_df)
    
    # Encoding categórico
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        train_df[categorical_cols] = encoder.fit_transform(train_df[categorical_cols])
        test_df[categorical_cols] = encoder.transform(test_df[categorical_cols])
    
    X_train = train_df.values.astype(np.float32)
    X_test = test_df.values.astype(np.float32)
    
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    
    y_train_log = np.log1p(y_train).astype(np.float32)
    
    return X_train, y_train_log, X_test, test_ids


class HousePriceNet(nn.Module):
    """Deeper neural network with residual connections"""
    
    def __init__(self, input_dim):
        super(HousePriceNet, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.drop4 = nn.Dropout(0.1)
        
        self.fc5 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.input_bn(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.drop3(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.drop4(x)
        
        x = self.fc5(x)
        return x


def train_nn_kfold(X_train, y_train, X_test, n_folds=5):
    """Train neural network with K-Fold cross validation"""
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    input_dim = X_train.shape[1]
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold + 1}/{n_folds}", end=" ")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        X_tr_tensor = torch.FloatTensor(X_tr).to(device)
        y_tr_tensor = torch.FloatTensor(y_tr).reshape(-1, 1).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model = HousePriceNet(input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(500):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = torch.sqrt(criterion(outputs, batch_y))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = torch.sqrt(criterion(val_outputs, y_val_tensor)).item()
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= 50:
                break
        
        model.load_state_dict(best_weights)
        model.eval()
        
        with torch.no_grad():
            oof_preds[val_idx] = model(X_val_tensor).cpu().numpy().flatten()
            test_preds += model(X_test_tensor).cpu().numpy().flatten() / n_folds
        
        fold_scores.append(best_val_loss)
        print(f"- RMSLE: {best_val_loss:.5f}")
    
    
    return oof_preds, test_preds


def train_xgb_kfold(X_train, y_train, X_test, n_folds=5):
    """Train XGBoost with K-Fold cross validation"""
    if not HAS_XGB:
        return np.zeros(len(X_train)), np.zeros(len(X_test))
    
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    params = {
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'max_depth': 4,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'random_state': SEED,
        'n_jobs': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold + 1}/{n_folds}", end=" ")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
        )
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / n_folds
        
        score = np.sqrt(np.mean((val_pred - y_val) ** 2))
        fold_scores.append(score)
        print(f"- RMSLE: {score:.5f}")
    
    
    return oof_preds, test_preds


def train_lgb_kfold(X_train, y_train, X_test, n_folds=5):
    """Train LightGBM with K-Fold cross validation"""
    if not HAS_LGB:
        return np.zeros(len(X_train)), np.zeros(len(X_test))
    
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    params = {
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'max_depth': 4,
        'num_leaves': 20,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'random_state': SEED,
        'n_jobs': -1,
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold + 1}/{n_folds}", end=" ")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
        )
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / n_folds
        
        score = np.sqrt(np.mean((val_pred - y_val) ** 2))
        fold_scores.append(score)
        print(f"- RMSLE: {score:.5f}")
    
    
    return oof_preds, test_preds


def train_ridge_kfold(X_train, y_train, X_test, n_folds=5):
    """Train Ridge Regression with K-Fold cross validation"""
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold + 1}/{n_folds}", end=" ")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = Ridge(alpha=10.0, random_state=SEED)
        model.fit(X_tr, y_tr)
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / n_folds
        
        score = np.sqrt(np.mean((val_pred - y_val) ** 2))
        fold_scores.append(score)
        print(f"- RMSLE: {score:.5f}")
    
    
    return oof_preds, test_preds


def train_gbr_kfold(X_train, y_train, X_test, n_folds=5):
    """Train Gradient Boosting Regressor with K-Fold cross validation"""
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold + 1}/{n_folds}", end=" ")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=SEED
        )
        model.fit(X_tr, y_tr)
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / n_folds
        
        score = np.sqrt(np.mean((val_pred - y_val) ** 2))
        fold_scores.append(score)
        print(f"- RMSLE: {score:.5f}")
    
    
    return oof_preds, test_preds


def optimize_weights(oof_predictions, y_train):
    """Find optimal weights for ensemble using scipy optimization"""
    from scipy.optimize import minimize
    
    def rmsle(weights):
        final_pred = np.zeros(len(y_train))
        for i, pred in enumerate(oof_predictions):
            if np.any(pred != 0):  # Solo incluir modelos que produjeron predicciones
                final_pred += weights[i] * pred
        return np.sqrt(np.mean((final_pred - y_train) ** 2))
    
    n_models = len(oof_predictions)
    initial_weights = np.ones(n_models) / n_models
    
    # Contar modelos activos
    active_models = sum(1 for pred in oof_predictions if np.any(pred != 0))
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]
    
    result = minimize(rmsle, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x


def main():

    
    # Load data
    train_df, test_df = load_data()
    
    # Preprocess
    X_train, y_train, X_test, test_ids = preprocess_data(train_df, test_df)

    
    # Train models    
    n_folds = 5
    nn_oof, nn_test = train_nn_kfold(X_train, y_train, X_test, n_folds)
    xgb_oof, xgb_test = train_xgb_kfold(X_train, y_train, X_test, n_folds)
    lgb_oof, lgb_test = train_lgb_kfold(X_train, y_train, X_test, n_folds)
    ridge_oof, ridge_test = train_ridge_kfold(X_train, y_train, X_test, n_folds)
    gbr_oof, gbr_test = train_gbr_kfold(X_train, y_train, X_test, n_folds)
    
    # Ensemble    
    oof_predictions = [nn_oof, xgb_oof, lgb_oof, ridge_oof, gbr_oof]
    test_predictions = [nn_test, xgb_test, lgb_test, ridge_test, gbr_test]
    model_names = ['NN', 'XGB', 'LGB', 'Ridge', 'GBR']
    
    # Optimizar pesos
    weights = optimize_weights(oof_predictions, y_train)
    
    for name, w in zip(model_names, weights):
        if w > 0.01:
            print(f"    {name}: {w:.3f}")
    
    # Predicción final con pesos optimizados
    final_test_pred = np.zeros(len(X_test))
    for w, pred in zip(weights, test_predictions):
        final_test_pred += w * pred
    
    # También calculamos ensemble simple promediado (a veces funciona mejor)
    active_preds = [p for p in test_predictions if np.any(p != 0)]
    simple_avg = np.mean(active_preds, axis=0)
    
    # OOF score para ambos métodos
    final_oof = np.zeros(len(y_train))
    for w, pred in zip(weights, oof_predictions):
        final_oof += w * pred
    
    simple_oof = np.mean([p for p in oof_predictions if np.any(p != 0)], axis=0)
    
    weighted_score = np.sqrt(np.mean((final_oof - y_train) ** 2))
    simple_score = np.sqrt(np.mean((simple_oof - y_train) ** 2))
    

    
    # Usar el mejor
    if simple_score < weighted_score:
        final_pred = simple_avg
        cv_score = simple_score
    else:
        final_pred = final_test_pred
        cv_score = weighted_score
    
    # Transform back
    predictions = np.expm1(final_pred)
    predictions = np.maximum(predictions, 0)
    
    # Save
    submission_df = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    submission_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()