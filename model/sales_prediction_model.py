import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from prophet import Prophet
import joblib
from datetime import datetime, timedelta

class EnsembleSalesPredictionModel:
    def __init__(self):
        # RandomForest 모델
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # XGBoost 모델
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Prophet 모델
        self.prophet_model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )
        
        # 스케일러와 데이터 저장소
        self.product_scalers = {}
        self.product_data = {}

    def prepare_features(self, df):
        # order_id를 기반으로 날짜 생성
        unique_orders = df['order_id'].unique()
        date_dict = {
            order_id: date for order_id, date in 
            zip(unique_orders, pd.date_range(start='2024-01-01', periods=len(unique_orders)))
        }
        
        # 기본 날짜 특성 추가
        df['order_date'] = df['order_id'].map(date_dict)
        df['day_of_week'] = df['order_date'].dt.weekday
        df['month'] = df['order_date'].dt.month
        
        # ML 모델용 특성
        base_features = df[['product_id', 'day_of_week', 'month', 'unit_price', 'discount']].copy()
        
        # 제품별 시계열 데이터 준비
        for product_id in df['product_id'].unique():
            product_df = df[df['product_id'] == product_id].copy()
            daily_sales = product_df.groupby('order_date')['quantity'].sum()
            
            # 날짜 범위 생성
            date_range = pd.date_range(
                start=daily_sales.index.min(), 
                end=daily_sales.index.max()
            )
            
            # 빈 날짜를 0으로 채우고 이동평균 적용
            sales_ts = daily_sales.reindex(date_range, fill_value=0)
            sales_ts = sales_ts.rolling(window=3, min_periods=1).mean()
            
            if len(sales_ts) >= 10:
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(sales_ts.values.reshape(-1, 1))
                
                self.product_scalers[product_id] = scaler
                self.product_data[product_id] = sales_ts
        
        # Prophet용 데이터
        sales_ts = df.groupby('order_date')['quantity'].sum().sort_index()
        prophet_data = pd.DataFrame({
            'ds': sales_ts.index,
            'y': sales_ts.values
        })
        
        return base_features, prophet_data

    def train(self, df):
        print("데이터 전처리 및 특성 추출 중")
        features, prophet_data = self.prepare_features(df)
        target = df['quantity']
        
        print("RandomForest 모델 학습 중...")
        self.rf_model.fit(features, target)
        
        print("XGBoost 모델 학습 중")
        self.xgb_model.fit(features, target)
        
        print("Prophet 모델 학습 중")
        try:
            self.prophet_model.fit(prophet_data)
            print("Prophet 모델 학습 완료")
        except Exception as e:
            print(f"Prophet 학습 중 오류 발생: {e}")

    def predict(self, product_id, future_days=30):
        predictions = {
            'dates': [],
            'predictions': [],
            'model_predictions': {
                'random_forest': [],
                'xgboost': [],
                'prophet': []
            }
        }
        
        try:
            future_dates = [datetime.now() + timedelta(days=x) for x in range(future_days)]
            
            future_features = pd.DataFrame({
                'product_id': [product_id] * future_days,
                'day_of_week': [d.weekday() for d in future_dates],
                'month': [d.month for d in future_dates],
                'unit_price': [0] * future_days,
                'discount': [0] * future_days
            })
            
            # 개별 모델 예측
            rf_pred = self.rf_model.predict(future_features)
            xgb_pred = self.xgb_model.predict(future_features)
            
            # Prophet 예측
            future_prophet = pd.DataFrame({'ds': future_dates})
            prophet_forecast = self.prophet_model.predict(future_prophet)
            prophet_pred = prophet_forecast['yhat'].values
            
            # 앙상블 예측
            ensemble_pred = (0.4 * rf_pred + 0.3 * xgb_pred + 0.3 * prophet_pred)
            ensemble_pred = np.maximum(ensemble_pred, 0)  # 음수 방지
            
            predictions = {
                'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                'predictions': ensemble_pred.tolist(),
                'model_predictions': {
                    'random_forest': rf_pred.tolist(),
                    'xgboost': xgb_pred.tolist(),
                    'prophet': prophet_pred.tolist()
                }
            }
            
        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            raise
        
        return predictions
    
    def save_model(self, path):
        try:
            print(f"모델 저장 중 경로: {path}")
            joblib.dump(self.rf_model, f"{path}_rf.joblib")
            joblib.dump(self.xgb_model, f"{path}_xgb.joblib")
            joblib.dump(self.prophet_model, f"{path}_prophet.joblib")
            joblib.dump(self.product_scalers, f"{path}_scalers.joblib")
            joblib.dump(self.product_data, f"{path}_product_data.joblib")
            print("모델 저장 완료")
        except Exception as e:
            print(f"모델 저장 중 오류 발생: {e}")
            raise
    
    def load_model(self, path):
        try:
            print(f"모델 로드 중 경로: {path}")
            self.rf_model = joblib.load(f"{path}_rf.joblib")
            self.xgb_model = joblib.load(f"{path}_xgb.joblib")
            self.prophet_model = joblib.load(f"{path}_prophet.joblib")
            self.product_scalers = joblib.load(f"{path}_scalers.joblib")
            self.product_data = joblib.load(f"{path}_product_data.joblib")
            print("모델 로드 완료")
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            raise

if __name__ == "__main__":
    print("데이터 로딩 중")
    df = pd.read_csv('../data/northwind_order_details.csv')
    
    # 데이터 체크
    print("\n데이터 컬럼:", df.columns)
    print("\n데이터 샘플:")
    print(df.head())
    
    # 필수 컬럼 검증
    required_columns = ['product_id', 'unit_price', 'quantity', 'discount', 'order_id']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼 누락: {col}")
    
    # 결측치 제거
    df = df.dropna()
    
    # 모델 학습
    print("\n모델 학습 시작")
    model = EnsembleSalesPredictionModel()
    model.train(df)
    
    # 모델 저장
    print("\n모델 저장 중")
    model.save_model('ensemble_model')
    
    # 테스트 예측
    print("\n테스트 예측 실행")
    test_product_id = df['product_id'].iloc[0]
    predictions = model.predict(test_product_id)
    print(f"\n제품 {test_product_id}에 대한 예측 결과:")
    print(predictions)