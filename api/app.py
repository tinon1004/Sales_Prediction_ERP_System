from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sys
import os

# 상위 디렉토리의 model 폴더를 참조하기 위한 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 모델 임포트
from model.sales_prediction_model import EnsembleSalesPredictionModel

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

def init_model():
    model = EnsembleSalesPredictionModel()
    model_path = os.path.join(parent_dir, 'model', 'ensemble_model')
    
    try:
        model.load_model(model_path)
    except Exception as e:
        print(f"기존 모델 로드 실패: {e}")
        
        try:
            # 데이터 로드
            data_path = os.path.join(parent_dir, 'data', 'northwind_order_details.csv')
            df = pd.read_csv(data_path)
            
            # 필수 컬럼 확인
            required_columns = ['product_id', 'unit_price', 'quantity', 'discount', 'order_id']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"필수 컬럼 누락: {col}")
            
            # 데이터 전처리
            df = df.dropna()
            
            # 이상치 처리
            Q1 = df['quantity'].quantile(0.25)
            Q3 = df['quantity'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df['quantity'] >= lower_bound) & (df['quantity'] <= upper_bound)]
            
            # 모델 학습
            model.train(df)
            
            # 모델 저장
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save_model(model_path)
            print("새로운 모델 학습 및 저장 완료")
            
        except Exception as train_error:
            print(f"모델 학습 중 오류 발생: {train_error}")
            raise
    
    return model

# 모델 초기화
try:
    model = init_model()
    print("모델 초기화 완료")
except Exception as e:
    print(f"모델 초기화 실패: {e}")
    model = None

@app.route('/api/predict', methods=['GET'])
def predict_sales():
    if model is None:
        return jsonify({'error': '모델이 초기화되지 않음'}), 500
        
    try:
        product_id = int(request.args.get('product_id'))
        days = int(request.args.get('days', 30))
        
        predictions = model.predict(product_id, days)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict_custom', methods=['POST'])
def predict_custom():
    if model is None:
        return jsonify({'error': '모델이 초기화되지 않음'}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        product_id = data.get('product_id')
        unit_price = data.get('unit_price')
        discount = data.get('discount')
        
        if product_id is None or unit_price is None:
            return jsonify({'error': '제품 ID와 단가는 필수 입력값입니다.'}), 400
            
        predictions = model.predict(
            product_id=int(product_id),
            future_days=30
        )
        
        return jsonify(predictions)
    except ValueError as ve:
        return jsonify({'error': f'잘못된 입력값: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'예측 중 오류 발생: {str(e)}'}), 400

@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        data_path = os.path.join(parent_dir, 'data', 'northwind_order_details.csv')
        df = pd.read_csv(data_path)
        
        if df.empty:
            return jsonify({'error': '데이터가 비어있습니다.'}), 400
            
        product_info = df.groupby('product_id').agg({
            'unit_price': 'mean',
            'quantity': ['mean', 'max'],
            'discount': 'mean'
        }).round(2)
        
        product_info.columns = ['avg_price', 'avg_quantity', 'max_quantity', 'avg_discount']
        product_info = product_info.reset_index()
        
        product_list = [{
            'id': str(p['product_id']),
            'avg_price': float(p['avg_price']),
            'avg_quantity': float(p['avg_quantity']),
            'max_quantity': float(p['max_quantity']),
            'avg_discount': float(p['avg_discount'])
        } for p in product_info.to_dict('records')]
        
        return jsonify({'products': product_list})
    except Exception as e:
        print(f"Error in get_products: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')