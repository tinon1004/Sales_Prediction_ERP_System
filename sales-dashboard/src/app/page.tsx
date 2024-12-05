"use client";

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

interface Product {
  id: string;
  avg_price: number;
  avg_quantity: number;
  max_quantity: number;
  avg_discount: number;
}

interface ChartDataPoint {
  date: string;
  [key: string]: string | number;
}

interface CustomApiResponse {
  dates: string[];
  predictions: number[];
  model_predictions?: {
    random_forest: number[];
    xgboost: number[];
    prophet: number[];
  };
}

const SalesEnsembleDashboard = () => {
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
  const [historicPredictions, setHistoricPredictions] = useState<ChartDataPoint[] | null>(null);
  const [customPredictions, setCustomPredictions] = useState<ChartDataPoint[] | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'historic' | 'custom'>('historic');

  const [customInput, setCustomInput] = useState({
    product_id: '',
    unit_price: '',
    discount: ''
  });

  useEffect(() => {
    fetch('http://localhost:5000/api/products')
      .then(res => res.json())
      .then((data: { products: Product[] }) => {
        setProducts(data.products);
        if (data.products.length > 0) {
          setSelectedProduct(data.products[0].id);
        }
      })
      .catch(() => setError('제품 목록을 불러오는데 실패했습니다.'));
  }, []);

  const fetchHistoricPredictions = async (productId: string) => {
    try {
      const response = await fetch(`http://localhost:5000/api/predict?product_id=${productId}&days=30`);
      if (!response.ok) {
        throw new Error('예측 데이터를 가져오는데 실패했습니다.');
      }
      const data = await response.json() as CustomApiResponse;
      
      const chartData = data.dates.map((date: string, index: number) => ({
        date,
        '앙상블 예측': Math.round(data.predictions[index]),
        'RandomForest': Math.round(data.model_predictions?.random_forest[index] || 0),
        'XGBoost': Math.round(data.model_predictions?.xgboost[index] || 0),
        'Prophet': Math.round(data.model_predictions?.prophet[index] || 0)
      }));
      
      setHistoricPredictions(chartData);
    } catch (error) {
      if (error instanceof Error) {
        setError(error.message);
      }
    }
  };

  useEffect(() => {
    if (selectedProduct) {
      fetchHistoricPredictions(selectedProduct);
    }
  }, [selectedProduct]);

  const handleCustomSubmit = async () => {
    if (!customInput.product_id || !customInput.unit_price) {
      setError('제품 ID와 단가는 필수 입력값입니다.');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:5000/api/predict_custom', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          product_id: parseInt(customInput.product_id),
          unit_price: parseFloat(customInput.unit_price),
          discount: customInput.discount ? parseFloat(customInput.discount) : 0
        })
      });

      if (!response.ok) {
        const errorData = await response.json() as { error: string };
        throw new Error(errorData.error || '예측에 실패했습니다.');
      }

      const data = await response.json() as CustomApiResponse;
      
      const chartData = data.dates.map((date: string, index: number) => ({
        date,
        '예측 판매량': Math.round(data.predictions[index])
      }));

      setCustomPredictions(chartData);
    } catch (error: unknown) {
      if (error instanceof Error) {
        setError(error.message);
      } else {
        setError('알 수 없는 오류가 발생했습니다.');
      }
    } finally {
      setLoading(false);
    }
  };

  const getProductInfo = (productId: string): Product | undefined => {
    return products.find(p => p.id === productId);
  };

  const getCurrentPredictions = (): ChartDataPoint[] => {
    if (activeTab === 'historic') {
      return historicPredictions || [];
    }
    return customPredictions || [];
  };

  const renderChart = () => {
    const currentData = getCurrentPredictions();
    if (currentData.length === 0) return null;

    return (
      <div className="mt-6">
        <div className="h-96">
          <LineChart
            width={1000}
            height={400}
            data={currentData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            {Object.keys(currentData[0])
              .filter(key => key !== 'date')
              .map((key) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={
                    key === '앙상블 예측' ? '#8884d8' :
                    key === 'RandomForest' ? '#82ca9d' :
                    key === 'LSTM' ? '#ffc658' : '#ff7300'
                  }
                  name={key}
                />
              ))}
          </LineChart>
        </div>
      </div>
    );
  };

  const renderModelExplanation = () => {
    return (
      <div className="mt-8 p-6 bg-gray-50 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">예측 모델 설명</h3>
        <div className="space-y-4">
          <div className="flex items-start space-x-3">
            <div className="w-4 h-4 mt-1 rounded-full" style={{ backgroundColor: '#8884d8' }}></div>
            <div>
              <p className="font-medium">앙상블 예측</p>
              <p className="text-sm text-gray-600">RandomForest, XGBoost, Prophet 모델의 예측을 결합한 최종 예측값입니다. 
              각 모델의 장점을 활용하여 더 안정적이고 정확한 예측을 제공합니다.</p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <div className="w-4 h-4 mt-1 rounded-full" style={{ backgroundColor: '#82ca9d' }}></div>
            <div>
              <p className="font-medium">RandomForest 예측</p>
              <p className="text-sm text-gray-600">제품 특성, 시간적 특성을 고려한 예측값입니다. 
              비선형적인 패턴과 특성 간의 복잡한 상호작용을 잘 포착합니다.</p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <div className="w-4 h-4 mt-1 rounded-full" style={{ backgroundColor: '#ffc658' }}></div>
            <div>
              <p className="font-medium">XGBoost 예측</p>
              <p className="text-sm text-gray-600">그래디언트 부스팅 기반의 고성능 예측 모델입니다. 
              시계열 데이터의 복잡한 패턴을 효과적으로 학습합니다.</p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <div className="w-4 h-4 mt-1 rounded-full" style={{ backgroundColor: '#ff7300' }}></div>
            <div>
              <p className="font-medium">Prophet 예측</p>
              <p className="text-sm text-gray-600">Facebook이 개발한 시계열 예측 모델입니다. 
              계절성, 휴일 효과, 전반적인 추세를 고려한 예측을 제공합니다.</p>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderCustomGuide = () => {
    return (
      <div className="mt-8 p-6 bg-gray-50 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">커스텀 예측 가이드</h3>
        <div className="space-y-4">
          <div>
            <h4 className="font-medium mb-2">사용 방법</h4>
            <ul className="list-disc list-inside text-sm text-gray-600 space-y-2">
              <li>제품 ID: 예측하고자 하는 제품의 고유 번호를 입력해주세요.</li>
              <li>단가: 제품의 판매 가격을 달러 단위로 입력해주세요.</li>
              <li>할인율: 0에서 1 사이의 값으로 할인율을 입력해주세요. (예: 20% 할인 = 0.2)</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium mb-2">예측 결과 해석</h4>
            <ul className="list-disc list-inside text-sm text-gray-600 space-y-2">
              <li>그래프는 향후 30일간의 일별 예상 판매량을 보여줍니다.</li>
              <li>예측값은 과거 판매 데이터, 가격, 할인율 등을 종합적으로 고려하여 산출됩니다.</li>
              <li>실제 판매량은 시장 상황, 계절성, 특별 이벤트 등 다양한 요인에 따라 달라질 수 있습니다.</li>
            </ul>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="w-full max-w-6xl mx-auto my-8 bg-white rounded-lg shadow-lg">
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-2xl font-bold text-gray-900">판매량 예측 대시보드</h2>
      </div>
      
      <div className="p-6">
        <div className="flex space-x-2 border-b border-gray-200 mb-6">
          <button
            className={`px-4 py-2 text-sm font-medium ${
              activeTab === 'historic'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('historic')}
          >
            히스토리 기반 예측
          </button>
          <button
            className={`px-4 py-2 text-sm font-medium ${
              activeTab === 'custom'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('custom')}
          >
            커스텀 예측
          </button>
        </div>

        {activeTab === 'historic' && (
          <div className="space-y-6">
            {products && products.length > 0 ? (
              <select
                value={selectedProduct || ''}
                onChange={(e) => setSelectedProduct(e.target.value)}
                className="w-64 px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              >
                {products.map(product => (
                  <option key={product.id} value={product.id}>
                    제품 {product.id} (평균가: ${product.avg_price})
                  </option>
                ))}
              </select>
            ) : (
              <div className="text-gray-500">제품 목록을 불러오는 중...</div>
            )}

            {selectedProduct && (
              <div className="grid grid-cols-4 gap-4 my-4">
                <div className="p-4 border rounded-lg bg-gray-50">
                  <div className="text-sm text-gray-500">평균 가격</div>
                  <div className="text-lg font-bold">
                    ${getProductInfo(selectedProduct)?.avg_price || 0}
                  </div>
                </div>
                <div className="p-4 border rounded-lg bg-gray-50">
                  <div className="text-sm text-gray-500">평균 판매량</div>
                  <div className="text-lg font-bold">
                    {getProductInfo(selectedProduct)?.avg_quantity || 0}
                  </div>
                </div>
                <div className="p-4 border rounded-lg bg-gray-50">
                  <div className="text-sm text-gray-500">최대 판매량</div>
                  <div className="text-lg font-bold">
                    {getProductInfo(selectedProduct)?.max_quantity || 0}
                  </div>
                </div>
                <div className="p-4 border rounded-lg bg-gray-50">
                  <div className="text-sm text-gray-500">평균 할인율</div>
                  <div className="text-lg font-bold">
                    {((getProductInfo(selectedProduct)?.avg_discount || 0) * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'custom' && (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <input
                type="number"
                placeholder="제품 ID"
                value={customInput.product_id}
                onChange={(e) => setCustomInput({...customInput, product_id: e.target.value})}
                className="px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              />
              <input
                type="number"
                placeholder="단가 ($)"
                value={customInput.unit_price}
                onChange={(e) => setCustomInput({...customInput, unit_price: e.target.value})}
                className="px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              />
              <input
                type="number"
                placeholder="할인율 (0-1)"
                value={customInput.discount}
                onChange={(e) => setCustomInput({...customInput, discount: e.target.value})}
                className="px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <button
              onClick={handleCustomSubmit}
              className="w-full px-4 py-2 text-white bg-blue-500 rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
              예측하기
            </button>
          </div>
        )}

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {loading ? (
          <div className="flex justify-center items-center h-64">
            <p className="text-gray-500">데이터를 불러오는 중...</p>
          </div>
        ) : (
          <>
            {renderChart()}
            {activeTab === 'historic' ? renderModelExplanation() : renderCustomGuide()}
          </>
        )}
      </div>
    </div>
  );
};

export default SalesEnsembleDashboard;