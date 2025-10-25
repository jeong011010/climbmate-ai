import * as api from '../api'

const StatsPage = ({ modelStats, convertGpt4ToTraining }) => {
  
  const handleTrainModel = async () => {
    try {
      await api.trainModel()
      alert('모델 훈련 완료!')
    } catch (err) {
      alert(`훈련 실패: ${err.message}`)
    }
  }
  
  return (
    <div className="w-full px-2 sm:px-4">
      <div className="glass-card p-4 sm:p-6">
        <h2 className="text-xl sm:text-2xl font-bold mb-4 text-slate-800">📊 통계</h2>
        
        {modelStats ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="glass-card p-4 text-center">
                <div className="text-2xl font-bold text-blue-600">{modelStats.total_problems}</div>
                <div className="text-sm text-slate-600">전체 문제</div>
              </div>
              <div className="glass-card p-4 text-center">
                <div className="text-2xl font-bold text-green-600">{modelStats.verified_problems}</div>
                <div className="text-sm text-slate-600">검증된 문제</div>
              </div>
            </div>
            
            {modelStats.verified_problems > 0 && (
              <div className="glass-card p-4">
                <h3 className="font-bold mb-2">AI 모델 성능</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>난이도 정확도:</span>
                    <span className="font-bold text-blue-600">
                      {Math.round(modelStats.difficulty_accuracy * 100)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>유형 정확도:</span>
                    <span className="font-bold text-green-600">
                      {Math.round(modelStats.type_accuracy * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
            
            <div className="flex flex-col sm:flex-row gap-2">
              <button
                onClick={convertGpt4ToTraining}
                className="glass-button flex-1 py-2 text-xs sm:text-sm"
              >
                <span className="hidden sm:inline">🤖 GPT-4 결과를 훈련 데이터로 변환</span>
                <span className="sm:hidden">🤖 GPT-4 변환</span>
              </button>
              <button
                onClick={handleTrainModel}
                className="glass-button flex-1 py-2 text-xs sm:text-sm"
              >
                <span className="hidden sm:inline">🎯 모델 재훈련</span>
                <span className="sm:hidden">🎯 재훈련</span>
              </button>
            </div>
          </div>
        ) : (
          <p className="text-slate-600 text-center py-8">통계 데이터를 불러올 수 없습니다.</p>
        )}
      </div>
    </div>
  )
}

export default StatsPage

