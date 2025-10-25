const HistoryPage = ({ analysisHistory, setResult, setPreview, setCurrentView }) => (
  <div className="w-full px-2 sm:px-4">
    <div className="glass-card p-4 sm:p-6">
      <h2 className="text-xl sm:text-2xl font-bold mb-4 text-slate-800">📚 분석 히스토리</h2>
      {analysisHistory.length === 0 ? (
        <p className="text-slate-600 text-center py-8">아직 분석한 문제가 없습니다.</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
          {analysisHistory.map((item) => (
            <div key={item.id} className="glass-card p-4 hover:shadow-lg transition-shadow">
              <img 
                src={item.image} 
                alt="분석된 문제" 
                className="w-full h-32 object-cover rounded-lg mb-3"
              />
              <div className="text-sm text-slate-600 mb-2">
                {new Date(item.timestamp).toLocaleString()}
              </div>
              <div className="space-y-1">
                {item.result.problems?.map((problem, idx) => (
                  <div key={idx} className="flex justify-between items-center">
                    <span className="text-sm font-medium">{problem.color}</span>
                    <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                      {problem.difficulty} {problem.type}
                    </span>
                  </div>
                ))}
              </div>
              <button
                onClick={() => {
                  setResult(item.result)
                  setPreview(item.image)
                  setCurrentView('analyze')
                }}
                className="w-full mt-3 glass-button text-sm py-2"
              >
                다시 보기
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  </div>
)

export default HistoryPage

