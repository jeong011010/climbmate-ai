const ComparePage = ({ result, selectedForCompare, toggleCompareMode, colorEmoji }) => {
  const selectedProblems = result?.problems?.filter(p => selectedForCompare.includes(p.id)) || []
  
  return (
    <div className="w-full px-2 sm:px-4">
      <div className="glass-card p-4 sm:p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl sm:text-2xl font-bold text-slate-800">🔍 문제 비교</h2>
          <button
            onClick={toggleCompareMode}
            className="glass-button px-4 py-2 text-sm"
          >
            비교 모드 종료
          </button>
        </div>
        
        {selectedProblems.length === 0 ? (
          <p className="text-slate-600 text-center py-8">
            비교할 문제를 선택해주세요. (최대 3개)
          </p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {selectedProblems.map((problem) => (
              <div key={problem.id} className="glass-card p-4">
                <div className="text-center mb-3">
                  <span className="text-3xl">{colorEmoji[problem.color_name] || '⭕'}</span>
                  <h3 className="text-lg font-bold mt-2">{(problem.color_name || 'UNKNOWN').toUpperCase()}</h3>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-slate-600">난이도:</span>
                    <span className="font-bold text-blue-600">{problem.difficulty}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-slate-600">유형:</span>
                    <span className="font-bold text-green-600">{problem.type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-slate-600">홀드 수:</span>
                    <span className="font-bold">{problem.hold_count}개</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-slate-600">분석 방법:</span>
                    <span className="font-bold">{problem.gpt4_reasoning ? 'GPT-4 AI' : '규칙 기반'}</span>
                  </div>
                </div>
                
                {problem.gpt4_reasoning && (
                  <div className="mt-3 p-2 bg-blue-50 rounded text-xs text-slate-700">
                    <strong>AI 분석:</strong> {problem.gpt4_reasoning}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default ComparePage

