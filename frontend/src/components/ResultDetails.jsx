import { useState } from 'react'

const ResultDetails = ({ 
  result, 
  selectedHold, 
  selectedProblem,
  setSelectedHold,
  setShowHoldFeedbackModal,
  setShowFeedbackModal,
  colorEmoji,
  onProblemSelect
}) => {
  const [isHoldInfoExpanded, setIsHoldInfoExpanded] = useState(true)
  
  if (!result) return null

  return (
    <div className="w-full space-y-4">
      {/* 통계 */}
      <div className="flex flex-row gap-3 mx-auto mb-4 w-full justify-center items-center">
        <div className="glass-card p-4 rounded-xl text-center shadow-md transition-all duration-300 flex-1 min-w-[70px] max-w-[100px] hover:translate-y-[-3px] hover:shadow-lg">
          <div className="text-2xl font-extrabold gradient-text mb-1">{result.statistics.total_problems}</div>
          <div className="text-xs text-slate-600 font-semibold">문제 수</div>
        </div>
        <div className="glass-card p-4 rounded-xl text-center shadow-md transition-all duration-300 flex-1 min-w-[70px] max-w-[100px] hover:translate-y-[-3px] hover:shadow-lg">
          <div className="text-2xl font-extrabold gradient-text mb-1">{result.statistics.total_holds}</div>
          <div className="text-xs text-slate-600 font-semibold">홀드 수</div>
        </div>
        <div className="glass-card p-4 rounded-xl text-center shadow-md transition-all duration-300 flex-1 min-w-[70px] max-w-[100px] hover:translate-y-[-3px] hover:shadow-lg">
          <div className="text-2xl font-extrabold gradient-text mb-1">{result.statistics.analyzable_problems}</div>
          <div className="text-xs text-slate-600 font-semibold">분석 가능</div>
        </div>
      </div>

      {/* 선택된 홀드 상세 (아코디언) */}
      {selectedHold && selectedProblem && (
        <div className="glass-card mx-auto mb-3 w-full shadow-md border-2 border-yellow-400 overflow-hidden">
          {/* 헤더 - 항상 보임 */}
          <div 
            className="flex items-center justify-between p-4 cursor-pointer hover:bg-slate-50 transition-colors"
            onClick={() => setIsHoldInfoExpanded(!isHoldInfoExpanded)}
          >
            <h3 className="text-xl text-slate-800 font-bold flex items-center gap-2">
              <span className="text-3xl">🎯</span>
              홀드 정보
              <span className="text-sm font-normal text-slate-600 ml-2">
                (홀드 #{selectedHold.id})
              </span>
            </h3>
            <div className="flex items-center gap-2">
              <span className="text-slate-600 text-sm">
                {isHoldInfoExpanded ? '접기' : '펼치기'}
              </span>
              <span className={`transform transition-transform duration-200 ${isHoldInfoExpanded ? 'rotate-180' : ''}`}>
                ▼
              </span>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  setSelectedHold(null)
                }}
                className="ml-2 px-3 py-1 text-slate-600 hover:text-slate-800 text-sm hover:bg-slate-200 rounded"
              >
                ✕
              </button>
            </div>
          </div>
          
          {/* 내용 - 펼쳤을 때만 보임 */}
          <div className={`transition-all duration-300 ease-in-out ${isHoldInfoExpanded ? 'max-h-[1000px] opacity-100' : 'max-h-0 opacity-0'}`}>
            <div className="p-4 pt-0">
              <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-white/80 backdrop-blur-sm p-4 rounded-xl shadow-md">
              <h4 className="text-xs mb-2 text-slate-600 font-semibold text-center">💎 홀드 실제 색상</h4>
              <div className="flex flex-col items-center justify-center gap-2">
                <div 
                  className="w-16 h-16 rounded-full shadow-lg border-4 border-white"
                  style={{
                    backgroundColor: selectedHold.rgb ? 
                      `rgb(${selectedHold.rgb[0]}, ${selectedHold.rgb[1]}, ${selectedHold.rgb[2]})` : 
                      '#808080'
                  }}
                />
                <div className="text-xs font-mono text-slate-600 text-center">
                  {selectedHold.rgb ? 
                    `RGB(${selectedHold.rgb[0]}, ${selectedHold.rgb[1]}, ${selectedHold.rgb[2]})` : 
                    'N/A'}
                </div>
                <span className="text-xs font-bold text-slate-800">{(selectedHold.individual_color || 'UNKNOWN').toUpperCase()}</span>
              </div>
              <p className="text-xs text-slate-500 text-center mt-1">AI 감지 색상</p>
            </div>
            
            <div className="bg-white/80 backdrop-blur-sm p-4 rounded-xl shadow-md">
              <h4 className="text-xs mb-2 text-slate-600 font-semibold text-center">📍 위치</h4>
              <div className="text-sm text-slate-700 text-center space-y-1">
                <div className="font-mono">X: {selectedHold.center ? Math.round(selectedHold.center[0]) : 'N/A'}</div>
                <div className="font-mono">Y: {selectedHold.center ? Math.round(selectedHold.center[1]) : 'N/A'}</div>
              </div>
              {selectedHold.hsv && (
                <div className="text-xs font-mono text-slate-600 text-center mt-2">
                  HSV({selectedHold.hsv[0]}, {selectedHold.hsv[1]}, {selectedHold.hsv[2]})
                </div>
              )}
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-yellow-50 to-orange-50 p-3 rounded-xl border border-yellow-200">
              <h4 className="text-sm mb-2 text-slate-800 font-bold text-center">💬 색상 피드백</h4>
              <p className="text-xs text-slate-600 mb-3 text-center">
                AI가 예측한 색상이 맞나요? 피드백을 주시면 더 정확해집니다!
              </p>
              <button
                onClick={() => setShowHoldFeedbackModal(true)}
                className="w-full px-4 py-2 bg-gradient-to-r from-yellow-400 to-orange-500 text-white rounded-xl text-sm font-semibold shadow-md hover:shadow-lg transition-all"
              >
                🎨 색상 피드백 제출
              </button>
            </div>
            </div>
          </div>
        </div>
      )}

      {/* 선택된 문제 상세 */}
      {selectedProblem && selectedProblem.difficulty && (
        <div className="glass-card p-3 mx-auto mb-3 w-full text-center shadow-md">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-lg text-slate-800 font-bold flex-1">
              {colorEmoji[selectedProblem.color_name] || '⭕'} {(selectedProblem.color_name || 'UNKNOWN').toUpperCase()} 문제
            </h3>
            <button
              onClick={() => setShowFeedbackModal(true)}
              className="px-3 py-1.5 bg-gradient-to-r from-primary-500 to-purple-600 text-white rounded-lg text-xs font-semibold shadow-md hover:shadow-lg transition-all"
            >
              📝 피드백
            </button>
          </div>

          <div className="grid grid-cols-2 gap-3 mb-3">
            <div className="bg-white/80 backdrop-blur-sm p-3 rounded-xl shadow-md">
              <h4 className="text-xs mb-2 text-slate-600 font-semibold text-center">🎯 난이도</h4>
              <div className="text-2xl font-extrabold gradient-text text-center mb-0.5">{selectedProblem.difficulty || 'V?'}</div>
              <div className="text-xs text-slate-600 mb-1 font-medium text-center">{selectedProblem.type || '미분석'}</div>
              <div className="text-base text-yellow-400 text-center">
                {'★'.repeat(Math.floor((selectedProblem.difficulty?.confidence || 0) * 5))}
                {'☆'.repeat(5 - Math.floor((selectedProblem.difficulty?.confidence || 0) * 5))}
              </div>
            </div>

            <div className="bg-white/80 backdrop-blur-sm p-3 rounded-xl shadow-md">
              <h4 className="text-xs mb-2 text-slate-600 font-semibold text-center">🏋️ 유형</h4>
              <div className="text-base font-bold text-slate-800 mb-1.5 text-center">{selectedProblem.type || '일반'}</div>
              <div className="flex flex-wrap gap-1.5 justify-center">
                {selectedProblem.climb_type?.types?.slice(0, 3).map((type, idx) => (
                  <span key={idx} className="px-2 py-0.5 bg-gradient-to-r from-primary-500 to-purple-600 text-white rounded-full text-xs font-semibold">
                    {type}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="bg-white/80 backdrop-blur-sm p-3 rounded-xl shadow-md mb-3">
            <h4 className="text-xs mb-2 text-slate-800 font-bold text-center">📊 문제 정보</h4>
            <div className="flex justify-between items-center py-1 border-b border-slate-200 text-xs">
              <span className="text-slate-600">홀드 개수:</span>
              <span className="font-bold gradient-text">{selectedProblem.hold_count}개</span>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-slate-200 text-xs">
              <span className="text-slate-600">난이도:</span>
              <span className="font-bold gradient-text">{selectedProblem.difficulty || 'V?'}</span>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-slate-200 text-xs">
              <span className="text-slate-600">유형:</span>
              <span className="font-bold gradient-text">{selectedProblem.type || '일반'}</span>
            </div>
            <div className="flex justify-between items-center py-1 text-xs">
              <span className="text-slate-600">분석 방법:</span>
              <span className="font-bold gradient-text">{selectedProblem.reasoning ? 'GPT-4 AI' : '규칙 기반'}</span>
            </div>
          </div>

          {selectedProblem.reasoning && (
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-3 rounded-xl shadow-md border-2 border-blue-200">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-lg">🤖</span>
                <h4 className="text-xs text-slate-800 font-bold">GPT-4 AI 상세 분석</h4>
                <span className="ml-auto text-xs bg-blue-500 text-white px-1.5 py-0.5 rounded-full">
                  신뢰도: {Math.round((selectedProblem.gpt4_confidence || 0.8) * 100)}%
                </span>
              </div>
              <div className="text-xs text-slate-700 leading-relaxed whitespace-pre-line">
                {selectedProblem.reasoning}
              </div>
            </div>
          )}
        </div>
      )}

      {/* 문제 목록 */}
      {result.problems && result.problems.length > 0 && (
        <div className="w-full">
          <h3 className="text-2xl font-extrabold gradient-text mb-6 text-center">
            🎨 감지된 문제들
          </h3>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {result.problems.map((problem, idx) => (
              <div 
                key={idx}
                onClick={() => onProblemSelect(problem)}
                className={`glass-card p-6 rounded-2xl cursor-pointer transition-all duration-300 hover:translate-y-[-5px] hover:shadow-xl ${
                  selectedProblem && selectedProblem.id === problem.id 
                    ? 'ring-4 ring-primary-500 shadow-2xl' 
                    : 'shadow-lg'
                }`}
              >
                <div className="text-center">
                  <div className="text-6xl mb-3 drop-shadow-lg">{colorEmoji[problem.color_name] || '⭕'}</div>
                  <div className="text-xl font-bold gradient-text mb-2">{problem.color_name?.toUpperCase()}</div>
                  <div className="text-sm text-slate-600 mb-3">홀드 {problem.hold_count}개</div>
                  
                  {problem.difficulty && (
                    <div className="bg-white/80 backdrop-blur-sm p-3 rounded-xl shadow-inner mb-2">
                      <div className="text-2xl font-extrabold gradient-text mb-1">{problem.difficulty}</div>
                      <div className="text-xs text-slate-600 font-medium">{problem.type || '일반'}</div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default ResultDetails

