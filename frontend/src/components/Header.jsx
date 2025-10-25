const Header = ({ modelStats, checkGpt4Status, testGpt4 }) => {
  return (
    <div className="fixed top-0 left-0 right-0 bg-white/95 backdrop-blur-md border-b border-slate-200 shadow-sm z-40">
      <div className="w-full px-2 sm:px-4 py-2 sm:py-3">
        <div className="text-center text-slate-800">
          <h1 className="text-xl sm:text-3xl font-extrabold gradient-text">
            🧗‍♀️ ClimbMate
          </h1>
          <p className="text-xs sm:text-sm opacity-70 font-medium">
            AI 기반 클라이밍 문제 분석
          </p>
          {modelStats && modelStats.verified_problems > 0 && (
            <div className="mt-3 text-xs text-slate-600">
              📊 학습 데이터: {modelStats.verified_problems}개
              {modelStats.ready_for_training && (
                <span className="ml-2 text-green-600 font-bold">✅ AI 학습 가능</span>
              )}
            </div>
          )}
          
          {/* GPT-4 디버깅 버튼 (개발용) */}
          <div className="mt-2 flex justify-center gap-2">
            <button
              onClick={checkGpt4Status}
              className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200 transition-colors"
            >
              🔍 GPT-4 상태
            </button>
            <button
              onClick={testGpt4}
              className="px-3 py-1 text-xs bg-green-100 text-green-700 rounded-full hover:bg-green-200 transition-colors"
            >
              🧪 GPT-4 테스트
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Header

