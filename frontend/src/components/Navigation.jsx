const Navigation = ({ currentView, setCurrentView, loadColorFeedbacks }) => {
  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white/95 backdrop-blur-md border-t border-slate-200 shadow-lg z-50">
      <div className="grid grid-cols-4 max-w-screen-lg mx-auto">
        <button
          onClick={() => setCurrentView('analyze')}
          className={`flex flex-col items-center justify-center py-3 transition-all ${
            currentView === 'analyze'
              ? 'text-blue-600'
              : 'text-slate-600 hover:text-blue-500'
          }`}
        >
          <span className="text-2xl mb-1">📸</span>
          <span className="text-xs font-medium">분석</span>
        </button>
        
        <button
          onClick={() => setCurrentView('history')}
          className={`flex flex-col items-center justify-center py-3 transition-all ${
            currentView === 'history'
              ? 'text-blue-600'
              : 'text-slate-600 hover:text-blue-500'
          }`}
        >
          <span className="text-2xl mb-1">📚</span>
          <span className="text-xs font-medium">히스토리</span>
        </button>
        
        <button
          onClick={() => setCurrentView('stats')}
          className={`flex flex-col items-center justify-center py-3 transition-all ${
            currentView === 'stats'
              ? 'text-blue-600'
              : 'text-slate-600 hover:text-blue-500'
          }`}
        >
          <span className="text-2xl mb-1">📊</span>
          <span className="text-xs font-medium">통계</span>
        </button>
        
        {/* 피드백 탭 */}
        <button
          onClick={() => {
            setCurrentView('feedbacks')
            loadColorFeedbacks()  // 탭 전환 시 피드백 로드
          }}
          className={`flex flex-col items-center justify-center py-3 transition-all ${
            currentView === 'feedbacks'
              ? 'text-blue-600'
              : 'text-slate-600 hover:text-blue-500'
          }`}
        >
          <span className="text-2xl mb-1">🎨</span>
          <span className="text-xs font-medium">피드백</span>
        </button>
      </div>
    </div>
  )
}

export default Navigation

