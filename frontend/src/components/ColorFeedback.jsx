/**
 * 🎨 색상 피드백 컴포넌트
 * 사용자가 잘못 분류된 홀드의 색상을 수정할 수 있음
 */

import { useState } from 'react'
import './ColorFeedback.css'

const COLOR_OPTIONS = [
  { value: 'black', label: '검정', color: '#000000' },
  { value: 'white', label: '흰색', color: '#FFFFFF' },
  { value: 'red', label: '빨강', color: '#FF0000' },
  { value: 'orange', label: '주황', color: '#FF8800' },
  { value: 'yellow', label: '노랑', color: '#FFFF00' },
  { value: 'green', label: '초록', color: '#00FF00' },
  { value: 'mint', label: '민트', color: '#00FFCC' },
  { value: 'blue', label: '파랑', color: '#0000FF' },
  { value: 'purple', label: '보라', color: '#8800FF' },
  { value: 'pink', label: '분홍', color: '#FF88CC' },
  { value: 'brown', label: '갈색', color: '#8B4513' },
]

export default function ColorFeedback({ problems, imageUrl, onFeedbackSubmit }) {
  const [feedbackMode, setFeedbackMode] = useState(false)
  const [feedbacks, setFeedbacks] = useState([])
  const [selectedHold, setSelectedHold] = useState(null)

  const handleColorChange = (holdId, problemId, oldColor, newColor) => {
    if (oldColor === newColor) return

    const feedback = {
      hold_id: holdId,
      predicted_color: oldColor,
      correct_color: newColor,
      problem_id: problemId,
      timestamp: new Date().toISOString()
    }

    setFeedbacks(prev => {
      // 같은 홀드의 기존 피드백 제거
      const filtered = prev.filter(f => f.hold_id !== holdId)
      return [...filtered, feedback]
    })
  }

  const handleSubmit = async () => {
    if (feedbacks.length === 0) {
      alert('수정된 색상이 없습니다.')
      return
    }

    try {
      const response = await fetch('/api/color-feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ feedbacks })
      })

      if (response.ok) {
        alert(`✅ ${feedbacks.length}개의 피드백이 저장되었습니다!\n다음 분석부터 개선된 색상 분류가 적용됩니다.`)
        setFeedbacks([])
        setFeedbackMode(false)
        if (onFeedbackSubmit) onFeedbackSubmit(feedbacks)
      } else {
        throw new Error('피드백 저장 실패')
      }
    } catch (error) {
      console.error('피드백 전송 오류:', error)
      alert('❌ 피드백 저장 중 오류가 발생했습니다.')
    }
  }

  const toggleFeedbackMode = () => {
    setFeedbackMode(!feedbackMode)
    if (feedbackMode) {
      setFeedbacks([])
    }
  }

  return (
    <div className="color-feedback-container">
      {/* 피드백 모드 토글 버튼 */}
      <div className="feedback-header">
        <button 
          className={`feedback-toggle ${feedbackMode ? 'active' : ''}`}
          onClick={toggleFeedbackMode}
        >
          {feedbackMode ? '✅ 피드백 모드 종료' : '✏️ 색상 수정하기'}
        </button>
        
        {feedbackMode && feedbacks.length > 0 && (
          <div className="feedback-actions">
            <span className="feedback-count">
              수정: {feedbacks.length}개
            </span>
            <button 
              className="feedback-submit"
              onClick={handleSubmit}
            >
              💾 피드백 저장
            </button>
            <button 
              className="feedback-cancel"
              onClick={() => setFeedbacks([])}
            >
              취소
            </button>
          </div>
        )}
      </div>

      {/* 피드백 모드 안내 */}
      {feedbackMode && (
        <div className="feedback-info">
          <p>💡 잘못 분류된 홀드의 올바른 색상을 선택해주세요. 피드백은 AI 학습에 사용됩니다.</p>
        </div>
      )}

      {/* 문제별 홀드 표시 */}
      {feedbackMode && (
        <div className="problems-grid">
          {Object.entries(problems).map(([problemId, problem]) => (
            <div key={problemId} className="problem-card">
              <div className="problem-header">
                <div 
                  className="color-badge" 
                  style={{ backgroundColor: problem.color_name }}
                >
                  {problem.holds?.length || 0}개
                </div>
                <h3>{problem.color_display || problem.color_name}</h3>
                {problem.confidence && (
                  <span className="confidence">
                    {(problem.confidence * 100).toFixed(0)}%
                  </span>
                )}
              </div>

              <div className="holds-list">
                {problem.holds?.map((hold, idx) => (
                  <HoldFeedbackItem
                    key={hold.id || idx}
                    hold={hold}
                    problemId={problemId}
                    currentColor={problem.color_name}
                    onColorChange={handleColorChange}
                    isModified={feedbacks.some(f => f.hold_id === hold.id)}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function HoldFeedbackItem({ hold, problemId, currentColor, onColorChange, isModified }) {
  const [selectedColor, setSelectedColor] = useState(currentColor)

  const handleChange = (newColor) => {
    setSelectedColor(newColor)
    onColorChange(hold.id, problemId, currentColor, newColor)
  }

  return (
    <div className={`hold-feedback-item ${isModified ? 'modified' : ''}`}>
      <div className="hold-info">
        <span className="hold-id">홀드 #{hold.id}</span>
        {hold.clip_confidence && (
          <span className={`hold-confidence ${hold.clip_confidence < 0.7 ? 'low' : ''}`}>
            {(hold.clip_confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>

      <select 
        value={selectedColor}
        onChange={(e) => handleChange(e.target.value)}
        className="color-select"
      >
        {COLOR_OPTIONS.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>

      {isModified && (
        <span className="modified-badge">✏️</span>
      )}
    </div>
  )
}

