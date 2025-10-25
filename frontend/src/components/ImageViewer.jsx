import { useRef } from 'react'

const ImageViewer = ({ 
  preview, 
  annotatedImage, 
  result, 
  selectedProblem,
  selectedHold,
  imageLoaded,
  setImageLoaded,
  onImageClick,
  onImageDoubleClick,
  colorEmoji 
}) => {
  const imageRef = useRef(null)

  if (!preview) return null

  return (
    <div className="relative w-full">
      <div className="relative w-full flex justify-center">
        <div className="relative" style={{ display: 'inline-block' }}>
          <img 
            ref={imageRef}
            src={annotatedImage || preview} 
            alt="Climbing Wall" 
            className={`max-h-[400px] lg:max-h-[600px] w-full object-contain rounded-2xl shadow-2xl border border-white/20 ${
              result ? 'cursor-pointer hover:opacity-90 transition-opacity' : ''
            }`}
            onClick={result ? onImageClick : undefined}
            onTouchEnd={result ? onImageClick : undefined}
            onDoubleClick={result ? onImageDoubleClick : undefined}
            onLoad={() => {
              setImageLoaded(true)
              console.log('Image loaded:', imageRef.current?.naturalWidth, 'x', imageRef.current?.naturalHeight)
            }}
          />
          
          {/* SVG 오버레이 - 선택된 문제의 홀드들 강조 */}
          {result && selectedProblem && imageRef.current && imageLoaded && (() => {
            const img = imageRef.current
            const rect = img.getBoundingClientRect()
            const scaleX = rect.width / img.naturalWidth
            const scaleY = rect.height / img.naturalHeight
            
            return (
              <svg
                className="absolute top-0 left-0 pointer-events-none"
                style={{
                  width: rect.width + 'px',
                  height: rect.height + 'px'
                }}
                viewBox={`0 0 ${img.naturalWidth} ${img.naturalHeight}`}
                preserveAspectRatio="xMidYMid meet"
              >
                {selectedProblem.holds?.map((hold, idx) => {
                  if (!hold.contour || hold.contour.length === 0) {
                    return null
                  }
                  
                  const isSelectedHold = selectedHold && 
                    selectedHold.center && hold.center &&
                    selectedHold.center[0] === hold.center[0] && 
                    selectedHold.center[1] === hold.center[1]
                  
                  const points = hold.contour.map(pt => `${pt[0]},${pt[1]}`).join(' ')
                  
                  return (
                    <g key={idx}>
                      {/* 세그먼테이션 윤곽선 */}
                      <polygon
                        points={points}
                        fill="none"
                        stroke={isSelectedHold ? "#FFD700" : "#00FF00"}
                        strokeWidth={isSelectedHold ? "4" : "2"}
                        strokeDasharray={isSelectedHold ? "6,3" : "none"}
                        opacity="0.8"
                      >
                        {isSelectedHold && (
                          <animate
                            attributeName="opacity"
                            values="0.8;1;0.8"
                            dur="1s"
                            repeatCount="indefinite"
                          />
                        )}
                      </polygon>
                      
                      {/* 홀드 중심점 */}
                      {hold.center && (
                        <circle
                          cx={hold.center[0]}
                          cy={hold.center[1]}
                          r={isSelectedHold ? "8" : "5"}
                          fill={isSelectedHold ? "#FFD700" : "#00FF00"}
                          opacity="0.9"
                        >
                          {isSelectedHold && (
                            <animate
                              attributeName="r"
                              values="8;12;8"
                              dur="1s"
                              repeatCount="indefinite"
                            />
                          )}
                        </circle>
                      )}
                    </g>
                  )
                })}
              </svg>
            )
          })()}
          
          {result && selectedProblem && (
            <div className="absolute top-2 right-2 px-4 py-2 bg-gradient-to-r from-primary-500 to-purple-600 text-white rounded-lg text-sm font-bold shadow-lg z-10">
              {colorEmoji[selectedProblem.color_name] || '⭕'} {(selectedProblem.color_name || 'UNKNOWN').toUpperCase()}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ImageViewer

