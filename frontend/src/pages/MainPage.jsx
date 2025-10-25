const MainPage = ({ 
  preview, 
  handleImageUpload, 
  handleCameraCapture,
  analyzeImage
}) => {
  return (
    <>
      {/* 히어로 섹션 (이미지 없을 때) - 좌우 분할 */}
      {!preview && (
        <div className="flex flex-col lg:flex-row gap-6 items-center">
          {/* 좌측: 히어로 */}
          <div className="w-full lg:w-1/2">
            <div className="glass-card p-6 lg:p-8">
              <div className="text-5xl lg:text-7xl mb-4 animate-bounce-slow text-center">🧗‍♀️</div>
              <h2 className="text-xl lg:text-2xl font-bold gradient-text mb-3 text-center">
                AI가 클라이밍 문제를 분석합니다
              </h2>
              <p className="text-sm text-slate-600 mb-4 text-center">
                클라이밍 벽 사진을 업로드하면 AI가 홀드를 감지하고
                난이도와 유형을 자동으로 분석해드립니다
              </p>
              
              <div className="grid grid-cols-3 gap-3">
                <div className="glass-card p-3 text-center">
                  <div className="text-2xl mb-1">🎯</div>
                  <div className="text-xs font-bold text-slate-800">정확한 분석</div>
                </div>
                <div className="glass-card p-3 text-center">
                  <div className="text-2xl mb-1">⚡</div>
                  <div className="text-xs font-bold text-slate-800">빠른 처리</div>
                </div>
                <div className="glass-card p-3 text-center">
                  <div className="text-2xl mb-1">📊</div>
                  <div className="text-xs font-bold text-slate-800">상세 정보</div>
                </div>
              </div>
            </div>
          </div>
          
          {/* 우측: 업로드/촬영 */}
          <div className="w-full lg:w-1/2">
            <div className="glass-card p-6 lg:p-8">
              <h3 className="text-lg font-bold gradient-text mb-4 text-center">
                📷 시작하기
              </h3>
              <div className="flex flex-col gap-3">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  id="file-input"
                  className="hidden"
                />
                
                <label 
                  htmlFor="file-input" 
                  className="glass-button inline-flex items-center gap-2 px-6 py-4 text-slate-800 rounded-xl text-base font-semibold cursor-pointer shadow-lg justify-center hover:shadow-xl hover:scale-105 transition-all"
                >
                  📁 사진 업로드
                </label>
                
                <button
                  onClick={handleCameraCapture}
                  className="glass-button inline-flex items-center gap-2 px-6 py-4 text-slate-800 rounded-xl text-base font-semibold cursor-pointer shadow-lg justify-center hover:shadow-xl hover:scale-105 transition-all"
                >
                  📸 카메라 촬영
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 이미지 업로드 후 (분석 전) - 재업로드 가능 */}
      {preview && (
        <div className="flex flex-col lg:flex-row gap-6 items-start">
          {/* 좌측: 업로드된 이미지 */}
          <div className="w-full lg:w-1/2">
            <div className="relative">
              <img 
                src={preview} 
                alt="Uploaded" 
                className="w-full max-h-[400px] lg:max-h-[600px] object-contain rounded-2xl shadow-2xl"
              />
            </div>
          </div>
          
          {/* 우측: 분석 시작 & 재업로드 */}
          <div className="w-full lg:w-1/2">
            <div className="glass-card p-6 lg:p-8">
              <h3 className="text-lg font-bold gradient-text mb-4 text-center">
                ✅ 이미지 업로드 완료
              </h3>
              <p className="text-sm text-slate-600 mb-6 text-center">
                이미지를 확인하시고 분석을 시작하거나<br/>
                다른 이미지를 선택하세요
              </p>
              
              <div className="flex flex-col gap-3">
                <button
                  onClick={analyzeImage}
                  className="w-full px-6 py-4 bg-gradient-to-r from-primary-500 to-purple-600 text-white rounded-xl text-base font-semibold shadow-lg hover:shadow-xl hover:scale-105 transition-all"
                >
                  🚀 분석 시작
                </button>
                
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-slate-300"></div>
                  </div>
                  <div className="relative flex justify-center text-xs">
                    <span className="px-2 bg-white text-slate-500">또는</span>
                  </div>
                </div>
                
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  id="file-input-reupload"
                  className="hidden"
                />
                
                <label 
                  htmlFor="file-input-reupload" 
                  className="glass-button inline-flex items-center gap-2 px-4 py-3 text-slate-800 rounded-xl text-sm font-semibold cursor-pointer shadow-md justify-center hover:shadow-lg hover:scale-105 transition-all"
                >
                  📁 다른 사진 선택
                </label>
                
                <button
                  onClick={handleCameraCapture}
                  className="glass-button inline-flex items-center gap-2 px-4 py-3 text-slate-800 rounded-xl text-sm font-semibold cursor-pointer shadow-md justify-center hover:shadow-lg hover:scale-105 transition-all"
                >
                  📸 다시 촬영
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

export default MainPage

