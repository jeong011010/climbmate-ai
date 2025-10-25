/**
 * 이미지 업로드 및 카메라 캡처 관련 Hook
 */
export const useImageUpload = ({
  setImage,
  setPreview,
  setResult,
  setSelectedProblem,
  setSelectedHold,
  setAnnotatedImage,
  setShowControlPanel,
  setImageLoaded
}) => {
  
  const handleImageUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setImage(file)
      const reader = new FileReader()
      reader.onload = (e) => setPreview(e.target.result)
      reader.readAsDataURL(file)
      setResult(null)
      setSelectedProblem(null)
      setSelectedHold(null)
      setAnnotatedImage(null)
      setShowControlPanel(true)
      setImageLoaded(false)
    }
  }

  const handleCameraCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        } 
      })
      
      const video = document.createElement('video')
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      
      video.srcObject = stream
      video.play()
      
      // 카메라 모달 생성
      const modal = document.createElement('div')
      modal.className = 'fixed inset-0 bg-black bg-opacity-90 z-50 flex flex-col items-center justify-center'
      modal.innerHTML = `
        <div class="bg-white p-4 rounded-t-2xl w-full max-w-md">
          <video class="w-full rounded-lg" autoplay playsinline></video>
          <div class="flex gap-4 mt-4">
            <button id="capture-btn" class="flex-1 bg-primary-500 text-white py-3 rounded-xl font-semibold">
              📸 촬영
            </button>
            <button id="cancel-btn" class="flex-1 bg-gray-300 text-gray-700 py-3 rounded-xl font-semibold">
              취소
            </button>
          </div>
        </div>
      `
      
      document.body.appendChild(modal)
      const videoEl = modal.querySelector('video')
      videoEl.srcObject = stream
      
      modal.querySelector('#capture-btn').onclick = () => {
        canvas.width = videoEl.videoWidth
        canvas.height = videoEl.videoHeight
        ctx.drawImage(videoEl, 0, 0)
        
        canvas.toBlob((blob) => {
          const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' })
          setImage(file)
          setPreview(URL.createObjectURL(blob))
          setResult(null)
          setSelectedProblem(null)
          setSelectedHold(null)
          setAnnotatedImage(null)
          setShowControlPanel(true)
          setImageLoaded(false)
          
          stream.getTracks().forEach(track => track.stop())
          document.body.removeChild(modal)
        }, 'image/jpeg', 0.9)
      }
      
      modal.querySelector('#cancel-btn').onclick = () => {
        stream.getTracks().forEach(track => track.stop())
        document.body.removeChild(modal)
      }
      
    } catch (error) {
      console.error('카메라 접근 실패:', error)
      alert('카메라에 접근할 수 없습니다. 파일 업로드를 사용해주세요.')
    }
  }

  return {
    handleImageUpload,
    handleCameraCapture
  }
}

