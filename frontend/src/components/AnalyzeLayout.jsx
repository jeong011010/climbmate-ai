import ImageViewer from './ImageViewer'
import ResultDetails from './ResultDetails'

const AnalyzeLayout = ({ 
  preview,
  annotatedImage,
  result,
  selectedProblem,
  selectedHold,
  imageLoaded,
  setImageLoaded,
  setSelectedHold,
  setShowHoldFeedbackModal,
  setShowFeedbackModal,
  setShowImageModal,
  handleImageClick,
  imageRef,
  onProblemSelect,
  colorEmoji
}) => {
  return (
    <>
      {/* 모바일: 세로 스크롤 / PC: 좌우 분할 */}
      <div className={result ? "flex flex-col lg:flex-row gap-6 items-start" : ""}>
        {/* 좌측: 이미지 영역 (PC에서 sticky) */}
        {preview && (
          <div className={result ? "w-full lg:w-1/2 lg:sticky lg:top-24 lg:self-start" : "w-full"}>
            <ImageViewer
              preview={preview}
              annotatedImage={annotatedImage}
              result={result}
              selectedProblem={selectedProblem}
              selectedHold={selectedHold}
              imageLoaded={imageLoaded}
              setImageLoaded={setImageLoaded}
              onImageClick={handleImageClick}
              onImageDoubleClick={() => setShowImageModal(true)}
              imageRef={imageRef}
              colorEmoji={colorEmoji}
            />
          </div>
        )}

        {/* 우측: 결과 영역 */}
        {result && (
          <div className="w-full lg:w-1/2">
            <ResultDetails
              result={result}
              selectedHold={selectedHold}
              selectedProblem={selectedProblem}
              setSelectedHold={setSelectedHold}
              setShowHoldFeedbackModal={setShowHoldFeedbackModal}
              setShowFeedbackModal={setShowFeedbackModal}
              colorEmoji={colorEmoji}
              onProblemSelect={onProblemSelect}
            />
          </div>
        )}
      </div>
    </>
  )
}

export default AnalyzeLayout

