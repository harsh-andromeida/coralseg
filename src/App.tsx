import React, { useState } from "react";
import { UploadView } from "./views/UploadView";
import { AnalysisView } from "./views/AnalysisView";
import { VideoAnalysisView } from "./views/VideoAnalysisView";
import { LoadedImage, UploadedVideoItem } from "./types";
import "./styles/app.css";

export const App: React.FC = () => {
  const [appStage, setAppStage] = useState<"upload" | "analysis" | "video_analysis">("upload");
  const [loadedImages, setLoadedImages] = useState<LoadedImage[]>([]);
  const [activeVideo, setActiveVideo] = useState<UploadedVideoItem | null>(null);
  const [activeVideoJobId, setActiveVideoJobId] = useState<string>("");

  const handleLaunchStudio = (images: LoadedImage[]) => {
    setLoadedImages(images);
    setActiveVideo(null);
    setAppStage("analysis");
  };

  const handleLaunchVideoStudio = (videoItem: UploadedVideoItem, jobId?: string) => {
    setActiveVideo(videoItem);
    setActiveVideoJobId(jobId || videoItem.job_id || "");
    setAppStage("video_analysis");
  };

  const handleBackToUpload = () => {
    setAppStage("upload");
  };

  if (appStage === "video_analysis" && activeVideo) {
    return (
      <VideoAnalysisView
        videoItem={activeVideo}
        initialJobId={activeVideoJobId}
        onBackToUpload={handleBackToUpload}
      />
    );
  }

  if (appStage === "analysis" && loadedImages.length > 0) {
    return (
      <AnalysisView
        images={loadedImages}
        onBackToUpload={handleBackToUpload}
      />
    );
  }

  return (
    <UploadView
      onLaunchStudio={handleLaunchStudio}
      onLaunchVideoStudio={handleLaunchVideoStudio}
    />
  );
};

export default App;
