import React, { useState } from "react";
import { UploadView } from "./views/UploadView";
import { AnalysisView } from "./views/AnalysisView";
import { LoadedImage } from "./types";
import "./styles/app.css";

export const App: React.FC = () => {
  const [appStage, setAppStage] = useState<"upload" | "analysis">("upload");
  const [loadedImages, setLoadedImages] = useState<LoadedImage[]>([]);

  const handleLaunchStudio = (images: LoadedImage[]) => {
    setLoadedImages(images);
    setAppStage("analysis");
  };

  const handleBackToUpload = () => {
    setAppStage("upload");
  };

  if (appStage === "upload" || loadedImages.length === 0) {
    return <UploadView onLaunchStudio={handleLaunchStudio} />;
  }

  return (
    <AnalysisView
      images={loadedImages}
      onBackToUpload={handleBackToUpload}
    />
  );
};

export default App;
