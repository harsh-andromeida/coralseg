import { ModelSpec } from "../../types";

export interface ModelGroupDef {
  id: "sam" | "bioclip" | "bleaching";
  name: string;
  fullName: string;
  task: string;
  size: string;
  filenames: string[];
}

export const MODEL_GROUPS: ModelGroupDef[] = [
  {
    id: "sam",
    name: "Segmentation Model",
    fullName: "CoralSCOP Segmentation Model",
    task: "Dense Instance Segmentation",
    size: "~383 MB",
    filenames: ["sam_image_encoder.onnx", "sam_mask_decoder.onnx"],
  },
  {
    id: "bioclip",
    name: "Taxonomical Model",
    fullName: "BioCLIP Taxonomy Model",
    task: "Taxonomy & Growth Form",
    size: "~345 MB",
    filenames: ["bioclip_visual.onnx", "coral_taxonomy_embeddings.npy"],
  },
  {
    id: "bleaching",
    name: "Bleach Detection Model",
    fullName: "NOAA Bleach Detection Model",
    task: "Reef Health & Bleaching",
    size: "~6.1 MB",
    filenames: ["bleaching_yolo11n.onnx"],
  },
];

export const isGroupReady = (
  groupId: string,
  allDownloaded: boolean,
  modelsList: ModelSpec[]
): boolean => {
  if (allDownloaded) return true;
  const group = MODEL_GROUPS.find((g) => g.id === groupId);
  if (!group) return false;
  return group.filenames.every((fname) => {
    const item = modelsList.find((m) => m.filename === fname || m.filename.includes(fname));
    return item?.cached ?? false;
  });
};
