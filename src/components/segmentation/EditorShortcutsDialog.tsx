import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Keyboard, MousePointer, Move, Wand2, Pentagon, Sparkles } from "lucide-react";

interface EditorShortcutsDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ShortcutItem {
  keys: string[];
  description: string;
}

interface ShortcutGroup {
  title: string;
  icon: React.ReactNode;
  items: ShortcutItem[];
}

export const EditorShortcutsDialog: React.FC<EditorShortcutsDialogProps> = ({
  isOpen,
  onClose,
}) => {
  const groups: ShortcutGroup[] = [
    {
      title: "Canvas Navigation",
      icon: <Move className="h-4 w-4 text-teal-600" />,
      items: [
        { keys: ["Space + Drag"], description: "Hand Tool pan canvas" },
        { keys: ["Middle Click + Drag"], description: "Pan canvas anytime" },
        { keys: ["Left Click + Drag"], description: "Pan canvas (empty area in Select mode)" },
        { keys: ["Mouse Wheel", "Trackpad Pinch"], description: "Smooth zoom anchored to cursor" },
        { keys: ["F", "0"], description: "Fit image to screen" },
      ],
    },
    {
      title: "Tool Modes",
      icon: <MousePointer className="h-4 w-4 text-teal-600" />,
      items: [
        { keys: ["V"], description: "Select & Edit Vertices mode" },
        { keys: ["W"], description: "SAM Smart Click mode" },
        { keys: ["P"], description: "Manual Polygon Draw mode" },
      ],
    },
    {
      title: "Coral & Vertex Editing",
      icon: <Sparkles className="h-4 w-4 text-teal-600" />,
      items: [
        { keys: ["Click"], description: "Select coral polygon or vertex" },
        { keys: ["Drag Vertex"], description: "Reposition boundary vertex" },
        { keys: ["Click Edge / (+)"], description: "Insert new vertex along edge" },
        { keys: ["Right Click", "Alt + Click"], description: "Delete hovered vertex" },
        { keys: ["Delete", "Backspace"], description: "Delete selected vertex or coral" },
        { keys: ["Esc"], description: "Deselect active vertex or coral" },
      ],
    },
    {
      title: "SAM Smart Click (W)",
      icon: <Wand2 className="h-4 w-4 text-teal-600" />,
      items: [
        { keys: ["Click"], description: "Add foreground (+) coral pin" },
        { keys: ["Alt + Click", "Shift + Click"], description: "Add background (−) exclusion pin" },
        { keys: ["Enter", "Space"], description: "Run SAM AI model on placed pins" },
        { keys: ["Backspace"], description: "Undo last prompt pin" },
        { keys: ["Esc"], description: "Discard pending mask / clear pins" },
      ],
    },
    {
      title: "Polygon Drawing (P)",
      icon: <Pentagon className="h-4 w-4 text-teal-600" />,
      items: [
        { keys: ["Click"], description: "Add polygon vertex" },
        { keys: ["Enter", "Click Start Point"], description: "Close and commit polygon" },
        { keys: ["Backspace"], description: "Undo last drawn point" },
        { keys: ["Esc"], description: "Cancel drawing in progress" },
      ],
    },
    {
      title: "General & Workflow",
      icon: <Keyboard className="h-4 w-4 text-teal-600" />,
      items: [
        { keys: ["⌘S", "Ctrl+S"], description: "Save annotations (auto-saves every 20s)" },
        { keys: ["⌘Z", "Ctrl+Z"], description: "Undo last change" },
        { keys: ["⌘⇧Z", "Ctrl+Y"], description: "Redo change" },
        { keys: ["[", "]"], description: "Navigate previous / next image" },
        { keys: ["?", "/"], description: "Toggle this shortcut guide" },
      ],
    },
  ];

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto p-6 sm:rounded-2xl border-slate-200">
        <DialogHeader className="pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-teal-50 text-teal-700">
              <Keyboard className="h-5 w-5" />
            </div>
            <div>
              <DialogTitle className="text-base font-bold text-slate-900">
                Segmentation Editor Controls & Shortcuts
              </DialogTitle>
              <DialogDescription className="text-xs text-slate-500">
                Quick reference for fluid canvas navigation, vertex editing, and AI tools.
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          {groups.map((group, idx) => (
            <div
              key={idx}
              className="bg-slate-50/70 border border-slate-200/80 rounded-xl p-3.5 flex flex-col gap-2.5"
            >
              <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800 uppercase tracking-wider">
                {group.icon}
                <span>{group.title}</span>
              </div>

              <div className="flex flex-col gap-1.5">
                {group.items.map((item, itemIdx) => (
                  <div
                    key={itemIdx}
                    className="flex items-center justify-between text-xs py-0.5 gap-2"
                  >
                    <span className="text-slate-600 text-[11px] leading-snug">
                      {item.description}
                    </span>
                    <div className="flex items-center gap-1 shrink-0">
                      {item.keys.map((k, kIdx) => (
                        <kbd
                          key={kIdx}
                          className="px-1.5 py-0.5 text-[10px] font-mono font-semibold bg-white border border-slate-300 rounded shadow-2xs text-slate-700"
                        >
                          {k}
                        </kbd>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
};
