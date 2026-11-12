import React from "react";
import { Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";

export interface RemoteBackendModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  remoteUrlInput: string;
  onChangeRemoteUrl: (val: string) => void;
  onSave: () => void;
  onResetToLocal: () => void;
}

export const RemoteBackendModal: React.FC<RemoteBackendModalProps> = ({
  open,
  onOpenChange,
  remoteUrlInput,
  onChangeRemoteUrl,
  onSave,
  onResetToLocal,
}) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md p-6">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-teal-600" />
            <DialogTitle className="text-base font-bold text-[#0f1e4a]">
              Connect to Remote / Colab Backend
            </DialogTitle>
          </div>
          <DialogDescription className="text-xs text-slate-500 pt-1">
            Connect this desktop app to a Google Colab GPU instance or remote server exposed via Cloudflare Tunnel or ngrok.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 my-3">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Backend Server URL
            </label>
            <input
              type="text"
              value={remoteUrlInput}
              onChange={(e) => onChangeRemoteUrl(e.target.value)}
              placeholder="https://your-tunnel-name.trycloudflare.com"
              className="w-full text-xs px-3 py-2 rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-teal-500/20 font-mono"
            />
            <p className="text-[10px] text-slate-500 mt-1">
              Leave empty or click Reset to use the local machine backend (<code>http://127.0.0.1:8000</code>).
            </p>
          </div>
        </div>

        <DialogFooter className="flex items-center justify-between sm:justify-between mt-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onResetToLocal}
            className="text-xs text-slate-600 hover:bg-slate-100"
          >
            Reset to Local
          </Button>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => onOpenChange(false)}
              className="text-xs"
            >
              Cancel
            </Button>
            <Button
              type="button"
              size="sm"
              onClick={onSave}
              className="text-xs bg-teal-600 hover:bg-teal-700 text-white font-medium"
            >
              Save & Connect
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
