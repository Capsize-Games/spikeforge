import { useState } from "react";

import { ConfirmDialog } from "./ConfirmDialog";
import type { DownloadState } from "../types";

interface Props {
  download: DownloadState;
  onCancel: () => void;
}

function megabytes(bytes: number): string {
  return (bytes / 1024 ** 2).toFixed(1);
}

/** Blocking overlay shown while a dataset is downloaded on the server. */
export function DownloadProgress({ download, onCancel }: Props) {
  const [confirmCancel, setConfirmCancel] = useState(false);

  return (
    <div className="download-overlay" role="status" aria-live="polite">
      <div className="download-card">
        <div className="download-title">
          Downloading {download.dataset}…
        </div>
        <div className="download-bar" aria-hidden="true">
          <span className="download-bar-fill" />
        </div>
        <div className="download-meta">
          <span>{megabytes(download.bytes)} MB</span>
          <button
            type="button"
            className="link danger"
            onClick={() => setConfirmCancel(true)}
          >
            Cancel
          </button>
        </div>
      </div>

      <ConfirmDialog
        open={confirmCancel}
        title="Cancel this download?"
        message={`Progress on ${download.dataset} will be lost and it will need to restart from scratch.`}
        confirmLabel="Cancel download"
        cancelLabel="Keep downloading"
        onConfirm={() => {
          onCancel();
          setConfirmCancel(false);
        }}
        onCancel={() => setConfirmCancel(false)}
      />
    </div>
  );
}
