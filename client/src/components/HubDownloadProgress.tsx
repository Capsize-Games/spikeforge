import { useState } from "react";

import { ConfirmDialog } from "./ConfirmDialog";
import type { HubDownloadState } from "../hubTypes";

interface Props {
  download: HubDownloadState;
  onCancel: () => void;
}

function megabytes(bytes: number): string {
  return (bytes / 1024 ** 2).toFixed(1);
}

/** Hub download progress with cancel and an honest verified state. */
export function HubDownloadProgress({ download, onCancel }: Props) {
  const [confirmCancel, setConfirmCancel] = useState(false);
  const total = download.total_bytes;
  const progress =
    total !== null && total > 0
      ? `${megabytes(download.bytes)} / ${megabytes(total)} MB`
      : `${megabytes(download.bytes)} MB`;

  return (
    <div className="hub-download" role="status" aria-live="polite">
      <div className="hub-download-title">
        Downloading {download.id || "model"}…
      </div>
      <div className="hub-download-bar" aria-hidden="true">
        <span className="hub-download-fill" />
      </div>
      <div className="hub-download-meta">
        <span>{progress}</span>
        <span
          className={`hub-verified ${download.verified ? "ok" : "off"}`}
        >
          {download.verified ? "verified" : "unverified"}
        </span>
        {download.status === "downloading" && (
          <button
            type="button"
            className="link danger"
            onClick={() => setConfirmCancel(true)}
          >
            Cancel
          </button>
        )}
      </div>

      <ConfirmDialog
        open={confirmCancel}
        title="Cancel this download?"
        message={`Progress on ${
          download.id || "this model"
        } will be lost and it will need to restart from scratch.`}
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
