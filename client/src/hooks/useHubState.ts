import { useCallback, useState } from "react";

import type {
  HubBundleBuiltPayload,
  HubDownloadState,
  HubImportPayload,
  HubInspectPayload,
  HubListPayload,
  HubPullState,
  HubSearchPayload,
} from "../hubTypes";
import type { ServerMsg } from "../types";

/** Own the hub browser state and fold hub server messages into it. */
export function useHubState() {
  const [list, setList] = useState<HubListPayload | null>(null);
  const [search, setSearch] = useState<HubSearchPayload | null>(null);
  const [download, setDownload] = useState<HubDownloadState | null>(null);
  const [inspect, setInspect] = useState<HubInspectPayload | null>(null);
  const [imported, setImported] = useState<HubImportPayload | null>(null);
  const [pull, setPull] = useState<HubPullState | null>(null);
  const [built, setBuilt] = useState<HubBundleBuiltPayload | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  /** Fold one server message into the hub state; other types pass through. */
  const onMessage = useCallback((msg: ServerMsg) => {
    switch (msg.type) {
      case "hub_list":
        setList(msg.payload);
        break;
      case "hub_search":
        setSearch(msg.payload);
        break;
      case "hub_download_state":
        setDownload(msg.payload);
        break;
      case "hub_inspect":
        setInspect(msg.payload);
        break;
      case "hub_import":
        setImported(msg.payload);
        break;
      case "hub_pull_state":
        setPull(msg.payload);
        break;
      case "bundle_built":
        setBuilt(msg.payload);
        break;
      default:
        break;
    }
  }, []);

  return {
    list,
    search,
    download,
    inspect,
    imported,
    pull,
    built,
    selected,
    setSearch,
    setDownload,
    setInspect,
    setImported,
    setPull,
    setSelected,
    onMessage,
  };
}
